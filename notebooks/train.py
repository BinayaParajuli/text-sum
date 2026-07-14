# %% [markdown]
# # Nepali Abstractive Summarization — LoRA Fine-Tuning (mT5-base / mBART-large-50)
# 
# Run on **Google Colab with a GPU** (`Runtime > Change runtime type`; T4 is enough, L4/A100 are faster).
# **Do not run locally** — it needs a CUDA GPU.
# 
# This is the *single, config-driven* training pipeline used for **both** candidate models, so the
# comparison in the proposal is over an identical pipeline. In the CONFIG cell set `MODEL_KEY` to
# `'mt5'` or `'mbart'`, then **Run all**. The notebook will:
# 
# 1. load the prepared XL-Sum Nepali splits (`train`/`validation`/`test`),
# 2. tokenize articles → summaries,
# 3. attach **LoRA** adapters (base weights frozen),
# 4. fine-tune while monitoring validation ROUGE,
# 5. evaluate on the test set with **ROUGE** and **BERTScore**, and
# 6. save metrics, sample outputs, and the adapter to `results/`.
# 
# Run it **twice** — once with `MODEL_KEY='mt5'`, once with `MODEL_KEY='mbart'` — then open
# `compare_results.ipynb`.

# %%
# IMPORTANT: Colab's default `transformers` is now 5.x, which breaks peft + Seq2SeqTrainer
# here (ImportError inside get_peft_model). Pin a tested 4.x stack instead.
# >>> After this cell finishes, RESTART THE KERNEL (Runtime > Restart), then Run All again. <<<
!pip -q install -U "transformers==4.46.*" "datasets==2.21.*" "peft==0.14.*" "accelerate>=1.1" evaluate rouge_score bert_score sacremoses sentencepiece

# %% [markdown]
# ## 1. Configuration
# Pick the model here. Everything downstream (tokenizer, LoRA targets, language tokens, precision)
# is derived from this one choice.

# %%
import os, json, random, numpy as np, torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- choose which model to train: 'mt5' or 'mbart' ----
MODEL_KEY = 'mt5'

MODEL_CONFIGS = {
    'mt5': {
        'name': 'google/mt5-base',
        'target_modules': ['q', 'v'],             # T5 attention projections
        'lang_code': None,                        # mT5 needs no language token
    },
    'mbart': {
        'name': 'facebook/mbart-large-50',
        'target_modules': ['q_proj', 'v_proj'],   # BART attention projections
        'lang_code': 'ne_NP',                     # mBART-50 uses explicit language tokens
    },
}
CFG = MODEL_CONFIGS[MODEL_KEY]

MAX_IN, MAX_TARGET = 512, 96      # article -> summary lengths (tokens)
EPOCHS   = 3
LR       = 3e-4
BATCH    = 2                       # per-device; keep small for a 16GB T4
GRAD_ACCUM = 8                     # effective batch = BATCH * GRAD_ACCUM = 16

# Where the prepared JSONL splits live (see the next cell for how to get them onto Colab).
DATA_DIR    = 'data/processed'
RESULTS_DIR = 'results'
OUTPUT_DIR  = f'{MODEL_KEY}-ne-lora'
os.makedirs(RESULTS_DIR, exist_ok=True)

print('Training:', CFG['name'], '| device:',
      'cuda' if torch.cuda.is_available() else 'CPU  <-- switch to a GPU runtime!')

# %% [markdown]
# ## 2. Load the data (self-contained)
# Uses local JSONL splits if they exist (from `src/prepare.py`). Otherwise — e.g. on a fresh Colab
# VM, including when Colab is driven from VS Code — it downloads XL-Sum Nepali straight from Hugging
# Face and builds the splits in `DATA_DIR`. No manual upload needed.

# %%
import tarfile, urllib.request

def ensure_data(data_dir):
    splits = ['train', 'validation', 'test']
    if all(os.path.exists(f'{data_dir}/{s}.jsonl') for s in splits):
        print('using local splits in', data_dir)
        return
    print('local splits not found -> downloading XL-Sum Nepali from Hugging Face ...')
    os.makedirs(data_dir, exist_ok=True)
    url = ('https://huggingface.co/datasets/csebuetnlp/xlsum/'
           'resolve/main/data/nepali_XLSum_v2.0.tar.bz2')
    archive = '/tmp/nepali_XLSum_v2.0.tar.bz2'
    if not os.path.exists(archive):
        urllib.request.urlretrieve(url, archive)
    tag2split = {'train': 'train', 'val': 'validation', 'test': 'test'}
    with tarfile.open(archive, 'r:bz2') as t:
        for mem in t.getmembers():
            if not mem.name.endswith('.jsonl'):
                continue
            tag = mem.name.split('_')[-1].replace('.jsonl', '')
            rows = [json.loads(l) for l in
                    t.extractfile(mem).read().decode('utf-8').splitlines() if l.strip()]
            with open(f'{data_dir}/{tag2split[tag]}.jsonl', 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps({'text': r['text'], 'summary': r['summary'],
                                        'title': r['title']}, ensure_ascii=False) + '\n')
            print(f'{tag2split[tag]}: {len(rows)} rows')

ensure_data(DATA_DIR)

# %%
from datasets import load_dataset

ds = load_dataset('json', data_files={
    'train':      f'{DATA_DIR}/train.jsonl',
    'validation': f'{DATA_DIR}/validation.jsonl',
    'test':       f'{DATA_DIR}/test.jsonl',
})
# Drop empty / whitespace-only rows.
ds = ds.filter(lambda r: bool(r['text'] and r['summary']
                              and r['text'].strip() and r['summary'].strip()))
print(ds)
print('example summary:', ds['train'][0]['summary'][:160])

# %% [markdown]
# ## 3. Tokenize
# mT5 needs no special language handling. mBART-50 is multilingual-with-language-tokens, so we set
# the source and target language to Nepali (`ne_NP`) before tokenizing.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(CFG['name'])
if CFG['lang_code']:                       # mBART-50: set source & target language
    tok.src_lang = CFG['lang_code']
    tok.tgt_lang = CFG['lang_code']

def preprocess(batch):
    model_inputs = tok(batch['text'], max_length=MAX_IN, truncation=True)
    labels = tok(text_target=batch['summary'], max_length=MAX_TARGET, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)

# %% [markdown]
# ## 4. Load model + attach LoRA
# The pretrained weights are frozen; only the low-rank adapter matrices on the attention
# projections are trained. For mBART we also force generation to start with the Nepali token.

# %%
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForSeq2SeqLM.from_pretrained(CFG['name'])

if CFG['lang_code']:                        # force mBART to begin decoding in Nepali
    # convert_tokens_to_ids is stable across transformers versions; the older
    # tok.lang_code_to_id dict was removed in transformers 5.x.
    forced = tok.convert_tokens_to_ids(CFG['lang_code'])
    assert forced is not None and forced != tok.unk_token_id, \
        f"language token {CFG['lang_code']!r} not found in tokenizer"
    model.config.forced_bos_token_id = forced
    model.generation_config.forced_bos_token_id = forced

lora = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=CFG['target_modules'],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# %% [markdown]
# ## 5. Trainer
# **Precision note:** T5/mT5 overflow to `NaN` under fp16, so we use bf16 when the GPU supports it
# (L4/A100) and fall back to fp32 for mT5 on a T4. mBART is fp16-stable on the T4. ROUGE is computed
# every epoch (fast) to pick the best checkpoint; BERTScore is computed once at the end (slower).

# %%
import evaluate
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
if bf16_ok:
    use_bf16, use_fp16 = True, False
elif MODEL_KEY == 'mt5':
    use_bf16, use_fp16 = False, False      # fp32 keeps mT5 from going NaN on a T4
else:
    use_bf16, use_fp16 = False, True       # mBART is fp16-stable
print(f'precision -> bf16={use_bf16} fp16={use_fp16}')

rouge = evaluate.load('rouge')
collator = DataCollatorForSeq2Seq(tok, model=model)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds  = np.where(preds  != -100, preds,  tok.pad_token_id)
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    dp = tok.batch_decode(preds,  skip_special_tokens=True)
    dl = tok.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=dp, references=dl, use_stemmer=False)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    bf16=use_bf16, fp16=use_fp16,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    generation_num_beams=4,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='rougeL',
    greater_is_better=True,
    logging_steps=50,
    report_to='none',
    seed=SEED,
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=tokenized['train'], eval_dataset=tokenized['validation'],
    data_collator=collator, compute_metrics=compute_metrics,
)

# %%
import torch
print(torch.cuda.is_available())

# %%
trainer.train()

# %% [markdown]
# ## 6. Test-set evaluation — ROUGE + BERTScore
# Generate once on the held-out test set, then score with both metrics and save to `results/`.

# %%
pred_out = trainer.predict(tokenized['test'], max_length=MAX_TARGET, num_beams=4)

preds  = np.where(pred_out.predictions != -100, pred_out.predictions, tok.pad_token_id)
labels = np.where(pred_out.label_ids   != -100, pred_out.label_ids,   tok.pad_token_id)
pred_txt = tok.batch_decode(preds,  skip_special_tokens=True)
ref_txt  = tok.batch_decode(labels, skip_special_tokens=True)

rouge_res = rouge.compute(predictions=pred_txt, references=ref_txt, use_stemmer=False)

bertscore = evaluate.load('bertscore')
bs = bertscore.compute(predictions=pred_txt, references=ref_txt,
                       model_type='bert-base-multilingual-cased')   # covers Nepali
bertscore_f1 = float(np.mean(bs['f1']))

metrics = {
    'model_key': MODEL_KEY, 'model_name': CFG['name'],
    'epochs': EPOCHS, 'lr': LR, 'max_in': MAX_IN, 'max_target': MAX_TARGET,
    'rouge1': float(rouge_res['rouge1']),
    'rouge2': float(rouge_res['rouge2']),
    'rougeL': float(rouge_res['rougeL']),
    'rougeLsum': float(rouge_res['rougeLsum']),
    'bertscore_f1': bertscore_f1,
    'n_test': len(pred_txt),
}
with open(f'{RESULTS_DIR}/{MODEL_KEY}_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 7. Save sample outputs + the LoRA adapter
# Keep a few generated-vs-reference summaries for qualitative analysis, and persist the trained
# adapter (only a few MB). Copy it to Drive if you want to keep it past the Colab session.

# %%
# Qualitative samples
with open(f'{RESULTS_DIR}/{MODEL_KEY}_samples.jsonl', 'w', encoding='utf-8') as f:
    for i in range(min(20, len(pred_txt))):
        f.write(json.dumps({
            'article':   ds['test'][i]['text'][:500],
            'reference': ref_txt[i],
            'generated': pred_txt[i],
        }, ensure_ascii=False) + '\n')

# LoRA adapter (small)
trainer.model.save_pretrained(f'{OUTPUT_DIR}/adapter')
tok.save_pretrained(f'{OUTPUT_DIR}/adapter')
print('saved adapter ->', f'{OUTPUT_DIR}/adapter')

# To download / keep it:
# !zip -r {MODEL_KEY}-adapter.zip {OUTPUT_DIR}/adapter
# from google.colab import files; files.download(f'{MODEL_KEY}-adapter.zip')


