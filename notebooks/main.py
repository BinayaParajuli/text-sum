# %% [markdown]
# # Comparison — mT5-base vs mBART-large-50 (LoRA)
# 
# Run **after** `train_lora.ipynb` has produced both `results/mt5_metrics.json` and
# `results/mbart_metrics.json`. Produces the comparison table (CSV + LaTeX) and bar chart for the
# report.

# %%
import json, os
import pandas as pd

rows = []
for key in ['mt5', 'mbart']:
    p = f'results/{key}_metrics.json'
    if os.path.exists(p):
        rows.append(json.load(open(p, encoding='utf-8')))
    else:
        print('MISSING:', p, '- run train_lora.ipynb with that MODEL_KEY first')

assert rows, 'No metrics files found in results/.'
df = pd.DataFrame(rows).set_index('model_name')
cols = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
table = (df[cols] * 100).round(2)      # display as percentages
table

# %%
os.makedirs('results', exist_ok=True)
table.to_csv('results/comparison.csv')
print(table.to_latex(
    caption='ROUGE and BERTScore (\\%) on the XL-Sum Nepali test set.',
    label='tab:results', float_format='%.2f'))

# %%
import matplotlib.pyplot as plt

ax = table.plot(kind='bar', figsize=(8, 5), rot=0)
ax.set_ylabel('Score (%)')
ax.set_title('mT5-base vs mBART-large-50 (LoRA) — XL-Sum Nepali')
ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/comparison.png', dpi=150)
plt.show()
print('saved results/comparison.png and results/comparison.csv')