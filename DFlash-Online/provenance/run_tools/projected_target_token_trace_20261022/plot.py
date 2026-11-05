"""Render the saved numerical trace; no model or training imports."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

root = Path(__file__).resolve().parent
case = json.loads((root/'results.json').read_text())['cases'][0]
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={'width_ratios': [1, 1.6]})
colors = ['#3467a4', '#9674b7', '#00877a', '#ca8a23']
x = np.arange(4)
axes[0].bar(x-.18, case['reference_q0'], width=.36, color='#a8bbce', label='Reference q0')
axes[0].bar(x+.18, case['target_r'], width=.36, color='#00877a', label='Fixed target r')
axes[0].set_xticks(x, ['A', 'B', 'C (correct)', 'D'])
axes[0].set_ylim(0, .46)
axes[0].set_title('Construct the target once')
axes[0].yaxis.set_major_formatter(PercentFormatter(1))
axes[0].legend(frameon=False, loc='upper right')
for i, value in enumerate(case['target_r']):
    axes[0].text(i+.18, value+.011, f'{value:.1%}', ha='center', fontsize=9)
rows = case['rows'][:31]
for token, color in zip('ABCD', colors):
    axes[1].plot([row['step'] for row in rows], [row[f'q_{token}'] for row in rows],
                 color=color, lw=2.5 if token=='C' else 1.6, label=f'q({token})')
axes[1].axhline(case['target_r'][2], color=colors[2], linestyle='--', lw=1.1, label='Fixed r(C) = 28.39%')
axes[1].axvline(12, color='#6b7280', linestyle=':', lw=1)
axes[1].text(12.6, .075, 'C first becomes\ntop-ranked at step 12', color='#374151', fontsize=10)
axes[1].set_xlim(0,30)
axes[1].set_ylim(0,.46)
axes[1].set_xlabel('Illustrative direct-logit gradient step')
axes[1].set_title('Current prediction q moves toward fixed r')
axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].legend(frameon=False, loc='upper right', ncol=3, fontsize=9)
for ax in axes:
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)
fig.suptitle('Token-level check: repair a wrong prediction without a one-hot target', fontweight='bold', y=1.01)
fig.text(.5,-.015,'Production target constructor + production soft-CE loss; four FP32 logits, step size 1.0. This is not DFlash model training.',ha='center',fontsize=10,color='#475569')
fig.tight_layout()
fig.savefig(root/'token_learning.png',dpi=160,bbox_inches='tight')
fig.savefig(root/'token_learning.svg',bbox_inches='tight')
