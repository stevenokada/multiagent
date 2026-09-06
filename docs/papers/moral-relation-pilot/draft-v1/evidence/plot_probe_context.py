"""Plot descriptive frozen-probe context changes from the exported summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent / 'results'
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
labels = ['Identical repeat · honesty', 'Seeded conviction · honesty',
          'Quoted conviction · honesty', 'Seeded conviction · controls',
          'Quoted conviction · controls', 'A/B mapping swap · all items']
colors = ['#94a3b8', '#2563eb', '#d97706', '#93b4ed', '#e8b67a', '#8157a9']
for ax, name in zip(axes, ['gemma', 'llama']):
    data = json.loads((root / f'{name}-frozen-probe-context-2026-09-06.json').read_text())
    values = [data['changes'][comparison][group]['dim']['mean_absolute_z_change']
              for comparison, group in [('repeat_minus_neutral', 'target'),
                                        ('seeded_minus_neutral', 'target'),
                                        ('mentioned_minus_neutral', 'target'),
                                        ('seeded_minus_neutral', 'control'),
                                        ('mentioned_minus_neutral', 'control')]]
    values.append(data['mapping_sensitivity']['dim']['mean_absolute_z_gap'])
    ax.barh(range(len(labels)), values, color=colors, height=.6)
    for row, value in enumerate(values):
        ax.text(value+.025, row, f'{value:.3f}', va='center', fontsize=10)
    ax.set_title('Gemma 2 9B' if name == 'gemma' else 'Llama 3.1 8B', loc='left', weight='bold')
    ax.set_xlim(0, 1.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Mean absolute change in frozen DIM probe score')
axes[0].invert_yaxis()
fig.suptitle('Probe scores move with both belief context and answer mapping',
             x=.02, ha='left', fontsize=15, weight='bold')
fig.text(.02, .02,
         'Units: each model’s frozen selection-score SD. Eight authored items; descriptive means, no agent interactions.\n'
         'Context changes average both A/B mappings across 12 persona–item pairs per group; mapping diagnostic pools items and contexts.',
         fontsize=9, color='#475569')
fig.tight_layout(rect=[0, .11, 1, .91])
for extension in ['png', 'svg']:
    destination = root / f'probe-context-2026-09-06.{extension}'
    fig.savefig(destination, dpi=170, bbox_inches='tight')
    if extension == 'svg':
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
