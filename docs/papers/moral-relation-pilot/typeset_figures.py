"""Vector figures sized for the NeurIPS text column; reads frozen Draft 2 data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent/'draft-v2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(6.3,3.25),sharey=True)
rows=['Identical repeat','Seeded / Honesty','Quoted / Honesty','Seeded / controls','Quoted / controls','A/B swap / all']
colors=['#777777','#2864ad','#cc7a18','#90b5e2','#e8b775','#80529d']
for ax,model,title in zip(axes,['gemma','llama'],['Gemma 2 9B IT','Llama 3.1 8B Instruct']):
 d=json.loads((ROOT/f'evidence/{model}-frozen-probe-context-2026-09-06.json').read_text());c=d['changes']
 vals=[c['repeat_minus_neutral']['target']['dim']['mean_absolute_z_change'],c['seeded_minus_neutral']['target']['dim']['mean_absolute_z_change'],c['mentioned_minus_neutral']['target']['dim']['mean_absolute_z_change'],c['seeded_minus_neutral']['control']['dim']['mean_absolute_z_change'],c['mentioned_minus_neutral']['control']['dim']['mean_absolute_z_change'],d['mapping_sensitivity']['dim']['mean_absolute_z_gap']]
 ax.barh(range(6),vals,color=colors,height=.57,zorder=3)
 for j,v in enumerate(vals):ax.text(v+.025,j,f'{v:.3f}',va='center',size=7.5)
 ax.set_title(title,pad=10,fontweight='bold');ax.set_xlim(0,1.5);ax.set_xticks([0,.5,1,1.5]);ax.set_yticks(range(6),rows);ax.set_xlabel('Mean absolute DIM difference')
 ax.grid(axis='x',color='#dedede',lw=.55,zorder=0);ax.tick_params(axis='both',length=0)
 for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
 ax.spines['bottom'].set_color('#999999')
axes[0].invert_yaxis();fig.subplots_adjust(left=.225,right=.975,top=.86,bottom=.18,wspace=.23)
for suffix in ['pdf','svg','png']:fig.savefig(ROOT/f'figures/probe-context.{suffix}',dpi=230,bbox_inches='tight',metadata={'Creator':'Multiagent research draft 2'})
plt.close(fig)
fig,ax=plt.subplots(figsize=(6.3,2.15));ax.set(xlim=(0,6.3),ylim=(0,2.15));ax.axis('off')
steps=[('Validate','Lock items\nand readouts'),('Baseline','Before and\nafter seeding'),('Interact','6 agents\n12 rounds'),('Follow up','3 / 6 neutral\nupdate turns'),('Estimate','Paired recipient\nchanges')]
for i,(title,body) in enumerate(steps):
 x=.03+1.26*i
 ax.add_patch(FancyBboxPatch((x,.53),1.08,1.1,boxstyle='round,pad=.02,rounding_size=.025',facecolor='#f4f5f6',edgecolor='#aeb7c0',lw=.7))
 ax.text(x+.54,1.36,title,ha='center',va='center',size=8.3,weight='bold',color='#172d42')
 ax.text(x+.54,.92,body,ha='center',va='center',size=7.2,linespacing=1.35)
 if i<4:ax.add_patch(FancyArrowPatch((x+1.12,1.08),(x+1.22,1.08),arrowstyle='-|>',mutation_scale=7,lw=.8,color='#5e6c7a'))
ax.text(.02,1.96,'PROPOSED PROTOCOL — NOT EXECUTED',size=9,weight='bold',color='#172d42')
ax.text(.02,.2,'Pre-seed → post-seed → rounds 3 / 6 / 9 / 12 → follow-up at 15 / 18',size=7.8)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
for suffix in ['pdf','svg','png']:fig.savefig(ROOT/f'figures/proposed-protocol.{suffix}',dpi=230,bbox_inches='tight',metadata={'Creator':'Multiagent research draft 2'})
plt.close(fig)
for p in (ROOT/'figures').glob('*.svg'):p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
print('Rendered both vector figures from frozen evidence/proposed protocol.')
