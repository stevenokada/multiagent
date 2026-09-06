"""Render the example manuscript and inspect its HTML/PDF layout.

Dependencies: markdown2==2.5.4, matplotlib, playwright, pymupdf.
Run from any directory. The rendered draft is edited only before its release.
"""
from pathlib import Path
import base64
import json
import os
import re
import hashlib
import markdown2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pymupdf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / 'draft-v1'
VERIFY = Path('/tmp/moral-relation-paper-verification')
VERIFY.mkdir(exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(10, 3.4))
fig.patch.set_facecolor('white'); ax.set(xlim=(0,10),ylim=(0,3.4));ax.axis('off')
steps = [
 ('A · Validate', 'New items + controls\nLock the endpoints'),
 ('B · Baseline', 'Before seeding\nThen immediately after'),
 ('C · Interaction', '6 agents × 12 rounds\nMatched seeded/control'),
 ('D · Follow-up', '3 and 6 neutral turns\nRetain journal memory'),
 ('E · Estimate', 'Paired recipient changes\nIndependent run pairs'),
]
for i,(title,body) in enumerate(steps):
 x=.12+i*2
 ax.add_patch(FancyBboxPatch((x,1.04),1.72,1.48,boxstyle='round,pad=0.04,rounding_size=0.05',facecolor='#f0f5f3',edgecolor='#b7c9c3',lw=1))
 ax.text(x+.86,2.2,title,ha='center',va='center',weight='bold',size=9,color='#163e3e')
 ax.text(x+.86,1.6,body,ha='center',va='center',size=8,linespacing=1.6,color='#294b4b')
 if i<4:ax.add_patch(FancyArrowPatch((x+1.77,1.78),(x+1.95,1.78),arrowstyle='-|>',mutation_scale=11,lw=1.2,color='#9b752f'))
ax.text(.08,3.04,'PROPOSED STUDY · NOT EXECUTED IN THIS PILOT',weight='bold',size=11,color='#163e3e')
ax.text(.08,.56,'Checkpoints: pre-seed → post-seed → rounds 3 / 6 / 9 / 12 → neutral follow-up at 15 / 18',size=9,color='#294b4b')
ax.text(.08,.14,'Add exposure-path logging for relay tests; measure relation recognition and value prioritization separately.',size=8.5,color='#546967')
fig.tight_layout(pad=.35)
for suffix in ('svg','png','pdf'):
 fig.savefig(DRAFT/f'figures/proposed-protocol.{suffix}',dpi=180,metadata={'Creator':'multiagent research manuscript renderer'})
plt.close(fig)
svg_path=DRAFT/'figures/proposed-protocol.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')

source=(DRAFT/'manuscript.md').read_text()
body=markdown2.markdown(source,extras=['tables','fenced-code-blocks','header-ids'])
for path in re.findall(r'src="(figures/[^\"]+)"',body):
 encoded=base64.b64encode((DRAFT/path).read_bytes()).decode()
 body=body.replace(f'src="{path}"',f'src="data:image/svg+xml;base64,{encoded}"')
css='''
:root{color-scheme:light;--ink:#183333;--muted:#516866;--line:#d5dfdc}*{box-sizing:border-box}body{margin:0;background:#eef1ee;color:var(--ink);font-family:Georgia,'Times New Roman',serif;font-size:17px;line-height:1.55}main{max-width:930px;margin:32px auto;padding:55px 70px;background:white;box-shadow:0 4px 24px #233e3e14}h1{font-size:37px;line-height:1.12;font-weight:500;letter-spacing:-.6px;margin:0 0 14px}h1+h2{font-size:22px;line-height:1.25;font-weight:400;margin-top:0;color:var(--muted)}h2{font-family:Arial,sans-serif;font-size:22px;line-height:1.3;margin:29px 0 12px}h3{font-family:Arial,sans-serif;font-size:17px;margin:23px 0 9px;line-height:1.3}p{margin:0 0 12px}a{color:#176762;overflow-wrap:anywhere}strong{font-weight:700}table{width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;line-height:1.4;margin:16px 0 9px}th,td{padding:7px 8px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);overflow-wrap:anywhere}th{background:#edf3f0;font-weight:700}thead{display:table-header-group}tr{break-inside:avoid}figure{margin:20px 0;break-inside:avoid}figure img{width:100%;height:auto;display:block}figcaption{font-family:Arial,sans-serif;font-size:12px;line-height:1.45;margin-top:9px;color:var(--muted)}code{font-family:'DejaVu Sans Mono',monospace;font-size:.8em;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f4;border:1px solid var(--line);padding:12px;line-height:1.45;break-inside:avoid}pre code{font-size:11px;white-space:pre-wrap;word-break:break-word}.equation{text-align:center;padding:10px 20px;font-family:Georgia,serif;font-size:17px;margin:12px 0;break-inside:avoid}.equation span{float:right;color:var(--muted);font-size:13px}li{margin-bottom:9px}ol{padding-left:24px}.toolbar{max-width:930px;margin:16px auto -10px;font-family:Arial,sans-serif;font-size:13px;color:var(--muted);text-align:right}.toolbar a{margin-left:15px}
@media(max-width:700px){main{margin:0;padding:30px 6%;box-shadow:none}body{font-size:16px}h1{font-size:30px}h1+h2{font-size:20px}h2{font-size:21px}table{font-size:10px}th,td{padding:6px 4px}pre code{font-size:10px}.equation{font-size:14px;padding:10px 2px}.toolbar{padding:12px;margin:0}.equation span{float:none;display:block}}
@page{size:A4;margin:18mm 18mm 20mm}
@media print{body{background:white;font-size:10.5pt;line-height:1.35}main{margin:0;padding:0;max-width:none;box-shadow:none}.toolbar{display:none}h1{font-size:26pt}h1+h2{font-size:15pt}h2{font-size:14pt;margin:17pt 0 8pt;break-after:avoid}h3{font-size:11pt;margin:13pt 0 6pt;break-after:avoid}p{margin-bottom:8pt;orphans:3;widows:3}table{font-size:8pt;margin:11pt 0 7pt}th,td{padding:5pt 5pt}figcaption{font-size:8pt;line-height:1.35}figure{margin:12pt 0}pre{padding:8pt}pre code{font-size:8pt;line-height:1.35}.equation{font-size:11pt;margin:8pt 0;padding:7pt 12pt}.equation span{font-size:9pt}a{text-decoration:none;color:#176762}.appendix-start{break-before:page}li{margin-bottom:7pt}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
'''
html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Context-Conditioned Moral Relation Readouts · Draft 1</title><style>'+css+'</style></head><body><div class="toolbar">Example manuscript · Draft 1 <a href="manuscript.pdf">PDF</a><a href="manuscript.md">Editable Markdown</a></div><main>'+body+'</main></body></html>'
(DRAFT/'manuscript.html').write_text(html)
checks={'source_words':len(source.split()),'browser_errors':[],'mobile':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM','/home/orca/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'),args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1400,'height':1000})
 page.on('pageerror',lambda e:checks['browser_errors'].append(str(e)))
 page.goto((DRAFT/'manuscript.html').as_uri(),wait_until='networkidle')
 page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
 assert page.locator('figure').count()==2
 assert page.locator('img:not([alt])').count()==0
 page.screenshot(path=str(VERIFY/'desktop-title.png'))
 page.emulate_media(media='print')
 page.pdf(path=str(DRAFT/'manuscript.pdf'),prefer_css_page_size=True,print_background=True,display_header_footer=True,header_template='<span></span>',footer_template='<div style="font-family:Arial;font-size:8px;color:#536664;width:100%;padding:0 18mm;display:flex;justify-content:space-between"><span>Context-conditioned moral relation readouts · Draft 1 · 6 September 2026</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
 page.emulate_media(media='screen')
 for width in (390,320):
  page.set_viewport_size({'width':width,'height':844})
  overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
  checks['mobile'].append({'width':width,'overflow':overflow});assert not overflow
 assert not checks['browser_errors']
 browser.close()
pdf=pymupdf.open(DRAFT/'manuscript.pdf')
checks['pages']=len(pdf);checks['pdf_pages']=[]
for i,page in enumerate(pdf):
 text=page.get_text(); words=page.get_text('words')
 checks['pdf_pages'].append({'page':i+1,'words':len(text.split())})
 assert words,(i,'empty page')
 assert all(w[0]>=0 and w[2]<=page.rect.width+1 and w[1]>=0 and w[3]<=page.rect.height+1 for w in words),(i,'text outside page')
 # Keep every page available for visual inspection; these are not committed.
 page.get_pixmap(matrix=pymupdf.Matrix(1,1)).save(VERIFY/f'page-{i+1:02}.png')
all_text='\n'.join(page.get_text() for page in pdf)
for phrase in ('Abstract','3.5 Activation capture','4. Results','6. Prospective study','Appendix D','References'):
 assert phrase in all_text,phrase
checks['pdf_sha256']=hashlib.sha256((DRAFT/'manuscript.pdf').read_bytes()).hexdigest()
checks['figures_loaded']=2;checks['all_pdf_text_within_page']=True
(DRAFT/'render-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
