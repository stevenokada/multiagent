"""Prepare Draft 2 in the official NeurIPS 2026 preprint style.

Use PANDOC to select pandoc; compilation is a separate tectonic/main.tex step.
The prior immutable draft supplies the manuscript and frozen evidence.
"""
from pathlib import Path
import os,re,subprocess,shutil,html as htmlmod,json
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'draft-v1';OUT=ROOT/'draft-v2'
PANDOC=os.environ.get('PANDOC','pandoc')
OUT.mkdir(exist_ok=True)
source=(OLD/'manuscript.md').read_text()
abstract=source.split('## Abstract\n\n',1)[1].split('\n\n**Keywords:',1)[0]
body=source.split('## 1. Introduction',1)[1]
body='## 1. Introduction'+body
body=body.split('\n## References\n',1)[0]
body=body.replace('<div class="appendix-start"></div>','LATEXAPPENDIXSTART')
body=re.sub(r'^## (\d+)\. ', '# ',body,flags=re.M)
body=re.sub(r'^### (\d+\.\d+) ', '## ',body,flags=re.M)
body=re.sub(r'^## Appendix ([A-D])\. ', '# ',body,flags=re.M)
body=re.sub(r'^### ', '## ',body,flags=re.M)
body=body.replace('$PWD/docs/papers/moral-relation-pilot/draft-v1', '$PWD/docs/papers/moral-relation-pilot/draft-v2')
# Authored stimuli keep their exact text, including punctuation and line breaks.
# Replace ad-hoc web links with standard BibTeX/natbib citations.
cites={
'https://arxiv.org/abs/2309.00779':'sorensen2024value',
'https://arxiv.org/abs/2310.06824':'marks2024geometry',
'https://aclanthology.org/D19-1275/':'hewitt2019designing',
'https://arxiv.org/abs/2507.21509':'chen2025persona',
'https://arxiv.org/abs/2306.03819':'belrose2023leace',
'https://github.com/JeffVallyath/geometry-of-endorsement/blob/ff3588c1d15ac89c6a3edd5da41c6fa121ad4157/artifacts/probe_weights/README.md':'vallyath2026geometry'}
for url,key in cites.items():
 body=re.sub(r'\[[^\]]+\]\('+re.escape(url)+r'\)',f'[@{key}]',body)
# Put citations within sentences rather than after an already completed sentence.
body=re.sub(r'\. (\[@[a-z0-9]+\])\.',r' \1.',body)
body=body.replace("Jeff's geometry-of-endorsement repository supplies", "Jeff's geometry-of-endorsement repository [@vallyath2026geometry] supplies")
body=body.replace("Steven's multiagent platform supplies", "Steven's multiagent platform [@multiagent2026] supplies")
body=body.replace('The completed result is a measurable dissociation', 'The earlier project report [@pilotreport2026] presents the same pilot evidence. The completed result is a measurable dissociation')
# Real display mathematics, numbered by LaTeX, replacing hand-positioned HTML.
equations=[
r'''\begin{equation}\label{eq:native}
p_S(x)=\frac{\exp(\ell_S)}{\exp(\ell_A)+\exp(\ell_B)}.
\end{equation}''',
r'''\begin{equation}\label{eq:raw}
s_D(h)=h^\top d-b_D,\qquad s_L(h)=w^\top h+b_L.
\end{equation}''',
r'''\begin{equation}\label{eq:scaled}
z_k(h)=\frac{s_k(h)-\mu_k}{\sigma_k}.
\end{equation}''',
r'''\begin{equation}\label{eq:absolute}
A_G(c,c')=\frac{1}{12}\sum_{a=1}^{3}\sum_{i\in G}
\left|\bar z_{aic}-\bar z_{aic'}\right|.
\end{equation}''',
r'''\begin{equation}\label{eq:paired}
\begin{aligned}
D_{s,t}=\frac{1}{|\mathcal R_s|}\sum_{a\in\mathcal R_s}\Big[&
\big(Y_{s,\mathrm{seeded},t,a}-Y_{s,\mathrm{seeded},\mathrm{pre},a}\big)\\
-&\big(Y_{s,\mathrm{control},t,a}-Y_{s,\mathrm{control},\mathrm{pre},a}\big)\Big].
\end{aligned}
\end{equation}''']
assert len(re.findall(r'<div class="equation">.*?</div>',body))==5
it=iter(equations);body=re.sub(r'<div class="equation">.*?</div>',lambda _:next(it),body)
body=body.replace('The primary effect is the mean of',r'Here $\mathcal R_s$ is the fixed set of five recipients in run pair $s$. The primary effect is the mean of')
mathmap={'ℓ':r'\ell','μ':r'\mu','σ':r'\sigma','z̄':r'\bar z'}
body=re.sub(r'([A-Za-zℓμσ]|z̄)<sub>([^<]+)</sub>',lambda m:'$'+mathmap.get(m[1],m[1])+'_{'+m[2]+'}$',body)
body=body.replace('k∈{D,L}',r'$k\in\{D,L\}$').replace('r∈{seeded,control}',r'$r\in\{\mathrm{seeded},\mathrm{control}\}$')
body=body.replace('10⁻¹²',r'$10^{-12}$')
# Figures are vector PDFs for LaTeX, with automatically numbered captions.
figure_blocks=[]
def figure(m):
 src,caption=m[1],m[2]
 caption=re.sub(r'<strong>Figure \d+\.</strong>\s*','',caption)
 caption=htmlmod.unescape(re.sub('<[^>]+>','',caption))
 name=Path(src).stem
 figure_blocks.append((name,caption))
 return 'LATEXFIGURE'+str(len(figure_blocks))
body=re.sub(r'<figure><img src="([^"]+)"[^>]*><figcaption>(.*?)</figcaption></figure>',figure,body,flags=re.S)
# Convert each table separately to preserve complete captions and use booktabs.
tables=[]
def table(m):
 rows=[ [x.strip() for x in line.strip().strip('|').split('|')] for line in m[1].strip().splitlines() if line.strip().startswith('|')]
 headers=rows[0];data=rows[2:]
 caption=m[2] or ''
 tables.append((headers,data,caption))
 return '\n\nLATEXTABLE'+str(len(tables))+'\n\n'
body=re.sub(r'(^\|[^\n]+\|[ \t]*\n(?:\n*^\|[^\n]+\|[ \t]*\n)*)(?:\n+\*\*Table \d+\. ([^\n]+)\n)?',table,body,flags=re.M)
assert len(tables)==6,len(tables)
# The canonical editable sources are main.tex, body.tex and references.bib.
def convert(text):
 result=subprocess.run([PANDOC,'--from=markdown+raw_tex','--to=latex','--natbib','--top-level-division=section','--wrap=none','--syntax-highlighting=none'],input=text,text=True,capture_output=True,check=True)
 return result.stdout
tex=convert(body)
for i,(name,caption) in enumerate(figure_blocks,1):
 cap=convert(caption).strip()
 replacement='\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/'+name+'.pdf}\n\\caption{'+cap+'}\n\\label{fig:'+name+'}\n\\end{figure}'
 tex=tex.replace('LATEXFIGURE'+str(i),replacement)
for i,(headers,data,caption) in enumerate(tables,1):
 cell=lambda s:convert(s).strip()
 n=len(headers)
 # The 4 main tables use flexible first columns and numeric alignment.
 if i<=4:
  specs='@{}X'+('r'*(n-1))+'@{}'
  if i==1:headers=['Model','Width','Block','HF index','Batch']
  if i in (2,3):headers=['Comparison','Gemma\\newline DIM','Gemma\\newline logistic','Llama\\newline DIM','Llama\\newline logistic'] if i==3 else ['Comparison','Gemma\\newline Honesty','Gemma\\newline controls','Llama\\newline Honesty','Llama\\newline controls']
  # Short two-line numeric headings, using makecell for consistent line breaks.
  hcells=[]
  for h in headers:
   if '\\newline' in h:hcells.append('\\makecell[r]{'+h.replace('\\newline','\\\\')+'}')
   else:hcells.append('\\textbf{'+cell(h)+'}')
  caption=caption.replace('**','')
  cap=cell(caption)
  block='\\begin{table}[t]\n\\caption{'+cap+'}\n\\label{tab:'+str(i)+'}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabularx}{\\linewidth}{'+specs+'}\n\\toprule\n'+' & '.join(hcells)+r' \\'+'\n\\midrule\n'
  block+='\n'.join(' & '.join(cell(c) for c in row)+r' \\' for row in data)
  block+='\n\\bottomrule\n\\end{tabularx}\n\\end{table}'
 else:
  specs='@{}p{.14\\linewidth}p{.39\\linewidth}p{.13\\linewidth}p{.13\\linewidth}p{.11\\linewidth}@{}' if i==5 else '@{}p{.36\\linewidth}p{.59\\linewidth}@{}'
  block='\\begingroup\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{'+specs+'}\n\\toprule\n'+' & '.join('\\textbf{'+cell(h)+'}' for h in headers)+r' \\'+'\n\\midrule\\endhead\n'
  block+='\n'.join(' & '.join(cell(c) for c in row)+r' \\' for row in data)
  block+='\n\\bottomrule\n\\end{longtable}\n\\endgroup'
 tex=tex.replace('LATEXTABLE'+str(i),block)
# References precede supplementary methods, matching the reference paper.
tex=tex.replace('LATEXAPPENDIXSTART',r'''\FloatBarrier
\clearpage
\bibliographystyle{plainnat}
\bibliography{references}
\clearpage
\appendix''')
# Pandoc's plain Verbatim environments become breakable, readable stimulus boxes.
tex=tex.replace('\\begin{verbatim}','\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\footnotesize,frame=single,rulecolor=\\color{black!20},framesep=5pt]').replace('\\end{verbatim}','\\end{Verbatim}')
# Hexadecimal identifiers and paths must wrap rather than protrude into margins.
tex=re.sub(r'\\texttt\{([a-f0-9]{40,64})\}',r'\\texttt{\\seqsplit{\1}}',tex)
tex=re.sub(r'\\texttt\{([^{}]*[/_][^{}]*)\}',lambda m:'\\nolinkurl{'+m[1].replace('\\_','_').replace('\\ ',' ')+'}',tex)
# Explicit unicode math support in prose, preserving textual stimuli in Verbatim.
text_math={'≤':r'\ensuremath{\leq}','≥':r'\ensuremath{\geq}','→':r'\ensuremath{\rightarrow}','×':r'\ensuremath{\times}','−':r'\ensuremath{-}','′':r'\ensuremath{\prime}'}
chunks=re.split(r'(\\begin\{Verbatim\}.*?\\end\{Verbatim\})',tex,flags=re.S)
for j,ch in enumerate(chunks):
 if not ch.startswith('\\begin{Verbatim}'):
  for old,new in text_math.items():ch=ch.replace(old,new)
  chunks[j]=ch
tex=''.join(chunks)
(OUT/'body.tex').write_text(tex)
(OUT/'abstract.tex').write_text(convert(abstract))
preamble=r'''\documentclass[10pt]{article}
\PassOptionsToPackage{authoryear,round}{natbib}
\usepackage[preprint]{neurips_2026}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{lmmono10-regular.otf}
\usepackage{amsmath,amssymb,booktabs,tabularx,longtable,array,makecell}
\usepackage{graphicx,microtype,xcolor,url,xurl,hyperref}
\usepackage{fvextra,seqsplit,placeins}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue!55!black,
 pdftitle={Context-Conditioned Moral Relation Readouts in Language Models},
 pdfauthor={Anonymous Authors},pdfsubject={Research draft 2; completed pilot and prospective protocol}}
\setcitestyle{authoryear,round}
\setlength{\emergencystretch}{1.5em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\pandocbounded}[1]{#1}
\title{Context-Conditioned Moral Relation Readouts\\in Language Models}
\author{Anonymous Authors\\Research draft, 6 September 2026}
\begin{document}
\maketitle
\begin{abstract}
\input{abstract}
\end{abstract}
\input{body}
\end{document}
'''
(OUT/'main.tex').write_text(preamble)
# Copy existing evidence and diagrams; numerical values remain unchanged.
if not (OUT/'evidence').exists():shutil.copytree(OLD/'evidence',OUT/'evidence')
for name in ('evidence-checks.json',):shutil.copy2(OLD/name,OUT/name)
for p in (OLD/'figures').iterdir():shutil.copy2(p,OUT/'figures'/p.name)
print(json.dumps({'tables':len(tables),'figures':len(figure_blocks),'equations':len(equations),'source':'draft-v1/manuscript.md','output':'draft-v2/main.tex'}))
