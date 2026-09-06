# Research manuscript

[Read Draft 2 PDF](draft-v2/manuscript.pdf) ·
[Overleaf / LaTeX source](moral-relation-pilot-draft-v2-overleaf.zip) ·
[Full archive](moral-relation-pilot-draft-v2.zip) · [Draft history](index.html)

**Context-Conditioned Moral Relation Readouts in Language Models: An
Exact-Checkpoint Pilot and a Protocol for Multi-Agent Transmission**

Draft 2, dated 2026-09-06, presents the manuscript in the official NeurIPS
2026 preprint style, following the supplied arXiv paper's single-column
layout. It adds standard author–year citations, an eight-entry BibTeX
bibliography, five numbered equations, booktabs tables, and two vector
figures. The scientific evidence is unchanged. See the
[Draft 2 source guide](draft-v2/README.md) for compilation instructions.
This is a full preprint draft with editable anonymous authorship, not a
submission to a selected workshop or a published arXiv article.

[Draft 1 HTML](draft-v1/manuscript.html), [PDF](draft-v1/manuscript.pdf), and
[editable Markdown](draft-v1/manuscript.md) remain unchanged.

Draft 1, dated 2026-09-06, contains the completed two-model context calibration,
exact extraction/scoring/aggregation methodology, numerical failures and
interpretation limits, and a prospective moral-reasoning/transmission protocol.
The appendices reproduce all eight battery items, three personas, four journals,
prompt templates, source identifiers and reproduction commands. No new GPU run
was conducted to write the manuscript.

The empirical results and future protocol are explicitly separated. Both figures
are embedded in the HTML. The manuscript ZIP includes the editable source, PDF,
HTML, figure files, frozen numerical evidence, selected source/driver copies,
and checksum/provenance records. Full raw activation archives and model weights
are not in the package; the paper records their availability and raw-export hashes.

`render_paper.py` creates the prospective protocol diagram and renders the HTML
and PDF from Markdown. It requires Python, markdown2 2.5.4, Matplotlib, Playwright
with Chromium, and PyMuPDF. Set `CHROMIUM` to select a different browser binary.
The environment used for this draft is the workspace's existing temporary report
Python environment. Rendering does not run any inference.

Checks include independent recomputation of every frozen-probe condition contrast
and mapping statistic from the per-call CSVs, evidence hashes, PDF text bounds,
loaded figures, and mobile layout. See `draft-v1/evidence-checks.json` and
`draft-v1/render-checks.json`.

Save future manuscript revisions in a new `draft-vN/` directory and update the
history page with a link opening in a new tab. The source commit records this
manuscript separately from the immutable published report editions V1–V4.
