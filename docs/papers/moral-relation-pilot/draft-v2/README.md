# Research manuscript · Draft 2

**Context-Conditioned Moral Relation Readouts in Language Models**

Read `manuscript.pdf`. This edition uses the official NeurIPS 2026 style in
preprint mode, with author–year citations, a BibTeX bibliography, numbered
equations, booktabs tables, and vector figures. It is a full research draft,
not a submission to a named workshop. Authorship remains an editable
`Anonymous Authors` placeholder; no affiliation or author order is asserted.

The completed pilot and the proposed transmission study remain explicitly
separated. All 21 frozen evidence files match Draft 1 byte for byte. No new
inference was run for this edition. Draft 1 remains in its original directory.

## Edit and compile

Upload the Overleaf ZIP as a new project. Select **XeLaTeX** as the compiler
and **main.tex** as the main document. Edit the author block in `main.tex`,
the abstract in `abstract.tex`, the manuscript in `body.tex`, and citations
in `references.bib`. The ZIP includes the compiled `main.bbl` and both
PDF figures. Evidence files are provided separately in the full archive.

With a current TeX Live installation:

```sh
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Alternatively, Tectonic handles the full build:

```sh
tectonic main.tex
```

The build produces `main.pdf`; `manuscript.pdf` is the checked release copy.
Compilation does not require Pandoc, Python, model weights, or an API key.
This edition was built with Tectonic 0.17.0; Pandoc 3.11 was used only for
the one-time conversion from Draft 1. The parent `build_latex.py` regenerates
this edition from Draft 1 and would replace direct edits to the TeX files.
After that conversion, `typeset_figures.py` regenerates the publication
figures from frozen evidence using Python, NumPy, and Matplotlib.

## Layout and provenance

The unchanged `neurips_2026.sty` comes from the
[official NeurIPS 2026 formatting package](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
The visual reference was
[Training on Documents About Monitoring Leads to CoT Obfuscation](https://arxiv.org/pdf/2605.15257).
That paper supplied layout guidance; its scientific claims are not used
as evidence in this manuscript. The complete draft has no claimed workshop
page-limit compliance and has not been submitted to a venue or arXiv.

`provenance.json` retains the scientific source identifiers and adds the
formatting lineage. `render-checks.json` records PDF and citation validation.
`checksums.json` covers the released files. `evidence-checks.json` preserves
the numerical validation performed for Draft 1.

The full archive includes the manuscript, editable sources, figure variants,
frozen evidence, and provenance records. Full raw activation exports, model
checkpoints, and Jeff's full training data are not included; their recorded
identifiers and availability are described in the paper.
