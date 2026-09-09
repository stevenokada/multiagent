# Moral relation pilot: a teammate learning guide

Start with the one-page brief, explore the setup, then run the notebook. These
materials explain the completed two-model context-calibration pilot. They separate
its measured results from the proposed agent-to-agent transmission study.

| Material | What to do |
| --- | --- |
| [One-page brief (PDF)](one-page.pdf) · [HTML](one-page.html) · [editable Markdown](one-page.md) | Read the motivation, exact methods, results, and interpretation limits. |
| [Interactive setup diagram](methodology.html) | Download and open the HTML in a browser. Choose a stage and switch between Gemma and Llama. The page works offline. |
| [Colab notebook](../../notebooks/moral_relation_pilot.ipynb) | Start in `replay` mode and run all cells. Inspect prompts, compare conditions and probes, then download the results. |
| [Static diagram: SVG](methodology.svg) · [PDF](methodology.pdf) · [PNG](methodology.png) | Copy the experimental setup into slides or share it separately. |

## Run the tutorial

[**Open the versioned tutorial in Colab**](https://colab.research.google.com/github/stevenokada/multiagent/blob/95cf20c728f8d3b85d176222dbd8d42cad0ec2e4/notebooks/moral_relation_pilot.ipynb)

**Replay** runs on CPU without credentials. It checks the frozen evidence hashes,
recomputes the contrasts from all 576 saved per-call scores per model, and provides
interactive prompt and results controls. It does not re-extract the original
activations, which are not distributed with the repository.

**Live** performs a fresh end-to-end experiment for Gemma 2 9B IT, Llama 3.1 8B
Instruct, or both sequentially. Use a fresh GPU runtime with a 40 GB-class GPU that
supports native BF16 and Hugging Face access to the selected checkpoints. It runs
the original numerical gate, complete calibration, activation capture, frozen
DIM/logistic scoring, analysis, and export. Exact model and tokenizer revisions,
probe hashes, and source commits are pinned in the notebook. The original GPU was
an RTX A6000 48 GB; the notebook records the hardware and software of a new run.

Use `HF_TOKEN` in Colab Secrets or the hidden token prompt. The notebook keeps model
caches outside its results folder. Live downloads need roughly 30 GB free disk for
one model or 55 GB for both, plus outputs. Follow the explanations and short
prediction exercises beside each cell. Download the ZIP before deleting a runtime.

## What the result supports

Gemma's categorical answers stayed stable while its continuous probe scores moved.
Quotation and control items also moved scores, and Llama was sensitive to the A/B
answer mapping. These findings show context-sensitive relation readouts. They do
not establish personal moral alignment change, persistence, or social contagion.
The eight authored starter items test whether a named consideration supports or
opposes an action; broader moral reasoning needs a separately validated battery.

## Verification and provenance

- The one-page PDF is exactly one A4 page, with values checked against frozen
  evidence. Its [checks](one-page-checks.json) include layout and source hashes.
- The interactive diagram passed desktop/mobile interaction and keyboard checks
  with no overflow or browser errors. See [visual checks](methodology-checks.json).
- All 10 tutorial code cells executed successfully in CPU replay mode from the
  pinned checkout. All 192 prompt combinations, 16 plot combinations, and repeated
  export passed. See [notebook checks](notebook-checks.json).
- The platform suite passed 152 tests, including 20 tutorial support tests covering
  complete observations, frozen metadata, fallback behavior, and archive contents.
- **A fresh live GPU run and execution in the hosted Colab UI were not performed
  during this validation.** The local Jupyter execution verifies the CPU path;
  the live orchestration is covered by tests using simulated backend results.

The notebook pins its runnable support and evidence to project commit
`1ccbec7de5dbb222b18a653d0cc1993513d00e6a` and Jeff's artifacts to
`ff3588c1d15ac89c6a3edd5da41c6fa121ad4157`. The completed experiment and full paper
remain available in the [V2 study archive](../papers/moral-relation-pilot/draft-v2/).
These learning materials are dated **9 September 2026**.

To rebuild the clean notebook, run:

```bash
python notebooks/build_pilot_notebook.py --revision 1ccbec7de5dbb222b18a653d0cc1993513d00e6a
```

The document and diagram renderers are [`render_one_page.py`](render_one_page.py)
and [`render_methodology.py`](render_methodology.py).
