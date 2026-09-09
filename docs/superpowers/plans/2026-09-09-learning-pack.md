# Moral Relation Pilot Learning Pack

> Execution: independent document and visual tasks run in parallel; the notebook and integration run in the primary agent. The user has authorized completing this work without further permission questions.

**Goal:** Help teammates understand and rerun the completed context-calibration pilot with a one-page brief, an executable Colab tutorial, and an interactive methodology visual.

**Scope:** Presentation and tutorial artifacts around runtime commit `571bbb980903e9a5aca219007fa099edba1e2c11` and Jeff commit `ff3588c1d15ac89c6a3edd5da41c6fa121ad4157`. Preserve existing paper/report editions and uncommitted documentation. The experiment is independent direct-journal measurements; no completed agent-to-agent transmission experiment is implied.

## Deliverables and verification

- [x] One-page methods/motivation/results brief: `docs/learning/one-page.{md,html,pdf}` plus renderer. Include primary citations and clearly qualify mapping/quotation effects. Verify exactly one PDF page, no overflow, and numerical agreement with frozen JSON.
- [x] Colab tutorial: `notebooks/moral_relation_pilot.ipynb` with CPU saved-evidence walkthrough and live exact-checkpoint GPU mode. Pin repositories and probe hashes; retain both A/B mappings, frozen scoring/sign/scalers, numerical gate, serial fallback, native/probe outputs, diagnostic plots, and export. Teach prerequisites and interpretation beside each action.
- [x] Method visual: `docs/learning/methodology.{html,svg,pdf}` with accessible interactive step explanations and a shareable static diagram. Show authored inputs → rendered prompts → local HF model → native and activation branches → frozen scoring/diagnostics. Distinguish completed pilot from proposed social simulations.
- [x] Verify all notebook code cells parse; execute CPU tutorial without credentials; run meaningful tests for live-run orchestration, fallback, model/probe metadata validation and archive boundaries using stubs. Do not claim a new GPU run if none occurred.
- [x] Inspect document and visual rendering on desktop/mobile, validate local links and cited numbers, commit only learning-pack files, and push a branch so the Colab launch link is usable.

**Verification:** exactly one A4 brief page; responsive visual at four viewport sizes;
21 notebook cells including 10 executable code cells; CPU execution and 192 prompt /
16 plot combinations passed; repeated export passed; 152 platform tests passed,
including 20 tutorial support tests. No fresh GPU inference or hosted Colab UI run
was performed. Source support is pinned to `1ccbec7de5dbb222b18a653d0cc1993513d00e6a`;
the generated notebook is versioned separately so its source pin is immutable.

## Scientific contract

Two exact checkpoints: Gemma 2 9B IT HF layer 28 (3584) and Llama 3.1 8B Instruct HF layer 20 (4096). Three personas × three repeats × eight items × four journals = 288 observations/model; two mappings = 576 captures/model. Inference is BF16 without quantization; saved vectors fp16, scores float32. Probe weights and scaling are frozen. Do not negate frozen probe scores for swapped answer symbols. Repeats/persona-item pairs are not independent social-simulation runs. A live notebook rerun uses newly recorded runtime/hardware and is not promised bitwise identity with the original A6000 run.
