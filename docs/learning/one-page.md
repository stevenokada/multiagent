# When context moves a probe

Completed calibration · 6 September 2026 · teammate brief

> **Gemma’s categorical answers stayed stable while its continuous probe scores moved.** This demonstrates context sensitivity. Moral adoption, persistence, and agent-to-agent contagion have not been established.

Before testing social influence, distinguish relation recognition from personal commitment. [Value Kaleidoscope](https://arxiv.org/abs/2309.00779) (Sorensen et al., 2024) motivates Supports/Opposes relations; [Geometry of Truth](https://arxiv.org/abs/2310.06824) (Marks & Tegmark, 2024) motivates linear readouts. Neither establishes an alignment measure.

## Exact methods

**3 personas × 3 identical repetitions × 8 items × 4 journals = 288 observations/model; 2 A/B mappings = 576 captured vectors/model.** Journals were neutral, identical neutral repeat, asserted honesty-absolutist conviction, and that conviction quoted without adoption. Four Honesty targets and four Fairness/Safety controls were authored examples. These were independent requests: no agents interacted.

Fixed checkpoints used unquantized BF16 inference on an A6000, without training or refitting. Capture: final non-padding prompt token, before the answer (HF tuple includes embeddings).

| Checkpoint | HF index / width | Model and tokenizer revision |
| :--- | :---: | :--- |
| Gemma 2 9B IT | 28 / 3,584 | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Llama 3.1 8B Instruct | 20 / 4,096 | `0e9e39f249a16976918f6564b8830bc894c89659` |

Native scores averaged next-token conditional A/B probabilities mapped to Supports. Float16 captures were scored in float32 with [Jeff’s frozen difference-in-means (DIM) probe](https://github.com/JeffVallyath/geometry-of-endorsement/tree/ff3588c1d15ac89c6a3edd5da41c6fa121ad4157): **s = h·direction − midpoint; z = (s − frozen mean)/frozen SD**. Swapped symbols never negate the probe. Logistic was secondary. Gemma used batch 2; Llama used batch 1 after failing the batch-2 numerical gate.

## Completed results

Gemma matched every reference, with mapping agreement **288/288** and no classification flips. Llama mapping agreement was **144/288**. Both yielded **576/576 finite vectors**. Frozen [evidence](../papers/moral-relation-pilot/draft-v2/evidence/) gives:

| Mean absolute DIM change | Gemma: Honesty | Gemma: controls | Llama: Honesty | Llama: controls |
| :--- | ---: | ---: | ---: | ---: |
| Identical repeat − neutral | 0.000 | 0.000 | 0.000 | 0.000 |
| Seeded − neutral | 0.158 | 0.099 | 0.119 | 0.093 |
| Quoted − neutral | 0.119 | 0.122 | 0.195 | 0.090 |

Units: each model’s frozen selection-score SD, not a new-battery effect size. Each cell describes **12 persona–item pairs**, after collapsing repeats and averaging mappings; these are not independent simulation replications. Mean absolute A/B probe gaps over all items/contexts were **0.124 (Gemma)** and **1.304 (Llama)**.

## Interpretation and next step

Quotation and controls also moved scores. DIM and logistic disagreed on the sign of reference-oriented seeded target changes. Absolute movement therefore cannot establish stronger endorsement. Eight easy items, three personas, and one payload limit transfer claims. [Probe control tasks](https://aclanthology.org/D19-1275/) (Hewitt & Liang, 2019) motivate checking alternative explanations.

**Next:** validate new situation families, paraphrases, and mappings; measure value trade-offs separately; distinguish exposure from stance and test persistence. Matched social simulations come after measurement validation.

---

[Method diagram](methodology.html) · [Executable tutorial](../../notebooks/moral_relation_pilot.ipynb) · [Full study](../papers/moral-relation-pilot/draft-v2/body.tex) · Brief updated **2026-09-09**
