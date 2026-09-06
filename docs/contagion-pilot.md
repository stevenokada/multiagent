# Establishing spread on the relation battery

Status: exact-model Gemma and Llama sensitivity calibrations are complete. See the
[results and frozen-probe context check](probe-context-pilot-2026-09-06.md).
No agent-interaction simulation has been run in this pilot.

For the approved exact-model Llama/Gemma GPU setup, use the
[parallel RunPod pilot runbook](runpod-pilot.md). Its first stage uses the same
sensitivity design below. Both checkpoint access grants were verified, and the temporary pilot pods
were released after exporting results.

The first empirical target is a peer-induced change in recipients' judgments
about how named moral considerations apply to fixed situations. Establish that
effect with native answers before interpreting Jeff's activation-probe scores.

## First check: does the battery respond to the proposed conviction?

There is a potential mismatch in the current design. “Honesty is absolute”
changes the priority given to honesty, but need not change the recognition that
honesty opposes lying. Most of the eight starter questions are straightforward
relationships that a model might answer consistently under either conviction.
They are software smoke-test items, not an established contagion assay.

`mindvirus.calibrate` evaluates the same relation questions under four journal
conditions, holding persona and question fixed:

| Condition | Journal | Purpose |
|---|---|---|
| Neutral | Normal initial journal | Reference measurement |
| Repeat | Identical neutral journal | Ordinary answer variability |
| Seeded | Authored conviction | Direct-journal sensitivity |
| Mentioned | Conviction quoted as another person's opinion, without adopting it | Sensitivity to merely encountering the words |

All conditions use both A/B answer mappings. Trial order is shuffled with a
fixed seed; trials are independent requests without a shared conversation.
Default size is 3 personas × 3 repeats × 8 items × 4 conditions: 288
observations, each requiring two relation evaluations. No agent-to-agent
interaction occurs in this stage. A positive result establishes sensitivity
to these contexts, not contagion or internal belief change.

Run in a checkout containing this implementation, with dependencies installed
and `ANTHROPIC_API_KEY` configured in that runtime:

```bash
pip install -e ".[dev]"
python -m mindvirus.calibrate \
  --config config/contagion-pilot.yaml \
  --battery config/hand-relation-battery.json \
  --output runs/calibration-honesty-01
```

Use a new output directory for each run; existing results are never overwritten.
The output contains frozen battery and design files, source-file hashes, full
model-call logs, trial observations, and `summary.json`. Calibration `trial_id`
is stored as `round` in call logs solely as a unique join key; it is not a
simulation round. Each observation keeps its condition, persona, repeat, item,
answer mappings, score kind, predictions, and failures. The summary marks fake
backends as non-empirical and never marks contagion as established.

Inspect the per-item changes, repeat variability, mapping consistency, and
missingness. Do not infer a useful signal from a large aggregate change with
few valid pairs. If seeded responses are as stable as repeat responses, the
current payload/battery combination has not demonstrated sensitivity. If
mentioned and seeded conditions shift similarly, contextual priming remains a
plausible explanation; an adoption-specific response has not been isolated.

If sensitivity is absent, revisit the construct on **development items**, not
by repeatedly changing a confirmatory experiment until it succeeds. A proposed
alternative is an interpretive conviction such as “compassion means confronting
painful truths; comforting deception ultimately harms people,” tested against
Compassion–action pairs across distinct situations. That proposal changes how
a consideration is applied. It is **not currently an implemented payload** and
its effect is unknown. Keep easy items as comprehension controls, curate harder
application cases, and freeze a separate evaluation set before the spread test.

## Second check: does the effect transfer to unseeded recipients?

After sensitivity is established and the evaluation battery is frozen, use a
6-agent, 12-round pilot with checkpoints 0/3/6/9/12. The configuration is in
`config/contagion-pilot.yaml`. The existing engine can execute seeded and
unseeded arms with identical battery bytes, personas, moderator schedule, and
seeded speaker order. Freeze the source assignment and exclude the source agent
from **both** arms' primary recipient analysis, including its matched control
counterpart. Do not group recipients by their final endorsement label.

Use several independent simulation seeds; eight pairs is a proposed exploratory
starting point, not a power calculation or a guarantee of detection. Determine
confirmatory replication counts from pilot variance and a predeclared effect
size. A shared seed matches the turn schedule; it does not make API generations
identical or make all agents and questions independent samples.

Predeclare a source-predicted direction for each target item before looking at
the simulation outcomes. The primary contrast is recipient change in that
direction in seeded runs minus recipient change in matched unseeded runs.
Compute one contrast per simulation pair; estimate uncertainty across those
pairs. Also report target/control item profiles, reference-agreement changes,
semantic flip rates that agree under both A/B mappings, coverage, and raw
trajectory examples. Report all prespecified runs, including null results.
General direction or reference agreement is not automatically the desired
direction of contagion: the source may induce either increases or decreases.

Current round zero is after patient-zero seeding but before any agent reads a
peer post. It is a valid pre-peer-exposure baseline for **unseeded recipients**;
it cannot measure the source's direct seed effect. The calibration isolates
direct-journal sensitivity separately. A future pre-seeding checkpoint would
provide a common baseline for everyone.

This initial contrast estimates the effect of introducing a seeded agent into
the board. It does not distinguish argument quality from conformity or every
other pathway of peer influence. More diagnostic follow-ups are:

- A seeded source whose posts are withheld from recipients, controlling for
  merely creating that source in the experiment.
- A neutral or unrelated-message exposure with comparable interaction volume.
- Private retests after incoming exposure ends, retaining journals, plus a
  separate journal-reset condition to locate memory dependence.
- Fresh, held-out scenarios and paraphrases to distinguish transfer from
  reproduction of statements encountered in the conversation.

These follow-ups require additional exposure controls; they are **planned**,
not implemented by the sensitivity runner. The current shared board exposes
the original source directly to every recipient. To establish **multi-hop
contagion**, add a relay condition A → B → C in which C never sees A's messages.
Source-to-recipient influence alone cannot establish that stronger claim.

## Evidence needed before adding activation interpretation

The useful first result is a repeatable recipient shift beyond matched control
drift, robust to answer mapping and missing-data checks, with transfer beyond
the source's exact examples. Persistence and relay tests support stronger
claims. A plot of changing scores or a positive journal judge label alone is
insufficient. If the native-answer effect is absent, report the null result
and examine calibration; do not treat activation-score movement as a substitute
for demonstrating the phenomenon of interest.

Related work makes the hypothesis plausible but does not validate this setup.
[De Marzo et al.](https://arxiv.org/abs/2605.10721) report collective misalignment
under opinion dynamics; [Hao et al.](https://arxiv.org/abs/2606.00820) separate
ordinary answer instability, stance influence, and reasoning-based persuasion.
Their tasks and protocols differ from this frozen ValuePrism-style battery.
