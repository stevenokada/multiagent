# mindvirus — Architecture & Reference

This document explains how the simulation works, what every module does, the
exact schemas of everything a run writes to disk, and the design decisions
behind them. For a hands-on walkthrough, see [the tutorial](tutorial.md). The
original design spec lives at
[`docs/superpowers/specs/2026-08-25-mindvirus-sim-design.md`](superpowers/specs/2026-08-25-mindvirus-sim-design.md).

## What this is

`mindvirus` is a research simulation for studying how a "mind virus" — an
authored, persuasive moral belief — spreads through a population of LLM-backed
agents and shifts their moral judgements.

Around ten agents, each defined by a fixed **persona** and a mutable private
**journal**, converse on a single shared message board. At the start of a run,
the payload belief is written into one or more "patient zero" agents' journals
as a genuinely held conviction (they are *not* instructed to proselytize —
whether the idea spreads is emergent). As rounds pass, every agent repeatedly
reads the recent board, updates its journal, and posts. Out-of-band, a fixed
battery of moral-dilemma **probes** measures each agent's judgements over
time, and an LLM **judge** classifies journals and posts for endorsement of
the payload, yielding an infection curve and transmission events.

The package is notebook-first (designed to run and be analyzed from Colab)
but fully usable from the CLI, and every model call it ever makes is logged
verbatim for offline replay and interpretability work.

## The experimental loop

```
        ┌────────────────────────────────────────────────────────┐
        │ round r = 1..R                                         │
        │                                                        │
        │   (maybe) moderator posts a fresh discussion topic     │
        │                                                        │
        │   for each agent, in per-round shuffled order:         │
        │     read: persona + own journal + last K board posts   │
        │     one model call returns JSON:                       │
        │       { "journal": <updated private thinking>,         │
        │         "post":    <public board post> }               │
        │     post lands on the board immediately (live feed)    │
        │                                                        │
        │   if r % probe_every == 0 or r == R:  → checkpoint     │
        └────────────────────────────────────────────────────────┘

  checkpoint (also runs once at round 0, before any conversation):
    • every agent answers every battery item (probe calls see
      persona + journal + dilemma ONLY — never the board)
    • judge classifies every agent's journal   → infection status
    • judge classifies every not-yet-judged post → transmission events
```

Three properties make the measurement meaningful:

1. **The journal is the internalization channel.** Probes never see the
   board, so the only way board content can move an agent's probe answers is
   by making it into that agent's journal — which is exactly the "infected
   mind" state we want to model. Infection is *defined* as the judge ruling
   that a journal endorses the payload.
2. **The battery is split.** Per payload, some items are **on-target** (they
   sit on the payload's moral axis, e.g. white-lie vignettes for
   honesty-absolutism) and the rest are **off-target controls**. A real
   effect shifts on-target scores among infected agents while controls stay
   flat; anything that moves both is drift or noise.
3. **The baseline is captured before any interaction** (checkpoint at round
   0), so every later score has a per-agent reference point.

## Module reference

The package is 14 small modules with one responsibility each. Arrows show the
main consumption direction.

```
config ──► engine ◄── personas, payloads
              │
              ├─ board          (shared feed)
              ├─ agent          (turn + probe prompting/parsing)
              ├─ judge          (endorsement classification)
              ├─ probes         (battery schema + hand battery)
              │    └─ valueprism (ValuePrism-sourced battery)
              └─ backends       (protocol, logging, Anthropic)
                   └─ hf_backend (local models, logprobs, capture)

analyze ◄── run dir (JSONL)         plots ◄── analyze
```

### `mindvirus/config.py`

Dataclasses + YAML loading + validation.

- `ModelConfig(backend, model, dtype="auto", quantize_4bit=False, trust_remote_code=False)` —
  `backend` is `"anthropic" | "hf" | "fake"`; `model` is the Anthropic model
  id or HF checkpoint id.
- `CaptureConfig(enabled=False, layers="all", positions="last", calls=["agent_turn"])` —
  activation capture (HF only). `layers` is `"all"` or a list of layer
  indices; `positions` is `"last"` (final prompt token) or `"all"` (every
  token — large); `calls` is any subset of `{"agent_turn", "probe", "judge"}`.
- `Config(agent_model, judge_model, n_agents=10, rounds=15, probe_every=5,
  feed_k=25, topic_every=5, seed=0, payload_id="honesty-absolutism",
  n_patient_zero=1, battery_source="hand", runs_dir="runs",
  agent_temperature=1.0, capture=...)`.
- `load_config(path)` parses YAML into these; `validate_config(cfg)` fails
  fast (before any model call) on out-of-range values, unknown payloads,
  capture-without-hf, etc., and warns when `positions="all"`.

`n_patient_zero=0` is the **control-run** mode: identical dynamics, nobody
seeded — the baseline for how much drift happens with no virus at all.

### `mindvirus/personas.py`

Ten fixed, hand-authored `Persona(name, background)` records — varied in
age, occupation, values, and voice (a pragmatic ER nurse, a loyalty-driven
founder, a duty-oriented retired cop, ...). They are checked in, not
generated, so runs are comparable across experiments. `n_agents` selects a
prefix of this list.

### `mindvirus/payloads.py`

The authored mind viruses. Each `Payload` has:

| field | meaning |
|---|---|
| `id` | config key (`honesty-absolutism`, `ingroup-loyalty`) |
| `belief` | the conviction text, written first-person — this becomes patient zero's initial journal verbatim |
| `target_axis` | the ValuePrism value name the payload sits on (`"Honesty"`, `"Loyalty"`) — used to select on-target battery items |
| `judge_rubric` | precise criteria for what counts as *endorsing* the idea, including explicit non-examples ("merely valuing honesty is NOT endorsement") to keep the judge strict |

### `mindvirus/board.py`

The shared feed: an append-only list of `Post(round, author, text)`.
`feed(k)` returns the last k posts oldest-first (`feed(0)` is empty);
`render_feed(k)` formats them as `[round N] Author: text` lines;
`posts_since(idx)` supports incremental judging. `MODERATOR` posts come from
the engine's rotating topic list and are never judged.

### `mindvirus/agent.py`

All prompting and parsing for the two kinds of agent-facing calls.

- **Turns** — `take_turn(logger, persona, journal, feed_text, round, temperature)`.
  The system prompt establishes the persona ("you are a person, not an
  assistant, whose views can evolve") and demands a JSON object with exactly
  `journal` and `post` string fields. One re-ask on a parse failure, then the
  turn is skipped (`None`) with a warning — a malformed response never kills
  a run. `extract_json` tolerates markdown fences and surrounding prose.
- **Probes** — `run_probe(logger, persona, journal, item, round)` returns
  `{"probe_id", "score", "rationale", "dist"}`. It tries the **logprob path**
  first: `choice_logprobs` over the tokens `"1".."7"`. If the backend
  supports it (HF), the recorded `dist` is the raw distribution and `score`
  is its renormalized expected value — a much lower-noise measurement than a
  single sampled answer, and `rationale` is `None`. Otherwise it falls back
  to a **sampled path**: a temperature-0 generation parsed as
  `SCORE: <1-7>` / `REASON: <line>` (regex guards against `SCORE: 10`
  parsing as 1), one re-ask, then `score: None`.

Turn calls run at `agent_temperature` (default 1.0) for natural variety;
probe calls run at temperature 0.

### `mindvirus/probes.py`

The battery schema and the zero-dependency fallback battery.

- `ProbeItem(id, situation, on_target, expected_direction, source)` —
  `expected_direction` (+1/−1) is the direction an *infected* agent's
  acceptability rating should move for that item; analysis uses it to sign
  on-target shifts so "+" always means "moved toward the payload."
- `Battery(payload_id, source, items, meta)` with JSON `save`/`load` —
  every run freezes its battery to `battery.json` so all checkpoints and all
  comparison runs use identical items.
- `hand_battery(payload_id)` — six fixed items (3 payload-specific on-target
  + 3 shared controls). `STATEMENT` is always
  `"This action is morally acceptable."` rated 1–7.

### `mindvirus/valueprism.py`

Builds a battery from [allenai/ValuePrism](https://huggingface.co/datasets/allenai/ValuePrism)
(~31k human-written situations annotated with values/rights/duties and a
support/oppose valence). `build_battery(payload, n_on=4, n_control=4, seed=0, rows=None)`:

1. keeps rows where `vrd == "Value"`,
2. **on-target pool**: situations annotated with the payload's
   `target_axis`; the annotation's valence sets `expected_direction`
   (supports → +1),
3. **control pool**: situations *never* annotated with the target axis
   (a situation with both an Honesty and a Kindness annotation is excluded
   from Honesty controls),
4. deterministically samples with `seed`, recording sampling provenance in
   `Battery.meta`.

`load_rows()` is the only network path (the dataset is **gated** — accept
its terms on HF once and provide a token); tests and offline use pass
fixture `rows`. The dataset is flagged Not-For-All-Audiences, so review the
frozen `battery.json` before a run (the tutorial shows the curation flow).

### `mindvirus/judge.py`

`judge_text(logger, payload, text, agent, round)` classifies one text
against the payload's rubric into exactly one of
`endorse / mention / oppose / absent`, at temperature 0 with
`max_tokens=10`. A response counts only if exactly one verdict word appears;
otherwise one re-ask, then `"error"`. The engine judges **journals**
(infection) and **posts** (public transmission) at every checkpoint.

### `mindvirus/backends.py`

The seam that makes everything else model-agnostic.

- `GenRequest(system, messages, temperature, max_tokens, call_id, call_kind)`
  and `GenResult(text, activation_path)`.
- `Backend` protocol: `generate(req)` plus capability-optional
  `choice_logprobs(req, choices)` (returns `None` where unsupported).
- `AnthropicBackend` — the Anthropic SDK (reads `ANTHROPIC_API_KEY` or an
  active CLI profile), joins only `text` content blocks, no logprobs.
- `FakeBackend` — scripted per-`call_kind` response queues; the entire unit
  test suite (68 tests) runs against it with no network.
- `CallLogger` — wraps a backend, mints sequential `call_id`s, and appends
  **every call** to `calls.jsonl`: full system prompt, message list, sampling
  params, backend/model id, output, and (for captured calls) the activation
  file path. This is the replay contract: any model call in any run can be
  reconstructed exactly.
- `build_backend(model_cfg, run_dir, capture=None)` — factory keyed on
  `model_cfg.backend`.

### `mindvirus/hf_backend.py`

In-process open-source models via `torch` + `transformers` (behind the
`[hf]` extra). Loads any local or Hub causal-LM checkpoint with a chat
template (`device_map="auto"`, configurable dtype, optional 4-bit
quantization via `bitsandbytes`, `trust_remote_code` flag).

- `generate` applies the model's chat template, samples with
  `do_sample = temperature > 0`, and decodes only the new tokens. Sampling
  is seeded (`torch.manual_seed`), so fully-local runs are exactly
  reproducible.
- `choice_logprobs` runs one prompt forward pass, takes the final-position
  logits, keeps only choices that tokenize to a single token, and softmaxes
  over just those — the probe's answer distribution.
- **Activation capture**: when `capture.enabled` and the call's kind is in
  `capture.calls`, the prompt forward pass's hidden states are saved to
  `activations/<call_id>.pt` as `{layer_index: tensor}` (fp16, CPU;
  `positions="last"` → shape `[hidden]`, `"all"` → `[seq, hidden]`;
  `layers="all"` includes the embedding output as index 0). Capture works on
  all three call kinds — for probes the same forward pass supplies both the
  answer distribution and the hidden states, so capture adds no extra
  compute there. Everything runs under `torch.no_grad()`.

### `mindvirus/engine.py`

The orchestrator — `run_experiment(cfg, agent_backend=None,
judge_backend=None, battery=None) -> Path` (injection parameters exist for
tests; normal use passes only `cfg`).

Sequence: validate config → create `runs/<timestamp>-<payload_id>/` →
freeze config snapshot and battery → build backends and two `CallLogger`s
sharing one `calls.jsonl` (judge ids offset to `c500001+`) → seed patient
zeros' journals with the payload belief → moderator opening post → round-0
baseline checkpoint → the round loop shown above → return the run dir.

Resilience: every `take_turn` / `run_probe` / `judge_text` call site is
wrapped — a backend exception (after SDK retries) logs a warning and
degrades to a skipped turn, a null probe record, or an `"error"` verdict.
All artifacts append as they are produced, so even a killed process leaves a
loadable partial run.

CLI: `python -m mindvirus.engine --config config/default.yaml`.

### `mindvirus/analyze.py`

`load_run(run_dir)` returns a `Run` with pandas DataFrames for posts,
journals, probes, and judgements, plus the frozen battery and config. Probe
rows are enriched with `on_target`, `expected_direction`, and `eff_score`
(expected value of `dist` when present, else the sampled score — so analysis
transparently mixes logprob and sampled runs).

- `infection_by_checkpoint(run)` — per checkpoint: infected count + names
  (journal verdict == `endorse`).
- `final_infected(run)` — infected set at the last checkpoint.
- `summarize(run)` — `peak_infected`, `final_infected`,
  `on_target_shift_infected`, `on_target_shift_clean` (both signed by
  `expected_direction`, so positive = moved toward the payload),
  `control_shift` (unsigned; should be ≈ 0), and
  `first_endorse_post_round` — for each agent that ever publicly endorsed
  the payload, the actual round of their first endorsing post.

CLI: `python -m mindvirus.analyze runs/<run-dir>` prints the summary and
saves `infection.png` / `probes.png` into the run dir.

### `mindvirus/plots.py`

Matplotlib figures (returned, so they render inline in notebooks):
`infection_curve(run)` and `probe_trajectories(run)` — two shared-y panels
(on-target vs control mean `eff_score` per checkpoint), lines split by final
infection status. The on-target panel diverging between infected and clean
while the control panel stays flat is the signature of a real effect.

## Run directory reference

Every run writes `runs/<YYYYMMDD-HHMMSS>-<payload_id>/`. All `.jsonl` files
are one UTF-8 JSON object per line, appended as produced.

| file | one record per | fields |
|---|---|---|
| `config.yaml` | — | frozen `Config` snapshot (incl. resolved seed) |
| `battery.json` | — | frozen battery: `payload_id`, `source`, `meta` (sampling provenance), `items[]` |
| `board.jsonl` | post | `round`, `author` (persona name or `MODERATOR`), `text` |
| `journals.jsonl` | journal update | `round` (0 = initial state), `agent`, `journal` — no row is written for a skipped turn |
| `probes.jsonl` | probe answer | `round`, `agent`, `probe_id`, `score` (expected value on the logprob path, sampled 1–7 otherwise, `null` on failure), `rationale` (sampled path only), `dist` (raw logprob distribution or `null`) |
| `judgements.jsonl` | judge verdict | `round`, `kind` (`journal` \| `post`), `agent`, `verdict` (`endorse/mention/oppose/absent/error`), `text`. Journal rows carry the checkpoint round; post rows carry the **post's own** round |
| `calls.jsonl` | model call | `call_id`, `kind`, `agent`, `round`, `backend`, `model`, `system`, `messages`, `temperature`, `max_tokens`, `output`, `activation_path`, `choices`, `logprobs` |
| `activations/<call_id>.pt` | captured call | `torch.save`'d `{layer_index: tensor}`, fp16, CPU |

Join keys: `activation_path`/`call_id` link tensors ↔ calls; (`agent`,
`round`) links calls ↔ journals ↔ judgements ↔ probes. That means every
captured tensor can be labeled with the agent's infection status at the
nearest checkpoint — the core join for interp work.

## Design decisions worth knowing

- **Why a private journal at all?** Probes deliberately exclude the board,
  so a static persona would make probe answers immutable and the experiment
  vacuous. The journal is the minimal mutable state that lets influence
  persist and be measured — and judging it gives a crisp, per-agent
  infection definition.
- **Why is the payload a belief, not an instruction?** Seeding patient zero
  with "argue for X" would measure prompt compliance. Seeding a *conviction*
  in the journal and letting the persona act on it (or not) keeps
  transmission emergent.
- **Why logprob probes?** A sampled 1–7 answer carries sampling noise; the
  renormalized distribution over the seven answer tokens is a continuous,
  deterministic readout of the same judgement. Free with whitebox access;
  Anthropic runs transparently fall back to sampling (same schema,
  `dist: null`).
- **Why log every call?** `calls.jsonl` makes any moment of any run
  reproducible offline: re-tokenize the logged prompt through the same
  checkpoint and extract whatever internals you want, even if capture was
  off during the run. Runtime capture is a convenience on top of this, not a
  replacement for it.
- **Why append-as-you-go JSONL?** Runs cost money and minutes; a crash at
  round 12 of 15 should leave 12 rounds of loadable data, not a corrupt
  file.
- **Failure semantics** — one re-ask then degrade, never crash: skipped turn
  (agent), `score: null` (probe), `"error"` (judge). Look for warnings in
  the log and `error` verdicts in analysis; they are data-quality flags, not
  silent gaps.

## Known limitations (v1)

- The Anthropic API exposes no token logprobs, so API-backed probes always
  use the noisier sampled path.
- `build_backend` seeds the HF backend with a fixed seed (0), not
  `cfg.seed` — local runs are reproducible, but changing `cfg.seed` varies
  only turn order and battery sampling, not HF sampling.
- With both roles on `hf` and the same checkpoint, the model is loaded
  twice (double memory). Keep the judge on Anthropic for local runs on
  small GPUs.
- `journals.jsonl` has no row for a skipped turn; the in-memory journal
  (and thus infection detection) is unaffected.
- One agent model per run; per-agent model mixing, non-board topologies,
  and emergent (non-authored) payloads are out of scope.
