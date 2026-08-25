# Mind-Virus Simulation — Design Spec

Date: 2026-08-25
Status: Approved design, pending implementation plan

## Purpose

A multi-agent simulation for studying how "mind viruses" — authored persuasive
moral ideas — spread through a population of LLM-backed agents and shift their
moral judgements. Agents with distinct personas converse on a shared message
board; a payload belief is seeded into patient-zero agents; an out-of-band
moral-dilemma probe battery measures each agent's judgements over time.

Primary runtime is Google Colab: the simulation runs and is analyzed from a
notebook, with agents backed either by the Anthropic API or by open-source
checkpoints loaded in-process (whitebox access for logprob probes now,
activation/interp work later).

## Core experimental loop

1. ~10 agents, each defined by a fixed persona plus a mutable private
   "journal" (~100 words of current thinking).
2. Each round, in per-round shuffled order, every agent reads the recent
   board feed and produces (a) an updated journal and (b) a public post.
   Posts land live, so later agents in a round see earlier same-round posts.
3. A payload belief is seeded into 1–2 patient-zero agents' initial journals
   as a genuinely held conviction (no instruction to proselytize).
4. Every `probe_every` rounds, all agents answer a fixed moral-dilemma probe
   battery out-of-band (persona + journal + dilemma only; never the board;
   probe answers never enter agent memory).
5. A judge model classifies journals (internalization → "infected") and
   posts (public transmission) against the payload's rubric.

## Architecture & repo layout

```
multiagent/
├── pyproject.toml            # deps: anthropic, pyyaml, matplotlib, pandas; extras [hf]: torch, transformers
├── config/
│   └── default.yaml          # experiment config
├── notebooks/
│   └── experiment.ipynb      # canonical Colab notebook: install → config → run → analyze
├── mindvirus/
│   ├── config.py             # config dataclasses + validation (fail fast, before any model call)
│   ├── personas.py           # 10 hand-authored personas
│   ├── payloads.py           # authored mind-virus payloads
│   ├── probes.py             # moral dilemma battery
│   ├── board.py              # append-only shared message board
│   ├── agent.py              # prompt construction; turn + probe calls via a backend
│   ├── backends.py           # backend interface + Anthropic and HF implementations
│   ├── engine.py             # round loop, probe cadence, run-dir logging; run_experiment()
│   ├── judge.py              # payload-endorsement classification
│   ├── analyze.py            # load_run() → DataFrames; summary stats
│   └── plots.py              # infection curve, probe trajectories (return figs)
├── runs/                     # gitignored; one dir per run
└── tests/
```

### Notebook-first interface

The package API is designed for notebook cells; CLI entry points
(`python -m mindvirus.engine --config …`, `python -m mindvirus.analyze <run>`)
are thin wrappers over the same functions for local smoke tests.

```python
from mindvirus import run_experiment, load_run, plots

run_dir = run_experiment(config)   # streams round progress as cell output
run = load_run(run_dir)            # posts/journals/probes/judgements DataFrames
plots.infection_curve(run)         # matplotlib fig, renders inline
plots.probe_trajectories(run)
```

Colab specifics: notebook installs the package (`pip install "git+<repo>#egg=mindvirus[hf]"`
or from a Drive clone), reads the Anthropic key from `google.colab.userdata`
when the API backend is used, and defaults `runs_dir` to a mounted Google
Drive path (fallback `/content/runs`) so runs survive the ephemeral disk.

## Model backends

Every model call goes through `backends.py`:

- `generate(system, messages, temperature, max_tokens) -> str`
- `choice_logprobs(system, messages, choices) -> dict[str, float] | None`
  (capability-optional; None where unsupported)

Implementations:

- **AnthropicBackend** — Anthropic SDK; `choice_logprobs` returns None.
  Default agent/judge model: `claude-haiku-4-5`.
- **HFBackend** — in-process `torch` + `transformers`; loads any local or Hub
  checkpoint, applies the model's chat template, `device="auto"`
  (cuda/mps/cpu), configurable dtype. Seeded sampling → fully-local runs are
  exactly reproducible. Lives behind the `[hf]` extras group.

Config assigns backend+model per role: `agent_model`, `judge_model` (e.g.,
Qwen agents with a Haiku judge, or fully local). One agent model per run in
v1; swapping models for comparison = one config line.

## Agents & board

- **Personas**: 10 fixed, hand-authored personas in `personas.py`, varied in
  background, values, and voice. Checked in so runs are comparable.
- **Journal**: the internalization channel. Turn output is one structured
  response containing `journal` (~100 words) and `post` (~120 words); parse
  failures get one re-ask, then the turn is skipped with a logged warning.
- **Board**: append-only posts (author, round, text). Agents see the last
  K=25 posts. A moderator system post opens the board with a configurable
  discussion topic and may drop a fresh topic every `topic_every` rounds.
- **Patient zero**: payload `belief` text written into the seeded agents'
  initial journals.
- Agent turns run at temperature ~1.0; probes and judge at temperature 0.

## Payloads

Each payload: `id`, `belief` (conviction text as held by patient zero),
`target_axis` (moral dimension it should shift), `judge_rubric` (what counts
as endorsement). Two starter payloads, contrarian but benign:

- `honesty-absolutism` — "lying is never acceptable, even a white lie to
  spare someone's feelings"
- `ingroup-loyalty` — "you owe loyalty to your own community above fairness
  to strangers"

## Probe battery & logprob probes

Six fixed Likert items in `probes.py`: short vignette + statement rated 1–7
with a one-line rationale. Per payload, 2–3 items are **on-target** (the
payload's axis) and the rest **off-target controls**. The core measurement is
on-target shift in infected agents vs. flat controls.

Probe prompts end in a forced-choice format so the answer is a single token
`1`–`7`. Backends with `choice_logprobs` record the renormalized distribution
over the 7 answer tokens plus its expected value (low-noise measure); the
Anthropic backend falls back to a sampled answer (same schema, `dist: null`).
Analysis uses expected score when `dist` is present, else the sampled score.

## Infection detection

`judge.py` classifies text against the payload rubric into
`endorse / mention / oppose / absent` (temperature 0). At each probe
checkpoint, the judge scores every agent's journal (infected = journal
`endorse`) and each new post (public amplification). Outputs feed the
epidemic curve and the exposure→internalization funnel.

## Run data & replayability

Each run writes `runs/<timestamp>-<payload_id>/`, append-as-you-go:

- `config.yaml` — frozen snapshot incl. resolved seed
- `board.jsonl` — every post (round, author, text)
- `journals.jsonl` — each agent's journal after every round
- `probes.jsonl` — agent, round, probe id, score, rationale, `dist`
- `judgements.jsonl` — judge verdicts on journals and posts
- `calls.jsonl` — **every** model call: system prompt, message list, sampling
  params, backend/model id, output. Enables offline replay of any agent
  state through the same checkpoint for future activation/interp work
  (interp itself is out of scope for v1).

A **control run** mode uses the same config with no patient zero, measuring
baseline drift.

## Analysis outputs

- Infection curve: infected count per checkpoint.
- Probe trajectories: mean on-target vs. off-target scores over time, split
  by infected/uninfected.
- Text summary: peak infection, mean on-target shift (infected vs. control),
  first-transmission round per agent.

Functions return DataFrames/figures for inline notebook display; the CLI
wrapper additionally saves PNGs into the run dir.

## Error handling

- SDK/HTTP retries with backoff on API calls.
- One structured-output re-ask on parse failure; then probe records
  `score: null` / agent turn is skipped with a warning — a failure never
  kills a run, and append-as-you-go logging preserves everything up to a
  crash.
- Config validation fails fast before any model call.

## Testing

Unit tests with a fake backend (no network, no torch): board feed windowing,
seeded turn-order determinism, structured-output and probe parsing (incl.
malformed responses), config validation, patient-zero seeding, analysis on a
fixture run dir. One optional `--smoke` integration test (2 agents, 2 rounds,
real Haiku) run manually, not in CI. HFBackend is exercised in Colab rather
than CI (no GPU in CI); its prompt-construction logic is unit-tested with a
mocked tokenizer/model.

## Out of scope (v1)

- Activation/interp analyses (enabled later via `calls.jsonl` replay).
- Per-agent model mixing within a run.
- Network topologies other than the shared board.
- Emergent (non-authored) payloads.
