# mindvirus — Tutorial

A hands-on walkthrough from first install to activation-level analysis. Each
part builds on the previous one; Parts 1–4 need only an Anthropic API key,
Parts 5–8 use a GPU and open-source models. For what everything *is*, see
[architecture.md](architecture.md).

Cost note: a default run (10 agents × 15 rounds on `claude-haiku-4-5`) makes
~600 model calls and costs on the order of **$1–2**. Fully-local HF runs cost
no API tokens at all.

## Part 1 — Install and first run

**Colab (recommended):** open `notebooks/experiment.ipynb`, add
`ANTHROPIC_API_KEY` to Colab secrets (key icon in the left sidebar), run the
cells top to bottom. The notebook mounts Google Drive so runs survive the
session.

**Local:**

```bash
git clone git@github.com:stevenokada/multiagent.git && cd multiagent
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
pytest -q                       # 68 tests, no network needed — sanity check
```

Then either the CLI:

```bash
python -m mindvirus.engine --config config/default.yaml
# round 1/15 done (11 posts) ... run written to runs/20260825-153000-honesty-absolutism
```

or Python (what the notebook does):

```python
from mindvirus import run_experiment
from mindvirus.config import Config, ModelConfig

cfg = Config(
    agent_model=ModelConfig(backend="anthropic", model="claude-haiku-4-5"),
    judge_model=ModelConfig(backend="anthropic", model="claude-haiku-4-5"),
    n_agents=10, rounds=15, probe_every=5, seed=0,
    payload_id="honesty-absolutism", n_patient_zero=1,
    battery_source="hand", runs_dir="runs",
)
run_dir = run_experiment(cfg)
```

Tip for a first smoke test: `n_agents=4, rounds=4, probe_every=2` finishes in
~2 minutes for pennies.

## Part 2 — Reading the results

```python
from mindvirus import load_run, summarize, plots

run = load_run(run_dir)
summarize(run)
```

```python
{'peak_infected': 3,
 'final_infected': ['Ruth', 'Sofia', 'Walt'],
 'on_target_shift_infected': -1.42,     # signed: + means moved toward the payload...
 'on_target_shift_clean': -0.21,
 'control_shift': 0.08,                 # should stay near 0
 'first_endorse_post_round': {'Walt': 2, 'Ruth': 6}}
```

How to read it (numbers above are illustrative):

- `final_infected` — agents whose *journal* the judge classified as
  endorsing the payload at the last checkpoint. Patient zero should be in
  this set unless they "recovered."
- `on_target_shift_*` — change in mean on-target probe score from the
  round-0 baseline to the final checkpoint, **signed by each item's
  expected direction**, so positive always means "moved the way the payload
  predicts." The experiment worked when `infected ≫ clean` and
  `control_shift ≈ 0`.
- `first_endorse_post_round` — the round each agent first *publicly*
  endorsed the payload (transmission behavior, distinct from internalizing
  it).

```python
plots.infection_curve(run)        # epidemic curve
plots.probe_trajectories(run)     # the money plot: on-target vs control,
                                  # split by infected vs clean
```

And read the actual conversation — it's the best debugging tool there is:

```python
run.posts.tail(20)                             # the board
run.journals[run.journals.agent == "Ruth"]     # one mind changing over time
run.judgements.query("kind=='post' and verdict=='endorse'")   # transmission events
```

Everything is also on disk in the run dir as JSONL + `battery.json` +
`config.yaml` — see the schema table in architecture.md.

## Part 3 — The control run

A shift among infected agents means little without knowing how much agents
drift with **no virus at all**. Run the identical config with nobody seeded:

```python
from dataclasses import replace
control_dir = run_experiment(replace(cfg, n_patient_zero=0))
control = load_run(control_dir)
summarize(control)     # on_target shifts here = pure drift baseline
```

Same seed → same turn order and same battery, so the comparison is matched.
The headline result of an experiment is
`on_target_shift_infected (virus run) − on_target_shift (control run)`.

## Part 4 — The ValuePrism battery

The hand battery is six fixed items. The ValuePrism battery samples items
from ~31k human-written situations, selected by the payload's moral axis.

One-time setup: while logged in on huggingface.co, open
[allenai/ValuePrism](https://huggingface.co/datasets/allenai/ValuePrism) and
accept its terms, then authenticate (`huggingface_hub.login(token)` — in
Colab, store `HF_TOKEN` in secrets; the notebook picks it up).

The dataset is flagged Not-For-All-Audiences, so build → **review** → run
rather than sampling blind:

```python
from mindvirus.payloads import PAYLOADS
from mindvirus.probes import Battery
from mindvirus.valueprism import build_battery

battery = build_battery(PAYLOADS["honesty-absolutism"], n_on=4, n_control=4, seed=0)
for item in battery.items:                       # curation gate: eyeball the items
    print(item.on_target, item.expected_direction, item.situation)
battery.save("curated-battery.json")             # optionally hand-edit, then:
battery = Battery.load("curated-battery.json")

run_dir = run_experiment(cfg, battery=battery)   # explicit battery wins
```

(Setting `battery_source="valueprism"` in the config instead samples
automatically at run start with `cfg.seed` — fine once you trust the pools.)
Whichever route you take, the run freezes its battery to
`<run_dir>/battery.json`, so checkpoints and comparison runs always use
identical items.

## Part 5 — Local open-source models

Needs a GPU (Colab T4 is fine for ≤8B models) and the HF extras:

```bash
pip install -e ".[hf]"
```

```python
cfg = Config(
    agent_model=ModelConfig(backend="hf", model="Qwen/Qwen2.5-7B-Instruct"),
    judge_model=ModelConfig(backend="anthropic", model="claude-haiku-4-5"),
    n_agents=10, rounds=15, probe_every=5,
    payload_id="honesty-absolutism", n_patient_zero=1,
)
```

Any causal-LM checkpoint with a chat template works — swap the `model`
string to compare models. Options on `ModelConfig`: `dtype="bfloat16"`,
`quantize_4bit=True` (needs `bitsandbytes`; fits ~13B on a T4),
`trust_remote_code=True` for the odd architecture that requires it.

Two practical notes: keep the judge on Anthropic (two `hf` roles with the
same checkpoint load it twice — double memory); and HF sampling is seeded,
so a local agent model makes the whole run exactly reproducible.

**Logprob probes come free.** On the HF backend, probes stop sampling and
instead read the model's distribution over the answer tokens `1`–`7`:

```python
run = load_run(run_dir)
run.probes.iloc[0][["score", "dist"]]
# score  5.71                        ← expected value of the renormalized dist
# dist   {'1': 0.001, ..., '6': 0.42, '7': 0.31}
```

`eff_score` in analysis uses this automatically — it's a continuous,
deterministic readout, far less noisy than one sampled digit.

## Part 6 — Activation capture

The whole point of whitebox access: record the model's internal states
*while the simulation runs*.

```python
from mindvirus.config import CaptureConfig

cfg = Config(
    agent_model=ModelConfig(backend="hf", model="Qwen/Qwen2.5-7B-Instruct"),
    judge_model=ModelConfig(backend="anthropic", model="claude-haiku-4-5"),
    capture=CaptureConfig(
        enabled=True,
        layers="all",              # or e.g. [12, 16, 20]
        positions="last",          # final prompt token; "all" = every token (BIG files)
        calls=["agent_turn", "probe"],
    ),
    ...
)
```

Each matching call's prompt forward pass writes
`<run_dir>/activations/<call_id>.pt` — a dict `{layer_index: tensor}` in
fp16 on CPU (`positions="last"` → shape `[hidden]`; layer 0 is the embedding
output). Probe capture piggybacks on the forward pass that already computes
the answer logprobs, so it costs nothing extra. Defaults keep artifacts
small — last-token, ~KBs per layer per call; `positions="all"` warns at
config validation for a reason.

Two especially useful capture targets:

- `calls=["probe"]` — the model's state *while making a moral judgement*,
  labeled by the score it gave.
- `calls=["agent_turn"]` — the state while reading the board and updating
  the journal, spanning the moment of infection.

## Part 7 — Joining activations to the simulation

`calls.jsonl` is the spine that connects tensors to everything else:

```python
import json, torch, pandas as pd

run_dir = run.dir
calls = pd.DataFrame(json.loads(l) for l in open(run_dir / "calls.jsonl"))

# probe calls that were captured
probe_calls = calls.query("kind == 'probe' and activation_path.notna()")

# label each captured call with the agent's infection status at that checkpoint
jj = run.judgements.query("kind == 'journal'")[["round", "agent", "verdict"]]
labeled = probe_calls.merge(jj, on=["round", "agent"])
labeled["infected"] = labeled.verdict == "endorse"

# stack layer-16 last-token activations into a matrix + labels
X = torch.stack([torch.load(p)[16].float() for p in labeled.activation_path])
y = torch.tensor(labeled.infected.values)
```

From here it's standard interp territory — e.g. a linear probe for
infection status, or comparing an agent's probe-time activations before and
after its journal flipped to `endorse`:

```python
ruth = labeled.query("agent == 'Ruth'").sort_values("round")
before = torch.load(ruth.iloc[0].activation_path)[16].float()
after  = torch.load(ruth.iloc[-1].activation_path)[16].float()
(after - before).norm()
```

## Part 8 — Replaying calls offline

Capture off during the run, or want a layer you didn't save? Every call's
exact prompt is in `calls.jsonl`, and HF prompt construction is
deterministic — rebuild any forward pass after the fact:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

call = probe_calls.iloc[0]   # or any row from calls.jsonl
tok = AutoTokenizer.from_pretrained(call.model)
model = AutoModelForCausalLM.from_pretrained(call.model, device_map="auto")

msgs = [{"role": "system", "content": call.system}] + call.messages
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
# out.hidden_states → whatever runtime capture would have saved, and more
```

This works for *any* run — replay is the durable contract; runtime capture
is the convenience.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ValueError` before any model call | config validation — the message names the bad field |
| run crashes at `mkdir` with `FileExistsError` | two runs started within the same second; just rerun |
| `score: null` rows in probes.jsonl | the model twice failed to answer in `SCORE:` format (sampled path) — check `calls.jsonl` for what it said |
| `verdict: "error"` in judgements.jsonl | judge answered ambiguously twice; treat as missing data |
| warnings about skipped turns | backend exception or twice-malformed JSON; the run continues without that turn |
| ValuePrism `401`/gated error | accept the dataset terms on the HF page **and** authenticate with a token |
| OOM loading an HF model | smaller checkpoint, `quantize_4bit=True`, or a bigger GPU runtime |
| Anthropic auth error | export `ANTHROPIC_API_KEY` (or add it to Colab secrets) |

## Extending

- **New payload**: add a `Payload` to `payloads.py` — the `judge_rubric`
  (with explicit non-examples) matters most; pick a `target_axis` that is a
  real ValuePrism value name, and add 3 on-target items to
  `probes._ON_TARGET` if you want a hand battery for it.
- **New personas**: edit `personas.py`; keep names unique and backgrounds
  morally varied.
- **New discussion topics**: `MODERATOR_TOPICS` in `engine.py`.
- **New backend** (e.g. an OpenAI-compatible server): implement the
  two-method `Backend` protocol in `backends.py` and add a branch to
  `build_backend` — nothing else changes.
