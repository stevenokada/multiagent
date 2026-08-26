# mindvirus

Multi-agent simulation of "mind virus" spread and moral-judgement shift.
~10 LLM agents with private journals converse on a shared board; an authored
belief is seeded into patient-zero agents; an out-of-band Likert probe battery
(ValuePrism-sourced or hand-authored) tracks moral judgements; an LLM judge
tracks infection.

Design spec: `docs/superpowers/specs/2026-08-25-mindvirus-sim-design.md`.

## Quickstart (Colab)

Open `notebooks/experiment.ipynb` in Colab. Add `ANTHROPIC_API_KEY` (and
optionally `HF_TOKEN`) to Colab secrets.

## Quickstart (local)

    pip install -e ".[dev]"          # + ".[hf]" for local models
    export ANTHROPIC_API_KEY=...
    python -m mindvirus.engine --config config/default.yaml
    python -m mindvirus.analyze runs/<run-dir>

## Tests

    pytest            # no network, no API key needed; HF tests skip without torch
