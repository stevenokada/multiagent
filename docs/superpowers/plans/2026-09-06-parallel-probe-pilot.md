# Parallel probe pilot implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare and run two exact-model calibration workers with batched relation measurements and activation capture.

**Architecture:** One isolated worker process owns one GPU and one model. Independent measurements are batched within a worker; simulation turns remain sequential. Provision only after code checks and Hugging Face access pass.

**Tech Stack:** Python, PyTorch, Transformers, pytest, subprocess, RunPod MCP and SSH.

**Spec:** `docs/superpowers/specs/2026-09-06-parallel-probe-pilot-design.md`

## Global constraints

- Exact Llama revision `0e9e39f249a16976918f6564b8830bc894c89659` and Gemma revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`; BF16, no quantization.
- Capture last real token at HF indices 20 (Llama) or 28 (Gemma).
- Preserve prompts, A/B swaps, sequential board updates, per-run seeds, explicit missingness, and immutable prior outputs/report versions.
- Credentials never enter experiment files, logs, Git, or cloud pod environment metadata.

### Task 1: Exact model loading and shared ownership

**Files:** modify `mindvirus/config.py`, `mindvirus/hf_backend.py`, `mindvirus/engine.py`; test `tests/test_hf_backend.py`, `tests/test_engine.py`; create `config/runpod-llama.yaml`, `config/runpod-gemma.yaml`.

**Interfaces:** `ModelConfig.revision: str | None = None`, `ModelConfig.max_input_tokens: int | None = None`, `Config.probe_batch_size: int = 1`. Matching HF configurations reuse `agent_backend` as `judge_backend`.

- [ ] Add tests for passing revision to both pretrained loaders, rejecting overlong inputs without truncation, and a single HF construction for identical agent/judge configs.
- [ ] Run those tests and confirm expected failures.
- [ ] Pass revision through `_load`; check input lengths before model execution; share the matching HF backend; write exact-model configs with batch size 2 and 2048-token input guard.
- [ ] Run targeted tests.

### Task 2: Batched relation measurements with activation joins

**Files:** modify `mindvirus/backends.py`, `mindvirus/hf_backend.py`, `mindvirus/relations.py`, `mindvirus/calibrate.py`, `mindvirus/engine.py`; create `tests/test_batch_probes.py`.

**Interfaces:** `ChoiceResult(probabilities, activation_path=None)`; optional backend `choice_logprobs_batch(requests, choices) -> list[ChoiceResult]`; `CallLogger.choice_logprobs_batch(requests: list[dict], choices) -> list[dict | None]`; `run_relation_probes(logger, trials: list[tuple[Persona, str, ProbeItem, int]], batch_size=1) -> list[dict]`. Calibration gains keyword-only `capture` and `batch_size`.

- [ ] Test tiny real Llama/Gemma models using unequal prompt lengths: serial and batched A/B probabilities and saved last-token vectors agree, and no padding tokens appear in all-position captures.
- [ ] Test batched mapping order, per-call IDs and activation files, fallback to individual requests on batch failure, and per-item missingness.
- [ ] Run tests to demonstrate missing interfaces.
- [ ] Implement batch logits gathering using attention masks, typed results, and per-request logging. Catch batch failures and retry separately. Keep generation fallback sequential. Use the same reducer for single and batched relation measurements.
- [ ] Integrate chunked measurements into calibration and checkpoints; accept capture config without directory collisions. Record batch size, timing and runtime provenance in calibration output.
- [ ] Run batch and legacy relation/calibration/engine tests.

### Task 3: Two independent GPU workers

**Files:** create `mindvirus/sweep.py`, `tests/test_sweep.py`, `docs/runpod-pilot.md`.

**Interfaces:** CLI `python -m mindvirus.sweep --stage calibration --config config/runpod-llama.yaml --config config/runpod-gemma.yaml --devices 0 1 --battery config/hand-relation-battery.json --output runs/pilot --seeds 0`. Stage `simulation` explicitly creates seeded and unseeded jobs for each seed. Internal worker mode consumes the frozen manifest and worker index.

- [ ] Test launch ordering starts all workers before waiting, unique GPU/output assignment, nonzero exit propagation, duplicate-device rejection and no overwrite.
- [ ] Run a real two-process fake-backend calibration with one persona/repeat and verify complete isolated records and non-empirical summaries.
- [ ] Implement the manifest/worker launcher; workers reuse their backend and reset torch RNG for each run. Use unique seed/arm directories and never pass credentials through CLI arguments or manifests.
- [ ] Document exact setup, secure Hugging Face login, dry-run plan, launch commands, artifact export and bounded shutdown.
- [ ] Run the full suite, inspect the diff, and request independent review.

### Task 4: Verified cloud pilot

- [ ] Confirm Hugging Face authentication and access to files at both exact revisions without downloading weights locally.
- [ ] Inspect current GPU availability and a suitable template, then provision the authorized pair of A6000s when ready.
- [ ] Verify SSH, GPU memory, host RAM, disk, package versions and BF16 support; deploy the tested working tree and frozen battery/probes.
- [ ] Compare serial and batched probabilities/activations on short representative items, then run both calibration workers.
- [ ] Export artifacts and measured timings, assess sensitivity/missingness/mapping consistency, stop bounded pilot compute, and report actual results and next research step.

## Progress

Initial state: code preparation in progress; GPU provisioning and empirical checks pending Hugging Face access.
