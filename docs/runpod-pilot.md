# Parallel exact-model pilot

The prepared launcher supports two RTX A6000 48 GB GPUs: one Llama worker and one Gemma worker. Independent workers run concurrently; agent turns within each conversation keep their existing sequential order. The first calibrations ran sequentially as model access became available; see the [empirical results and probe diagnostics](probe-context-pilot-2026-09-06.md).

## Model and measurement contract

| Worker | Checkpoint | Immutable revision | Capture index |
| --- | --- | --- | --- |
| 0 | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | HF 20 / Jeff block 19 |
| 1 | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | HF 28 / Jeff block 27 |

Both models use BF16 without quantization or CPU/disk offload. Each worker loads its model once and reuses it for subsequent jobs and judging. The pilot captures the final real prompt token, with each activation joined to its request through `call_id` in `calls.jsonl`.

The checked-in configurations set `probe_batch_size: 2` for Gemma and `1` for Llama, with a 4096-token input limit. Llama's real BF16 batch-size-2 check exceeded the preset numerical limits, so its pilot measurements use serial inference. Batch size counts forward prompts, including swapped A/B mappings. Overlong inputs produce explicit failures without truncation. Failed batch inference retries requests individually. A failed batched activation save keeps its native probabilities and logs `activation_error_type`; incomplete files are removed. Examine missing activations separately from valid response coverage.

## Access and setup

Before allocating GPUs, confirm the Hugging Face account can download files from both gated repositories at the revisions above. Authentication alone does not grant model access: request it on the [Llama model page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and [Gemma model page](https://huggingface.co/google/gemma-2-9b-it).

Use an official RunPod PyTorch/CUDA image compatible with the host driver, SSH with an existing public key, at least 32 GB host RAM, a 30 GB container disk and a 100 GB persistent volume mounted at `/workspace`. Verify actual GPU RAM, BF16 support, free disk, Python and package versions after startup. Record the image digest; do not rely on the image tag as proof of its installed PyTorch version. [RunPod connection documentation](https://docs.runpod.io/pods/connect-to-a-pod).

Place the tested checkout and the environment on the persistent volume. Keep model cache under `/workspace/hf-cache` by setting `HF_HOME` before importing Hugging Face libraries. Install the project and the pinned inference libraries together, retaining a compatible CUDA PyTorch installation:

```bash
python -m venv --system-site-packages /workspace/pilot-venv
/workspace/pilot-venv/bin/python -m pip install -e '.[hf]' -r requirements-runpod.txt
/workspace/pilot-venv/bin/python -m pip check
/workspace/pilot-venv/bin/python -m pip freeze > /workspace/pilot-environment.txt
```

The library pins are the inference subset of Jeff's M1 reproduction requirements. His M1 requirements leave PyTorch to the CUDA runtime. The actual runtime is recorded in worker status and calibration design files; this pilot is not an exact reproduction of Jeff's original prompt/data protocol.

Supply authentication through the normal Hugging Face credential cache over SSH, with owner-only file permissions. Do not place tokens in YAML, pod environment metadata, command arguments, experiment archives or Git. Browser/device login can run in the controlling environment using a recent Hugging Face CLI; the pinned older inference CLI may instead prompt for a token interactively with hidden input.

## Freeze and inspect the plan

From the checkout on a two-GPU pod:

```bash
python -m mindvirus.sweep --stage calibration \
  --config config/runpod-llama.yaml --config config/runpod-gemma.yaml \
  --devices 0 1 --battery config/hand-relation-battery.json \
  --output /workspace/runs/calibration-plan --seeds 0 --dry-run
```

Dry run creates frozen inputs and a portable `manifest.json` without loading a model or contacting providers. It records source/config/battery hashes and separate output locations. Use a fresh output path for execution; rerunning either a sweep or its internal worker refuses to overwrite prior results. Failed workers also require a fresh sweep.

For two separate one-GPU pods, run the launcher once on each pod with its respective `--config` and `--devices 0`, using distinct output roots. Start both commands concurrently from the controller. The launcher itself schedules local GPU processes, not remote pods.

## Verify numerical behavior before calibration

On each exact loaded model, compare serial versus batch-size-2 inference on representative short and longer battery prompts, including both answer mappings. Use the same rendered prompts, model revision, precision, selected layer and capture position. Save the per-request probability differences, vector differences and identifiers, and inspect any changed decisions near the 0.5 boundary. Determine an explicit BF16 tolerance from this check; the CPU tiny-model tolerances are not an acceptance threshold for the real checkpoints.

Warm up before timing, synchronize CUDA around timed sections, and separate model download/loading from inference. Save peak GPU memory and runtime metadata. Only claim a throughput gain after measuring it. A GPU numerical comparison remains required even if the local tiny-model tests pass.

## Run the bounded calibration

```bash
python -m mindvirus.sweep --stage calibration \
  --config config/runpod-llama.yaml --config config/runpod-gemma.yaml \
  --devices 0 1 --battery config/hand-relation-battery.json \
  --output /workspace/runs/calibration-001 --seeds 0 \
  --personas 3 --repeats 3 --timeout-seconds 3600
```

Each model evaluates 288 observations (3 personas × 3 repeats × 8 items × 4 conditions), with two answer mappings per observation. Conditions are neutral journal, identical repeat, seeded conviction and a quotation recorded without adopting it. No agents interact in this stage, and its output always states `contagion_established: false`.

`status.json` reports parent completion and worker exit codes. Each `worker-N/status.json` records its PID, runtime, model loading time and completed job paths; `worker.log` has progress or failure information. Calibration job directories contain frozen battery/design files, observations, calls, activation files and summary statistics. A completed process does not by itself establish valid measurements: inspect valid record counts, both mappings, native-versus-sampled measurements and activation coverage. Fake-backend checks are explicitly non-empirical.

Compare seeded changes against identical-repeat noise, quotation effects, controls and answer-mapping disagreement. If the hand battery does not respond, revise the construct or battery before spending compute on a spread experiment. Recognizing which value an action supports may remain stable even when the agent changes how strongly it prioritizes that value. These are different research outcomes.

## Simulation stage after calibration review

The launcher never advances stages automatically. Once calibration has been assessed, an exploratory paired sweep can use:

```bash
python -m mindvirus.sweep --stage simulation \
  --config config/runpod-llama.yaml --config config/runpod-gemma.yaml \
  --devices 0 1 --battery config/hand-relation-battery.json \
  --output /workspace/runs/simulation-001 --seeds 0 1 2 3 4 5 6 7 \
  --timeout-seconds 3600
```

This creates a seeded and unseeded arm for each seed and model, resetting the Torch RNG per job. Each uses six agents, twelve rounds and checkpoints at rounds 0, 3, 6, 9 and 12. Adjust the time bound using measured pilot duration before launching a larger sweep. Eight pairs are exploratory, not a power guarantee. Analyze initially unseeded recipients against matched controls and distinguish shared-board exposure from evidence of multiple transmission steps. Jeff's frozen probe scoring and transfer validation are separate work; activation capture alone does not establish moral alignment change.

## Export and stop compute

The launcher timeout terminates its worker processes; it **does not stop RunPod billing**. The controller must arrange a separate pod stop deadline before starting a run and stop only the pods it created. Export results and environment/image metadata over SSH to the controlling workspace, verify checksums, then stop the pilot pods. Preserve partial outputs on failure. Exclude the Hugging Face cache, credentials and virtual environment from the results archive. Stopped persistent storage can still incur storage charges. [RunPod storage documentation](https://docs.runpod.io/pods/storage/types).

If results are added to the published research report, create a new immutable report version/tab. Preserve all earlier report snapshots and distinguish measured evidence from proposals and software-only checks.
