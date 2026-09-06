# Parallel exact-model probe pilot

The user approved two RTX A6000 GPUs to accelerate the moral-relation research pilot. The existing workspace is already an isolated Git worktree. Earlier bug fixes, the relation battery, calibration runner, and report archives remain part of the working context.

## Research contract

Run `meta-llama/Meta-Llama-3.1-8B-Instruct` at `0e9e39f249a16976918f6564b8830bc894c89659` and `google/gemma-2-9b-it` at `11c9b309abf73637e4b6f9a3fa1e92e615547819` in BF16 without quantization. Verify tokenizer and model use the same revision. Capture the final real prompt token at HF hidden-state index 20 for Llama and 28 for Gemma. These correspond to Jeff's zero-indexed block layers 19 and 27.

Start with the fixed eight-item relation battery and four-condition calibration. This is a sensitivity/smoke test, not evidence of moral contagion. Batch independent probability evaluations; preserve each original prompt, swapped A/B mapping, annotation, identifier, activation, and missing result. Keep original conversation turn ordering and immediate board updates. Independent runs and models may execute concurrently; dependent agents within a conversation may not.

## Implementation

1. Extend model configuration with optional immutable revision and input-token limit; add a positive probe batch-size setting. Preserve legacy defaults. Load matching HF agent/judge configurations once and use separate loggers with distinct call IDs.
2. Add optional backend batch probability support and a typed probability/activation result. CallLogger retains one log record per original request. HF padding must not change the selected real token or include padding in saved activations. Batch failure falls back to isolated requests so one failure does not erase the entire group. No silent truncation; overlong inputs fail explicitly.
3. Batch relation measurements in calibration and simulation checkpoints. Existing generation fallback and per-item missingness remain intact. Calibration accepts capture configuration and records requested batching, hardware/package provenance, wall time, and failures. Output-directory creation must coexist safely with activation capture and reject previous output directories before loading a model.
4. Add a small process launcher with one worker per configured GPU/model. Workers receive explicit CUDA visibility, isolated output paths and independent seeds. They reuse a model between runs, reset RNG per independent run, and share it with the judge only when configurations match. The parent starts both workers before waiting, writes a manifest, records exit codes, and returns failure if either worker fails. It does not provision resources or change the experimental stage automatically.

## Cloud execution

Hugging Face access to both gated revisions is required before provisioning. Target two 48 GB A6000 GPUs, in one pod if available or separate pods otherwise. Use existing public SSH key, a PyTorch/CUDA image, adequate host RAM and persistent model/artifact storage. Current starting GPU quotes were $0.66/hour total Community or $1.06/hour total Secure; actual availability and quote must be checked at deployment. The user authorized this setup; further permission prompts are unnecessary.

Perform a small serial-versus-batched GPU comparison before a full calibration; compare native probabilities and captured vectors with explicit tolerances and preserve the measurements. Measure runtime instead of claiming a speedup from hardware alone. Export results before stopping compute after the bounded pilot. Do not launch the simulation sweep until calibration results have been assessed. No empirical results exist at design time.

## Verification

Use real tiny local transformer models to compare serial and padded batched probability/activation outputs without network downloads. Cover revision propagation, token-limit enforcement, shared HF loading, call/activation joins, swapped mappings, partial failures, calibration output handling, process isolation, failed workers and no-overwrite behavior. Run the complete existing test suite after targeted checks, then obtain an independent code review. Preserve existing report versions; any future report update must create a new version.
