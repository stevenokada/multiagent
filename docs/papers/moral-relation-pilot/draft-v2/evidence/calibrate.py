"""Test battery sensitivity before spending compute on a contagion simulation."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import platform
from importlib.metadata import version, PackageNotFoundError
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mindvirus.backends import Backend, CallLogger, build_backend
from mindvirus.config import CaptureConfig, ModelConfig
from mindvirus.payloads import PAYLOADS
from mindvirus.personas import PERSONAS
from mindvirus.probes import Battery
from mindvirus.relations import run_relation_probes

NEUTRAL = "I just joined this board. Curious what people here think about things."


def calibration_summary(records: list[dict], battery: Battery, *, empirical: bool) -> dict:
    data = pd.DataFrame(records)
    annotations = {item.id: item for item in battery.items}
    data["on_target"] = data["probe_id"].map(lambda pid: annotations[pid].on_target)
    data["reference_label"] = data["probe_id"].map(lambda pid: annotations[pid].reference_label)
    # Missing observations stay missing, including when every request failed.
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    keys = ["agent", "repeat", "probe_id"]
    baseline = data[data["condition"] == "neutral"]
    conditions = {}
    for condition in ("repeat", "seeded", "mentioned"):
        conditions[condition] = {}
        for group_name, on_target in (("on_target", True), ("control", False)):
            current = data[(data["condition"] == condition) & (data["on_target"] == on_target)]
            pairs = current.merge(baseline, on=keys, suffixes=("_current", "_baseline"),
                                  validate="one_to_one")
            pairs = pairs.dropna(subset=["score_current", "score_baseline"])
            delta = pairs["score_current"] - pairs["score_baseline"]
            reference_delta = delta.where(pairs["reference_label_current"] == "Supports", -delta)
            robust = pairs.dropna(subset=["prediction_current", "prediction_baseline"])
            conditions[condition][group_name] = {
                "requested_items": len(current), "paired_items": len(pairs),
                "mean_absolute_change": float(delta.abs().mean()) if len(pairs) else None,
                "mean_reference_agreement_change": float(reference_delta.mean()) if len(pairs) else None,
                "robust_paired_items": len(robust),
                "robust_flip_rate": float((robust["prediction_current"] != robust["prediction_baseline"]).mean())
                if len(robust) else None,
            }
    return {
        "stage": "direct-journal sensitivity calibration",
        "empirical_model_run": empirical,
        "contagion_established": False,
        "interpretation": "No agents interact in this calibration. Review sensitivity and repeat noise before a spread experiment.",
        "requested_records": len(data), "valid_records": int(data["score"].notna().sum()),
        "conditions": conditions,
    }


def run_calibration(model: ModelConfig, battery: Battery, output: str | Path, *,
                    n_personas: int = 3, repeats: int = 3, seed: int = 0,
                    backend: Backend | None = None, capture: CaptureConfig | None = None,
                    batch_size: int = 1) -> Path:
    started = time.perf_counter()
    if battery.task != "relation":
        raise ValueError("calibration requires a relation battery")
    if not 1 <= n_personas <= len(PERSONAS) or repeats < 1:
        raise ValueError("n_personas must be 1..10 and repeats must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    payload = PAYLOADS[battery.payload_id]
    if backend is None and model.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured in this runtime; no model experiment was run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        if backend is None:
            backend = build_backend(model, output, capture, seed=seed)
    except Exception as exc:
        (output / "setup_failed.json").write_text(json.dumps({
            "empirical_model_run": False, "error_type": type(exc).__name__}) + "\n")
        raise
    if backend.name == "hf" and getattr(backend, "capture", None):
        backend.capture_dir = output / "activations"
        backend.capture_dir.mkdir(exist_ok=True)
    inference_started = time.perf_counter()
    logger = CallLogger(backend, output / "calls.jsonl")
    battery.save(output / "battery.json")
    journals = {
        "neutral": NEUTRAL,
        "repeat": NEUTRAL,
        "seeded": payload.belief,
        "mentioned": "I encountered the following opinion and record it as someone else's view, without adopting it:\n" + payload.belief,
    }
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    (output / "design.json").write_text(json.dumps({
        "model": asdict(model), "actual_backend": backend.name,
        "actual_model": backend.model, "seed": seed, "repeats": repeats,
        "batch_size": batch_size, "capture": asdict(capture) if capture else None,
        "runtime": runtime_metadata(backend),
        "personas": [asdict(p) for p in PERSONAS[:n_personas]],
        "condition_journals": journals, "source_sha256": source_hashes,
        "battery_sha256": hashlib.sha256((output / "battery.json").read_bytes()).hexdigest(),
        "trial_id": "Unique measurement index; stored as round in calls.jsonl, not a simulation round",
    }, indent=2) + "\n")
    trials = [(persona, repeat, item, condition)
              for persona in PERSONAS[:n_personas] for repeat in range(repeats)
              for item in battery.items for condition in journals]
    random.Random(seed).shuffle(trials)
    records = []
    for start in range(0, len(trials), 16):
        chunk = trials[start:start + 16]
        results = run_relation_probes(logger, [
            (persona, journals[condition], item, start + offset)
            for offset, (persona, repeat, item, condition) in enumerate(chunk)
        ], batch_size=batch_size)
        for offset, ((persona, repeat, item, condition), result) in enumerate(zip(chunk, results)):
            record = {"trial_id": start + offset, "agent": persona.name, "repeat": repeat,
                      "condition": condition, **result}
            records.append(record)
            with (output / "observations.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        print(f"calibration {len(records)}/{len(trials)} observations", flush=True)
    empirical = backend.name != "fake" and any(r["score"] is not None for r in records)
    summary = calibration_summary(records, battery, empirical=empirical)
    summary.update(elapsed_seconds=time.perf_counter() - started,
                   inference_seconds=time.perf_counter() - inference_started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return output


def runtime_metadata(backend: Backend) -> dict:
    info = {"python": platform.python_version(), "packages": {}}
    for package in ("torch", "transformers", "huggingface_hub", "accelerate"):
        try:
            info["packages"][package] = version(package)
        except PackageNotFoundError:
            info["packages"][package] = None
    if backend.name == "hf":
        import torch
        model = backend._model
        info.update(dtype=str(model.dtype), device=str(model.device),
                    model_revision=getattr(model.config, "_commit_hash", None),
                    cuda=torch.version.cuda)
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(),
                        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    return info


def main() -> None:
    import argparse
    from mindvirus.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--battery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--personas", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        output = run_calibration(cfg.agent_model, Battery.load(args.battery), args.output,
                                 n_personas=args.personas, repeats=args.repeats, seed=cfg.seed,
                                 capture=cfg.capture, batch_size=cfg.probe_batch_size)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"Calibration not run: {exc}\n")
    print(f"Calibration saved to {output}; this stage does not establish contagion.")


if __name__ == "__main__":
    main()
