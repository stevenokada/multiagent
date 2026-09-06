"""Run isolated, exact-model calibration or simulation workers on assigned GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

from mindvirus.backends import build_backend
from mindvirus.calibrate import run_calibration, runtime_metadata
from mindvirus.config import load_config, validate_config
from mindvirus.engine import run_experiment
from mindvirus.personas import PERSONAS
from mindvirus.probes import Battery

EXACT_MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct":
        ("0e9e39f249a16976918f6564b8830bc894c89659", 20),
    "google/gemma-2-9b-it":
        ("11c9b309abf73637e4b6f9a3fa1e92e615547819", 28),
}


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {p.name: _sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _validate_pilot_config(cfg, battery: Battery) -> None:
    validate_config(cfg)
    if cfg.battery_task != "relation" or battery.task != "relation":
        raise ValueError("the pilot requires a relation battery")
    if cfg.payload_id != battery.payload_id:
        raise ValueError("config and battery payloads differ")
    model = cfg.agent_model
    if cfg.judge_model != model:
        raise ValueError("each worker requires matching agent and judge models")
    if model.backend == "fake":
        return
    if model.backend != "hf" or model.model not in EXACT_MODELS:
        raise ValueError("use an exact Jeff HF checkpoint or a fake offline fixture")
    revision, layer = EXACT_MODELS[model.model]
    if (model.revision != revision or model.dtype != "bfloat16"
            or model.quantize_4bit or model.trust_remote_code):
        raise ValueError("the pilot requires the pinned checkpoint in unquantized BF16")
    if (not cfg.capture.enabled or cfg.capture.layers != [layer]
            or cfg.capture.positions != "last" or cfg.capture.calls != ["probe"]):
        raise ValueError("capture must select the exact probe layer and final prompt token")
    if model.max_input_tokens is None:
        raise ValueError("set an explicit input-token limit for the pilot")


def run_sweep(config_paths, devices, battery_path, output, *, stage="calibration",
              seeds=(0,), personas=3, repeats=3, dry_run=False,
              timeout_seconds=3600) -> Path:
    """Freeze inputs, start every worker, and propagate failures without overwrites."""
    configs = [load_config(path) for path in config_paths]
    devices, seeds = list(map(str, devices)), list(seeds)
    if (not configs or len(configs) != len(devices) or len(set(devices)) != len(devices)
            or any(not re.fullmatch(r"[0-9]+|GPU-[a-zA-Z0-9-]+", d) for d in devices)):
        raise ValueError("assign one distinct GPU device to each config")
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    if stage not in ("calibration", "simulation"):
        raise ValueError("stage must be calibration or simulation")
    if not 1 <= personas <= len(PERSONAS) or repeats < 1 or timeout_seconds <= 0:
        raise ValueError("invalid persona count, repeats, or timeout")
    battery = Battery.load(battery_path)
    for cfg in configs:
        _validate_pilot_config(cfg, battery)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_battery = output / "battery.json"
    frozen_battery.write_bytes(Path(battery_path).read_bytes())
    workers = []
    for index, (cfg, device) in enumerate(zip(configs, devices)):
        directory = output / f"worker-{index}"
        directory.mkdir()
        config = directory / "config.yaml"
        config.write_text(yaml.safe_dump(cfg.to_dict()))
        jobs = []
        for seed in seeds:
            for arm in (["calibration"] if stage == "calibration" else ["seeded", "control"]):
                jobs.append({"seed": seed, "arm": arm,
                             "output": f"worker-{index}/seed-{seed}/{arm}"})
        workers.append({"index": index, "device": device,
                        "config": str(config.relative_to(output)),
                        "config_sha256": _sha(config), "jobs": jobs})
    manifest = {"schema_version": 1, "stage": stage, "personas": personas, "repeats": repeats,
                "timeout_seconds": timeout_seconds, "battery": "battery.json",
                "battery_sha256": _sha(frozen_battery), "source_sha256": _sources(),
                "workers": workers}
    manifest_path = output / "manifest.json"
    _write(manifest_path, manifest)
    status = {"state": "planned" if dry_run else "running", "workers": []}
    _write(output / "status.json", status)
    if dry_run:
        return output
    processes, logs = [], []
    deadline = time.monotonic() + timeout_seconds
    try:
        for worker in workers:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = worker["device"]
            stream = (output / f"worker-{worker['index']}" / "worker.log").open("w")
            logs.append(stream)
            process = subprocess.Popen(
                [sys.executable, "-m", "mindvirus.sweep", "--worker-manifest", str(manifest_path),
                 "--worker-index", str(worker["index"])],
                env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            status["workers"].append({"index": worker["index"], "pid": process.pid,
                                      "exit_code": None})
            _write(output / "status.json", status)
        for process, row in zip(processes, status["workers"]):
            row["exit_code"] = process.wait(timeout=max(.001, deadline - time.monotonic()))
            _write(output / "status.json", status)
        if any(w["exit_code"] != 0 for w in status["workers"]):
            raise RuntimeError("one or more workers failed; inspect worker status and logs")
        status["state"] = "completed"
    except BaseException as exc:
        status.update(state="failed", error_type=type(exc).__name__)
        for process, row in zip(processes, status["workers"]):
            if row["exit_code"] is not None:
                continue
            if process.poll() is None:
                process.terminate()
            try:
                row["exit_code"] = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                row["exit_code"] = process.wait()
        raise
    finally:
        _write(output / "status.json", status)
        for stream in logs:
            stream.close()
    return output


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("manifest path escapes the sweep directory")
    return path


def _run_worker(manifest_path: Path, index: int) -> None:
    root = manifest_path.resolve().parent
    manifest = json.loads(manifest_path.read_text())
    worker = manifest["workers"][index]
    status_path = root / f"worker-{index}" / "status.json"
    # Claim this worker once, including failed attempts. A fresh sweep directory
    # is required for a retry so completed or partial research records survive.
    with (status_path.parent / "started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
    status = {"state": "running", "pid": os.getpid(), "jobs": []}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != worker["device"]:
            raise ValueError("worker GPU assignment changed")
        config_path = _inside(root, worker["config"])
        battery_path = _inside(root, manifest["battery"])
        if (_sha(config_path) != worker["config_sha256"]
                or _sha(battery_path) != manifest["battery_sha256"]
                or _sources() != manifest["source_sha256"]):
            raise ValueError("frozen config, battery or source code changed")
        cfg, battery = load_config(config_path), Battery.load(battery_path)
        _validate_pilot_config(cfg, battery)
        if cfg.agent_model.backend == "hf":
            import torch
            if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                    or not torch.cuda.is_bf16_supported()
                    or torch.cuda.get_device_properties(0).total_memory < 23000 * 1024**2):
                raise RuntimeError("worker requires one BF16-capable GPU with at least 23000 MiB")
        loaded = time.perf_counter()
        backend = build_backend(cfg.agent_model, status_path.parent, cfg.capture,
                                seed=worker["jobs"][0]["seed"])
        if backend.name == "hf":
            if (backend._model.dtype != torch.bfloat16
                    or any(p.device.type != "cuda" for p in backend._model.parameters())
                    or backend._model.config._commit_hash != cfg.agent_model.revision):
                raise RuntimeError("loaded model differs from the exact GPU/BF16 checkpoint contract")
        status.update(model_load_seconds=time.perf_counter() - loaded,
                      runtime=runtime_metadata(backend))
        _write(status_path, status)
        for job in worker["jobs"]:
            if backend.name == "hf":
                torch.manual_seed(job["seed"])
                torch.cuda.manual_seed_all(job["seed"])
            destination = _inside(root, job["output"])
            started = time.perf_counter()
            if manifest["stage"] == "calibration":
                result = run_calibration(
                    cfg.agent_model, battery, destination, n_personas=manifest["personas"],
                    repeats=manifest["repeats"], seed=job["seed"], backend=backend,
                    capture=cfg.capture, batch_size=cfg.probe_batch_size)
            else:
                run_cfg = replace(cfg, seed=job["seed"], runs_dir=str(destination),
                                  n_patient_zero=1 if job["arm"] == "seeded" else 0)
                result = run_experiment(run_cfg, agent_backend=backend,
                                        judge_backend=backend, battery=battery)
            status["jobs"].append({**job, "state": "completed",
                                   "result": str(result.relative_to(root)),
                                   "elapsed_seconds": time.perf_counter() - started})
            _write(status_path, status)
        status["state"] = "completed"
    except BaseException as exc:
        status.update(state="failed", error_type=type(exc).__name__)
        raise
    finally:
        _write(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["calibration", "simulation"], default="calibration")
    parser.add_argument("--config", action="append")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--battery")
    parser.add_argument("--output")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--personas", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_manifest is not None:
        if args.worker_index is None or args.worker_index < 0:
            parser.error("worker index must be nonnegative")
        try:
            _run_worker(args.worker_manifest, args.worker_index)
        except Exception as exc:
            parser.exit(1, f"Worker failed: {type(exc).__name__}; see status.json\n")
        return
    if not all((args.config, args.devices, args.battery, args.output)):
        parser.error("config, devices, battery and output are required")
    try:
        output = run_sweep(args.config, args.devices, args.battery, args.output, stage=args.stage,
                           seeds=args.seeds, personas=args.personas, repeats=args.repeats,
                           dry_run=args.dry_run, timeout_seconds=args.timeout_seconds)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Sweep failed: {type(exc).__name__}: {exc}\n")
    print(f"{'Plan' if args.dry_run else 'Worker outputs'} saved to {output}")


if __name__ == "__main__":
    main()
