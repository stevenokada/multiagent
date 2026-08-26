"""Experiment configuration: dataclasses, YAML loading, validation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    backend: str  # "anthropic" | "hf" | "fake"
    model: str
    dtype: str = "auto"
    quantize_4bit: bool = False
    trust_remote_code: bool = False


@dataclass
class CaptureConfig:
    enabled: bool = False
    layers: str | list[int] = "all"   # "all" or explicit layer indices
    positions: str = "last"           # "last" (final prompt token) | "all"
    calls: list[str] = field(default_factory=lambda: ["agent_turn"])


@dataclass
class Config:
    agent_model: ModelConfig
    judge_model: ModelConfig
    n_agents: int = 10
    rounds: int = 15
    probe_every: int = 5
    feed_k: int = 25
    topic_every: int = 5
    seed: int = 0
    payload_id: str = "honesty-absolutism"
    n_patient_zero: int = 1           # 0 = control run
    battery_source: str = "hand"      # "hand" | "valueprism"
    runs_dir: str = "runs"
    agent_temperature: float = 1.0
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for key, cls in (("agent_model", ModelConfig), ("judge_model", ModelConfig)):
        if key in raw:
            raw[key] = cls(**raw[key])
    if "capture" in raw:
        raw["capture"] = CaptureConfig(**raw["capture"])
    return Config(**raw)


def validate_config(cfg: Config) -> None:
    from mindvirus.personas import PERSONAS
    from mindvirus.payloads import PAYLOADS

    if not 2 <= cfg.n_agents <= len(PERSONAS):
        raise ValueError(f"n_agents must be 2..{len(PERSONAS)}")
    if cfg.rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not 1 <= cfg.probe_every <= cfg.rounds:
        raise ValueError("probe_every must be 1..rounds")
    if not 0 <= cfg.n_patient_zero < cfg.n_agents:
        raise ValueError("n_patient_zero must be 0..n_agents-1")
    if cfg.payload_id not in PAYLOADS:
        raise ValueError(f"unknown payload_id {cfg.payload_id!r}")
    if cfg.battery_source not in ("hand", "valueprism"):
        raise ValueError("battery_source must be 'hand' or 'valueprism'")
    for mc in (cfg.agent_model, cfg.judge_model):
        if mc.backend not in ("anthropic", "hf", "fake"):
            raise ValueError(f"unknown backend {mc.backend!r}")
    if cfg.capture.enabled and cfg.agent_model.backend != "hf":
        raise ValueError("activation capture requires agent_model.backend == 'hf'")
    if cfg.capture.positions not in ("last", "all"):
        raise ValueError("capture.positions must be 'last' or 'all'")
    if not set(cfg.capture.calls) <= {"agent_turn", "probe", "judge"}:
        raise ValueError("capture.calls entries must be agent_turn|probe|judge")
    if cfg.capture.enabled and cfg.capture.positions == "all":
        warnings.warn("capture.positions='all' saves every token position; artifacts will be large")
