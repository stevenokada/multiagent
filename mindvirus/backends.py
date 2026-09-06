"""Backend protocol, fake backend for tests, call logging, backend factory."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mindvirus.config import CaptureConfig, ModelConfig


@dataclass
class GenRequest:
    system: str
    messages: list[dict]
    temperature: float
    max_tokens: int
    call_id: str
    call_kind: str


@dataclass
class GenResult:
    text: str
    activation_path: str | None = None


@dataclass
class ChoiceResult:
    probabilities: dict[str, float] | None
    activation_path: str | None = None
    activation_error_type: str | None = None


class Backend(Protocol):
    name: str
    model: str

    def generate(self, req: GenRequest) -> GenResult: ...
    def choice_logprobs(self, req: GenRequest, choices: list[str]) -> dict[str, float] | None: ...


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses: dict[str, list[str]] | None = None,
                 default: str = "ok", logprobs: dict[str, float] | None = None):
        self.responses = responses or {}
        self.default = default
        self.logprobs = logprobs
        self.requests: list[GenRequest] = []

    def generate(self, req: GenRequest) -> GenResult:
        self.requests.append(req)
        queue = self.responses.get(req.call_kind)
        text = queue.pop(0) if queue else self.default
        return GenResult(text=text)

    def choice_logprobs(self, req: GenRequest, choices: list[str]) -> dict[str, float] | None:
        self.requests.append(req)
        return self.logprobs


class CallLogger:
    """Wraps a Backend; mints call_ids and appends every call to calls.jsonl."""

    def __init__(self, backend: Backend, path: Path):
        self.backend = backend
        self.path = Path(path)
        self._n = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"c{self._n:06d}"

    def _write(self, entry: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _base(self, req: GenRequest, agent, round) -> dict:
        return {
            "call_id": req.call_id, "kind": req.call_kind, "agent": agent, "round": round,
            "backend": self.backend.name, "model": self.backend.model,
            "system": req.system, "messages": req.messages,
            "temperature": req.temperature, "max_tokens": req.max_tokens,
        }

    def generate(self, *, system, messages, temperature, max_tokens,
                 call_kind, agent=None, round=None, probe_id=None, answer_mapping=None) -> GenResult:
        req = GenRequest(system, messages, temperature, max_tokens, self._next_id(), call_kind)
        res = self.backend.generate(req)
        entry = self._base(req, agent, round)
        if probe_id is not None:
            entry.update(probe_id=probe_id, answer_mapping=answer_mapping)
        entry.update(output=res.text, activation_path=res.activation_path,
                     choices=None, logprobs=None)
        self._write(entry)
        return res

    def choice_logprobs(self, *, system, messages, choices,
                        call_kind, agent=None, round=None,
                        probe_id=None, answer_mapping=None) -> dict[str, float] | None:
        req = GenRequest(system, messages, 0.0, 1, self._next_id(), call_kind)
        entry = self._base(req, agent, round)
        if probe_id is not None:
            entry.update(probe_id=probe_id, answer_mapping=answer_mapping)
        try:
            dist = self.backend.choice_logprobs(req, choices)
        except Exception as exc:
            entry.update(output=None, activation_path=None, choices=choices,
                         logprobs=None, error_type=type(exc).__name__)
            self._write(entry)
            raise
        entry.update(output=None, activation_path=getattr(self.backend, "last_activation_path", None),
                     choices=choices, logprobs=dist)
        self._write(entry)
        return dist

    def choice_logprobs_batch(self, requests: list[dict], choices: list[str]) -> list[dict | None]:
        """Keep per-request identity; isolate failures if batch execution fails."""
        if not requests:
            return []
        batch = getattr(self.backend, "choice_logprobs_batch", None)
        if callable(batch) and len(requests) > 1:
            reqs = [GenRequest(r["system"], r["messages"], 0.0, 1,
                               self._next_id(), r["call_kind"]) for r in requests]
            try:
                results = batch(reqs, choices)
                if len(results) != len(reqs) or not all(isinstance(r, ChoiceResult) for r in results):
                    raise ValueError("backend returned an invalid batch result")
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Batch evaluation failed (%s); retrying requests individually", type(exc).__name__)
            else:
                for req, metadata, result in zip(reqs, requests, results):
                    entry = self._base(req, metadata.get("agent"), metadata.get("round"))
                    if metadata.get("probe_id") is not None:
                        entry.update(probe_id=metadata["probe_id"],
                                     answer_mapping=metadata.get("answer_mapping"))
                    entry.update(output=None, activation_path=result.activation_path,
                                 choices=choices, logprobs=result.probabilities,
                                 batch_size=len(reqs))
                    if result.activation_error_type:
                        entry["activation_error_type"] = result.activation_error_type
                    self._write(entry)
                return [r.probabilities for r in results]
        results = []
        for request in requests:
            try:
                results.append(self.choice_logprobs(**request, choices=choices))
            except Exception:
                results.append(None)
        return results


class AnthropicBackend:
    """Claude via the Anthropic SDK. No logprobs; probes fall back to sampling."""

    name = "anthropic"

    def __init__(self, model: str, client=None):
        self.model = model
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def generate(self, req: GenRequest) -> GenResult:
        # anthropic>=1.0 removed sampling params (temperature/top_p/top_k);
        # req.temperature is honored only by backends that support it (hf).
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=req.messages,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return GenResult(text=text)

    def choice_logprobs(self, req: GenRequest, choices: list[str]) -> dict[str, float] | None:
        return None


def build_backend(model_cfg: ModelConfig, run_dir: Path,
                  capture: CaptureConfig | None = None, *, seed: int = 0) -> Backend:
    if model_cfg.backend == "fake":
        return FakeBackend()
    if model_cfg.backend == "anthropic":
        return AnthropicBackend(model_cfg.model)
    if model_cfg.backend == "hf":
        from mindvirus.hf_backend import HFBackend
        return HFBackend(model_cfg, capture, run_dir / "activations", seed=seed)
    raise ValueError(f"unknown backend {model_cfg.backend!r}")
