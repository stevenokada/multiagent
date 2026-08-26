"""In-process HuggingFace backend: generation, choice logprobs, activation capture."""
from __future__ import annotations

from pathlib import Path

from mindvirus.backends import GenRequest, GenResult
from mindvirus.config import CaptureConfig, ModelConfig


class HFBackend:
    name = "hf"

    def __init__(self, model_cfg: ModelConfig, capture: CaptureConfig | None = None,
                 capture_dir: Path | None = None, seed: int = 0,
                 tokenizer=None, model=None):
        import torch

        self.torch = torch
        self.model_cfg = model_cfg
        self.model_id = model_cfg.model
        self.model = model_cfg.model
        self._model = model
        self.tokenizer = tokenizer
        self.capture = capture if (capture and capture.enabled) else None
        self.capture_dir = Path(capture_dir) if capture_dir else None
        if self.capture and self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.last_activation_path: str | None = None
        torch.manual_seed(seed)
        if self._model is None or self.tokenizer is None:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = {"trust_remote_code": self.model_cfg.trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kw)
        dtype = "auto" if self.model_cfg.dtype == "auto" else getattr(torch, self.model_cfg.dtype)
        if self.model_cfg.quantize_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto", **kw)

    def _render(self, req: GenRequest) -> str:
        msgs = [{"role": "system", "content": req.system}] + req.messages
        return self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)

    def _capture_applies(self, req: GenRequest) -> bool:
        return bool(self.capture and req.call_kind in self.capture.calls and self.capture_dir)

    def _do_capture(self, req: GenRequest, inputs, outputs=None) -> str:
        if outputs is None:
            with self.torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states  # tuple: embeddings + one per layer
        layers = (range(len(hs)) if self.capture.layers == "all"
                  else self.capture.layers)
        saved = {}
        for li in layers:
            t = hs[li][0]  # [seq, hidden]
            if self.capture.positions == "last":
                t = t[-1]
            saved[int(li)] = t.detach().to("cpu", self.torch.float16)
        path = self.capture_dir / f"{req.call_id}.pt"
        self.torch.save(saved, path)
        return str(path)

    def generate(self, req: GenRequest) -> GenResult:
        self.last_activation_path = None
        prompt = self._render(req)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        activation_path = self._do_capture(req, inputs) if self._capture_applies(req) else None
        self.last_activation_path = activation_path
        with self.torch.no_grad():
            out_ids = self._model.generate(
                **inputs, max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenResult(text=text, activation_path=activation_path)

    def choice_logprobs(self, req: GenRequest, choices: list[str]) -> dict[str, float] | None:
        self.last_activation_path = None
        prompt = self._render(req)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        capture_applies = self._capture_applies(req)
        with self.torch.no_grad():
            out = self._model(**inputs, output_hidden_states=capture_applies)
            logits = out.logits[0, -1]
        if capture_applies:
            self.last_activation_path = self._do_capture(req, inputs, outputs=out)
        ids, kept = [], []
        for c in choices:
            toks = self.tokenizer.encode(c, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
                kept.append(c)
        if not kept:
            return None
        sel = logits[ids]
        probs = self.torch.softmax(sel.float(), dim=0)
        return {c: float(p) for c, p in zip(kept, probs)}
