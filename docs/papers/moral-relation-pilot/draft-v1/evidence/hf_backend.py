"""In-process HuggingFace backend: generation, choice logprobs, activation capture."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from mindvirus.backends import ChoiceResult, GenRequest, GenResult
from mindvirus.config import CaptureConfig, ModelConfig


def render_chat_prompt(tokenizer, req: GenRequest) -> str:
    """Render a request, folding system instructions into the user turn for Gemma."""
    from jinja2.exceptions import TemplateError

    msgs = [{"role": "system", "content": req.system}] + req.messages
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
    except TemplateError as exc:
        # Gemma 1/2 templates explicitly reject a separate system role. Only
        # handle that rejection; malformed conversations/templates still fail.
        if ("system role not supported" not in str(exc).lower()
                or not req.messages or req.messages[0]["role"] != "user"):
            raise
        msgs = [dict(m) for m in req.messages]
        msgs[0]["content"] = f"{req.system}\n\n{msgs[0]['content']}"
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)


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
        if self._model is None or self.tokenizer is None:
            self._load()
        torch.manual_seed(seed)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = {"trust_remote_code": self.model_cfg.trust_remote_code}
        if self.model_cfg.revision is not None:
            kw["revision"] = self.model_cfg.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kw)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = "auto" if self.model_cfg.dtype == "auto" else getattr(torch, self.model_cfg.dtype)
        if self.model_cfg.quantize_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto", **kw)

    def _render(self, req: GenRequest) -> str:
        return render_chat_prompt(self.tokenizer, req)

    def _inputs(self, prompt):
        kwargs = {"padding": True} if isinstance(prompt, list) else {}
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, **kwargs)
        limit = self.model_cfg.max_input_tokens
        if limit is not None and int(inputs["attention_mask"].sum(-1).max()) > limit:
            raise ValueError(f"input token limit {limit} exceeded; prompt was not truncated")
        return {k: v.to(self._model.device) for k, v in inputs.items()}

    def _capture_applies(self, req: GenRequest) -> bool:
        return bool(self.capture and req.call_kind in self.capture.calls and self.capture_dir)

    def _do_capture(self, req: GenRequest, inputs, outputs=None, row: int = 0) -> str:
        if outputs is None:
            with self.torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states  # tuple: embeddings + one per layer
        layers = (range(len(hs)) if self.capture.layers == "all"
                  else self.capture.layers)
        saved = {}
        for li in layers:
            t = hs[li][row][inputs["attention_mask"][row].bool()]  # real tokens only
            if self.capture.positions == "last":
                t = t[-1]
            saved[int(li)] = t.detach().to("cpu", self.torch.float16)
        path = self.capture_dir / f"{req.call_id}.pt"
        with NamedTemporaryFile(dir=self.capture_dir, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            self.torch.save(saved, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)

    def generate(self, req: GenRequest) -> GenResult:
        self.last_activation_path = None
        prompt = self._render(req)
        inputs = self._inputs(prompt)
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
        inputs = self._inputs(prompt)
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

    def choice_logprobs_batch(self, requests: list[GenRequest], choices: list[str]) -> list[ChoiceResult]:
        if not requests:
            return []
        self.last_activation_path = None
        inputs = self._inputs([self._render(r) for r in requests])
        mask = inputs["attention_mask"]
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every input must contain at least one real token")
        # Normalize real-token positions for both left and right padding.
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
        capture = any(self._capture_applies(r) for r in requests)
        with self.torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=capture)
        indices = (mask.long() * self.torch.arange(mask.shape[1], device=mask.device)).max(-1).values
        ids, kept = [], []
        for choice in choices:
            tokens = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(tokens) == 1:
                ids.append(tokens[0])
                kept.append(choice)
        # Resolve all distributions before any capture side effects. A failed
        # capture must not erase usable measurements or their original IDs.
        results = []
        for row in range(len(requests)):
            dist = None
            if kept:
                logits = outputs.logits[row, indices[row], ids]
                probs = self.torch.softmax(logits.float(), dim=0)
                dist = {c: float(p) for c, p in zip(kept, probs)}
            results.append(ChoiceResult(dist))
        for row, (request, result) in enumerate(zip(requests, results)):
            if self._capture_applies(request):
                try:
                    result.activation_path = self._do_capture(request, inputs, outputs, row)
                except Exception as exc:
                    result.activation_error_type = type(exc).__name__
        return results
