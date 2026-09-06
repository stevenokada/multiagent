from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mindvirus.backends import GenRequest
from mindvirus.config import CaptureConfig, ModelConfig
from mindvirus.hf_backend import HFBackend


class StubTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert messages[0]["role"] == "system"
        return "|".join(m["content"] for m in messages) + "<gen>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        n = min(len(text.split("|")) + 3, 8)
        return {"input_ids": torch.arange(n).unsqueeze(0),
                "attention_mask": torch.ones(1, n, dtype=torch.long)}

    def encode(self, text, add_special_tokens=False):
        table = {"1": [11], "2": [12], "3": [13], "4": [14],
                 "5": [15], "6": [16], "7": [17], "xx": [1, 2]}
        return table[text]

    def decode(self, ids, skip_special_tokens=True):
        return "generated!"


class StubModel:
    device = "cpu"

    def __init__(self, vocab=32, hidden=4, layers=3):
        self.vocab, self.hidden, self.layers = vocab, hidden, layers

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.last_input_ids = input_ids.clone()
        new = torch.tensor([[5, 6, 7]])
        return torch.cat([input_ids, new], dim=1)

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        self.last_input_ids = input_ids.clone()
        seq = input_ids.shape[1]
        logits = torch.zeros(1, seq, self.vocab)
        logits[0, -1, 15] = 10.0  # choice "5" dominates
        logits[0, -1, 11] = 8.0
        out = {"logits": logits}
        if output_hidden_states:
            out["hidden_states"] = tuple(
                torch.randn(1, seq, self.hidden) for _ in range(self.layers + 1))
        import types
        return types.SimpleNamespace(**out)


def make_backend(tmp_path, capture=None):
    return HFBackend(
        ModelConfig(backend="hf", model="stub/model"),
        capture=capture, capture_dir=tmp_path / "activations", seed=0,
        tokenizer=StubTokenizer(), model=StubModel(),
    )


def req(kind="agent_turn"):
    return GenRequest(system="sys", messages=[{"role": "user", "content": "hi"}],
                      temperature=1.0, max_tokens=10, call_id="c000001", call_kind=kind)


def test_generate_decodes_new_tokens_only(tmp_path):
    res = make_backend(tmp_path).generate(req())
    assert res.text == "generated!"
    assert res.activation_path is None


def test_choice_logprobs_renormalizes(tmp_path):
    be = make_backend(tmp_path)
    dist = be.choice_logprobs(req("probe"), ["1", "2", "3", "4", "5", "6", "7"])
    assert set(dist) == {"1", "2", "3", "4", "5", "6", "7"}
    assert abs(sum(dist.values()) - 1.0) < 1e-5
    assert dist["5"] > dist["1"] > dist["2"]


def test_choice_logprobs_skips_multitoken(tmp_path):
    dist = make_backend(tmp_path).choice_logprobs(req("probe"), ["1", "xx"])
    assert "xx" not in dist and "1" in dist


def test_capture_last_position(tmp_path):
    cap = CaptureConfig(enabled=True, layers="all", positions="last", calls=["agent_turn"])
    res = make_backend(tmp_path, capture=cap).generate(req("agent_turn"))
    assert res.activation_path is not None
    saved = torch.load(res.activation_path)
    assert set(saved) == {0, 1, 2, 3}          # embeddings + 3 layers
    assert saved[0].shape == (4,)              # [hidden], last position only
    assert saved[0].dtype == torch.float16


def test_capture_respects_kind_and_layers(tmp_path):
    cap = CaptureConfig(enabled=True, layers=[1, 2], positions="all", calls=["probe"])
    be = make_backend(tmp_path, capture=cap)
    assert be.generate(req("agent_turn")).activation_path is None  # kind not captured
    res = be.generate(req("probe"))
    saved = torch.load(res.activation_path)
    assert set(saved) == {1, 2}
    assert saved[1].ndim == 2                  # [seq, hidden]


def test_choice_logprobs_captures_for_probe_calls(tmp_path):
    cap = CaptureConfig(enabled=True, layers="all", positions="last", calls=["probe"])
    be = make_backend(tmp_path, capture=cap)
    dist = be.choice_logprobs(req("probe"), ["1", "2", "3", "4", "5", "6", "7"])
    assert dist is not None
    assert be.last_activation_path is not None
    path = Path(be.last_activation_path)
    assert path.exists()
    saved = torch.load(path)
    assert set(saved) == {0, 1, 2, 3}          # embeddings + 3 layers
    assert saved[0].shape == (4,)              # [hidden], last position only
    assert saved[0].dtype == torch.float16


def test_generate_does_not_capture_when_calls_is_probe_only(tmp_path):
    cap = CaptureConfig(enabled=True, layers="all", positions="last", calls=["probe"])
    be = make_backend(tmp_path, capture=cap)
    res = be.generate(req("agent_turn"))
    assert res.activation_path is None
    assert be.last_activation_path is None


def local_tokenizer(template):
    """Real HF tokenization and Jinja rendering, with no Hub/model downloads."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "hi": 3, "1": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>",
        eos_token="<eos>", chat_template=template,
        model_input_names=["input_ids", "attention_mask"])


CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "[{{ message['role'] }}]{{ message['content'] }}{{ eos_token }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}[assistant]{% endif %}"
)
GEMMA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    + CHAT_TEMPLATE
)


@pytest.mark.parametrize("method", ["generate", "choice_logprobs"])
@pytest.mark.parametrize("template", [CHAT_TEMPLATE, GEMMA_TEMPLATE], ids=["system", "gemma"])
def test_chat_template_inputs_are_tokenized_once(tmp_path, method, template):
    tokenizer = local_tokenizer(template)
    model = StubModel()
    be = HFBackend(ModelConfig(backend="hf", model="local/test"),
                   tokenizer=tokenizer, model=model)
    if method == "generate":
        be.generate(req())
    else:
        be.choice_logprobs(req("probe"), ["1"])
    assert model.last_input_ids[0, 0] == tokenizer.bos_token_id
    assert (model.last_input_ids == tokenizer.bos_token_id).sum() == 1


@pytest.mark.parametrize("gemma", [False, True])
def test_chat_rendering_preserves_system_and_retry_history(tmp_path, gemma):
    tokenizer = local_tokenizer(GEMMA_TEMPLATE if gemma else CHAT_TEMPLATE)
    be = HFBackend(ModelConfig(backend="hf", model="local/test"),
                   tokenizer=tokenizer, model=StubModel())
    request = req()
    request.messages += [{"role": "assistant", "content": "bad JSON"},
                         {"role": "user", "content": "retry"}]
    original = [dict(m) for m in request.messages]
    prompt = be._render(request)
    if gemma:
        assert "[system]" not in prompt
        assert "[user]sys\n\nhi<eos>" in prompt
    else:
        assert "[system]sys<eos>[user]hi<eos>" in prompt
    assert "[assistant]bad JSON<eos>[user]retry<eos>[assistant]" in prompt
    assert request.messages == original  # logging/retries retain the original request


def test_render_does_not_hide_unrelated_template_errors(tmp_path):
    from jinja2.exceptions import TemplateError

    tokenizer = local_tokenizer("{{ raise_exception('Invalid conversation') }}")
    be = HFBackend(ModelConfig(backend="hf", model="local/test"),
                   tokenizer=tokenizer, model=StubModel())
    with pytest.raises(TemplateError, match="Invalid conversation"):
        be._render(req())


def test_generation_uses_configured_seed(tmp_path):
    class SamplingModel(StubModel):
        def generate(self, input_ids=None, **kwargs):
            self.sample = torch.multinomial(torch.ones(32), 12, replacement=True)
            return torch.cat([input_ids, self.sample.unsqueeze(0)], dim=1)

    def sample(seed):
        model = SamplingModel()
        be = HFBackend(ModelConfig(backend="hf", model="stub/model"), seed=seed,
                       tokenizer=StubTokenizer(), model=model)
        be.generate(req())
        return model.sample

    assert torch.equal(sample(42), sample(42))
    assert not torch.equal(sample(42), sample(43))


def test_pinned_revision_reaches_model_and_tokenizer(monkeypatch):
    import transformers
    loaded = []
    def tokenizer_loader(model, **kwargs):
        loaded.append(("tokenizer", kwargs.get("revision")))
        return StubTokenizer()
    def model_loader(model, **kwargs):
        loaded.append(("model", kwargs.get("revision")))
        return StubModel()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_loader)
    HFBackend(ModelConfig("hf", "fixture", revision="a" * 40))
    assert loaded == [("tokenizer", "a" * 40), ("model", "a" * 40)]


@pytest.mark.parametrize("method", ["generate", "choice_logprobs"])
def test_input_limit_rejects_without_truncating_or_running_model(method):
    model = StubModel()
    backend = HFBackend(ModelConfig("hf", "fixture", max_input_tokens=4),
                        tokenizer=StubTokenizer(), model=model)
    with pytest.raises(ValueError, match="input.*limit"):
        if method == "generate":
            backend.generate(req())
        else:
            backend.choice_logprobs(req(), ["1"])
    assert not hasattr(model, "last_input_ids")
