import pytest

torch = pytest.importorskip("torch")

from mindvirus.backends import GenRequest
from mindvirus.config import CaptureConfig, ModelConfig
from mindvirus.hf_backend import HFBackend


class StubTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert messages[0]["role"] == "system"
        return "|".join(m["content"] for m in messages) + "<gen>"

    def __call__(self, text, return_tensors="pt"):
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
        new = torch.tensor([[5, 6, 7]])
        return torch.cat([input_ids, new], dim=1)

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
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
