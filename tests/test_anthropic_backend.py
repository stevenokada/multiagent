from types import SimpleNamespace

from mindvirus.backends import AnthropicBackend, GenRequest


class FakeClient:
    def __init__(self):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="text", text="world"),
        ])


def make_req():
    return GenRequest(system="sys", messages=[{"role": "user", "content": "hi"}],
                      temperature=0.5, max_tokens=99, call_id="c1", call_kind="probe")


def test_generate_passes_params_and_joins_text():
    fc = FakeClient()
    be = AnthropicBackend("claude-haiku-4-5", client=fc)
    res = be.generate(make_req())
    assert res.text == "hello world"
    assert res.activation_path is None
    assert fc.kwargs["model"] == "claude-haiku-4-5"
    assert fc.kwargs["system"] == "sys"
    assert fc.kwargs["max_tokens"] == 99
    # anthropic>=1.0 removed sampling params; the backend must not send them
    assert "temperature" not in fc.kwargs
    assert fc.kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_choice_logprobs_is_none():
    be = AnthropicBackend("claude-haiku-4-5", client=FakeClient())
    assert be.choice_logprobs(make_req(), ["1", "2"]) is None
