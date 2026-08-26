from mindvirus.backends import CallLogger, FakeBackend
from mindvirus.judge import judge_text, VERDICTS
from mindvirus.payloads import PAYLOADS

PAYLOAD = PAYLOADS["honesty-absolutism"]


def make_logger(fb, tmp_path):
    return CallLogger(fb, tmp_path / "calls.jsonl")


def test_verdicts_tuple():
    assert VERDICTS == ("endorse", "mention", "oppose", "absent")


def test_judge_parses_verdict(tmp_path):
    fb = FakeBackend(responses={"judge": ["ENDORSE"]})
    v = judge_text(make_logger(fb, tmp_path), PAYLOAD, "Lying is always wrong, period.",
                   agent="Ruth", round=5)
    assert v == "endorse"
    req = fb.requests[0]
    assert req.call_kind == "judge" and req.temperature == 0.0
    assert PAYLOAD.judge_rubric in req.system
    assert "Lying is always wrong, period." in req.messages[0]["content"]


def test_judge_tolerates_padding(tmp_path):
    fb = FakeBackend(responses={"judge": ["Verdict: absent."]})
    assert judge_text(make_logger(fb, tmp_path), PAYLOAD, "nice weather", "Dev", 1) == "absent"


def test_judge_reask_then_error(tmp_path):
    fb = FakeBackend(responses={"judge": ["dunno", "beats me"]})
    assert judge_text(make_logger(fb, tmp_path), PAYLOAD, "x", "Dev", 1) == "error"
    assert len(fb.requests) == 2
