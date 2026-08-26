from mindvirus.agent import extract_json, parse_turn, take_turn
from mindvirus.backends import CallLogger, FakeBackend
from mindvirus.personas import PERSONAS

GOOD = '{"journal": "I am thinking.", "post": "Hello board."}'
WRAPPED = 'Sure! Here is my response:\n```json\n' + GOOD + '\n```'


def test_extract_json_variants():
    assert extract_json(GOOD) == {"journal": "I am thinking.", "post": "Hello board."}
    assert extract_json(WRAPPED)["post"] == "Hello board."
    assert extract_json("no json here") is None


def test_parse_turn():
    assert parse_turn(GOOD) == ("I am thinking.", "Hello board.")
    assert parse_turn('{"journal": "x"}') is None      # missing post
    assert parse_turn("garbage") is None


def make_logger(fb, tmp_path):
    return CallLogger(fb, tmp_path / "calls.jsonl")


def test_take_turn_success(tmp_path):
    fb = FakeBackend(responses={"agent_turn": [GOOD]})
    out = take_turn(make_logger(fb, tmp_path), PERSONAS[0], "old journal",
                    "[round 0] MODERATOR: hi", round=1, temperature=1.0)
    assert out == ("I am thinking.", "Hello board.")
    req = fb.requests[0]
    assert req.call_kind == "agent_turn"
    assert PERSONAS[0].name in req.system and PERSONAS[0].background in req.system
    assert "old journal" in req.messages[0]["content"]
    assert "MODERATOR: hi" in req.messages[0]["content"]


def test_take_turn_reasks_once_then_none(tmp_path):
    fb = FakeBackend(responses={"agent_turn": ["garbage", "still garbage"]})
    out = take_turn(make_logger(fb, tmp_path), PERSONAS[0], "j", "feed", 1, 1.0)
    assert out is None
    assert len(fb.requests) == 2


def test_take_turn_reask_recovers(tmp_path):
    fb = FakeBackend(responses={"agent_turn": ["garbage", GOOD]})
    out = take_turn(make_logger(fb, tmp_path), PERSONAS[0], "j", "feed", 1, 1.0)
    assert out == ("I am thinking.", "Hello board.")
