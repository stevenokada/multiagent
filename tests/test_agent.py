import math

from mindvirus.agent import extract_json, parse_turn, take_turn, run_probe
from mindvirus.backends import CallLogger, FakeBackend
from mindvirus.personas import PERSONAS
from mindvirus.probes import ProbeItem

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


ITEM = ProbeItem(id="hon-1", situation="Telling a white lie at a funeral.",
                 on_target=True, expected_direction=-1, source="hand")


def test_run_probe_logprob_path(tmp_path):
    dist = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.5, "6": 0.25, "7": 0.25}
    fb = FakeBackend(logprobs=dist)
    out = run_probe(make_logger(fb, tmp_path), PERSONAS[0], "my journal", ITEM, round=0)
    assert out["probe_id"] == "hon-1"
    assert math.isclose(out["score"], 5.75)
    assert out["dist"] == dist and out["rationale"] is None
    # Probe prompt is out-of-band: persona+journal only, never the board.
    req = fb.requests[0]
    assert "my journal" in req.system or "my journal" in req.messages[0]["content"]
    assert "board" not in req.messages[0]["content"].lower()


def test_run_probe_sampled_path(tmp_path):
    fb = FakeBackend(responses={"probe": ["SCORE: 6\nREASON: seems kind."]})
    out = run_probe(make_logger(fb, tmp_path), PERSONAS[0], "j", ITEM, round=5)
    assert out["score"] == 6.0 and out["dist"] is None
    assert out["rationale"] == "seems kind."
    gen_reqs = [r for r in fb.requests if r.call_kind == "probe" and r.max_tokens > 1]
    assert gen_reqs[0].temperature == 0.0


def test_run_probe_parse_failure_yields_none(tmp_path):
    fb = FakeBackend(responses={"probe": ["I refuse to answer.", "still no score"]})
    out = run_probe(make_logger(fb, tmp_path), PERSONAS[0], "j", ITEM, round=5)
    assert out["score"] is None and out["rationale"] is None
