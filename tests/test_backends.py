import json

from mindvirus.backends import CallLogger, FakeBackend, GenRequest


def test_fake_backend_queues_and_records():
    fb = FakeBackend(responses={"agent_turn": ["r1", "r2"]}, default="d")
    req = GenRequest(system="s", messages=[{"role": "user", "content": "u"}],
                     temperature=1.0, max_tokens=100, call_id="c1", call_kind="agent_turn")
    assert fb.generate(req).text == "r1"
    assert fb.generate(req).text == "r2"
    assert fb.generate(req).text == "d"
    assert len(fb.requests) == 3


def test_call_logger_generate_writes_jsonl(tmp_path):
    fb = FakeBackend(default="hello")
    log = CallLogger(fb, tmp_path / "calls.jsonl")
    res = log.generate(system="sys", messages=[{"role": "user", "content": "hi"}],
                       temperature=0.0, max_tokens=64, call_kind="judge",
                       agent="Maria", round=3)
    assert res.text == "hello"
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["call_id"] == "c000001"
    assert entry["kind"] == "judge" and entry["agent"] == "Maria" and entry["round"] == 3
    assert entry["backend"] == "fake" and entry["output"] == "hello"
    assert entry["system"] == "sys" and entry["temperature"] == 0.0


def test_call_logger_choice_logprobs(tmp_path):
    fb = FakeBackend(logprobs={"1": 0.9, "2": 0.1})
    log = CallLogger(fb, tmp_path / "calls.jsonl")
    dist = log.choice_logprobs(system="s", messages=[{"role": "user", "content": "q"}],
                               choices=["1", "2"], call_kind="probe", agent="Dev", round=0)
    assert dist == {"1": 0.9, "2": 0.1}
    entry = json.loads((tmp_path / "calls.jsonl").read_text())
    assert entry["logprobs"] == {"1": 0.9, "2": 0.1} and entry["output"] is None


def test_call_ids_increment(tmp_path):
    log = CallLogger(FakeBackend(), tmp_path / "calls.jsonl")
    for _ in range(3):
        log.generate(system="s", messages=[], temperature=0, max_tokens=1, call_kind="probe")
    ids = [json.loads(l)["call_id"] for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert ids == ["c000001", "c000002", "c000003"]
