import json
from pathlib import Path

from mindvirus.backends import FakeBackend, GenResult
from mindvirus.config import Config, ModelConfig
from mindvirus.engine import run_experiment
from mindvirus.payloads import PAYLOADS


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def make_cfg(tmp_path, **over):
    kw = dict(
        agent_model=ModelConfig(backend="fake", model="fake-model"),
        judge_model=ModelConfig(backend="fake", model="fake-model"),
        n_agents=3, rounds=4, probe_every=2, seed=1, n_patient_zero=1,
        runs_dir=str(tmp_path / "runs"),
    )
    kw.update(over)
    return Config(**kw)


TURN = '{"journal": "thinking about honesty", "post": "interesting discussion"}'


class KindedFake(FakeBackend):
    """Agent backend whose reply depends on call_kind: probes always score 4."""

    def generate(self, req):
        self.requests.append(req)
        if req.call_kind == "probe":
            return GenResult(text="SCORE: 4\nREASON: neutral.")
        return GenResult(text=TURN)


def run(tmp_path, **over):
    cfg = make_cfg(tmp_path, **over)
    agent_kb = KindedFake()
    judge_fb = FakeBackend(default="ABSENT")
    run_dir = run_experiment(cfg, agent_backend=agent_kb, judge_backend=judge_fb)
    return cfg, run_dir, agent_kb, judge_fb


def test_run_dir_artifacts(tmp_path):
    cfg, run_dir, agent_fb, judge_fb = run(tmp_path)
    for name in ("config.yaml", "battery.json", "board.jsonl", "journals.jsonl",
                 "probes.jsonl", "judgements.jsonl", "calls.jsonl"):
        assert (run_dir / name).exists(), name


def test_board_and_journals(tmp_path):
    cfg, run_dir, *_ = run(tmp_path)
    board = read_jsonl(run_dir / "board.jsonl")
    assert board[0]["author"] == "MODERATOR"
    agent_posts = [b for b in board if b["author"] != "MODERATOR"]
    assert len(agent_posts) == cfg.n_agents * cfg.rounds
    journals = read_jsonl(run_dir / "journals.jsonl")
    # initial journals (round 0) + one per agent per round
    assert len(journals) == cfg.n_agents * (cfg.rounds + 1)


def test_patient_zero_seeded(tmp_path):
    cfg, run_dir, *_ = run(tmp_path)
    initial = [j for j in read_jsonl(run_dir / "journals.jsonl") if j["round"] == 0]
    belief = PAYLOADS[cfg.payload_id].belief
    seeded = [j for j in initial if belief in j["journal"]]
    assert len(seeded) == cfg.n_patient_zero


def test_control_run_has_no_seed(tmp_path):
    cfg, run_dir, *_ = run(tmp_path, n_patient_zero=0)
    initial = [j for j in read_jsonl(run_dir / "journals.jsonl") if j["round"] == 0]
    belief = PAYLOADS[cfg.payload_id].belief
    assert not any(belief in j["journal"] for j in initial)


def test_probe_checkpoints(tmp_path):
    cfg, run_dir, *_ = run(tmp_path)  # rounds=4, probe_every=2 -> checkpoints 0,2,4
    probes = read_jsonl(run_dir / "probes.jsonl")
    rounds = sorted({p["round"] for p in probes})
    assert rounds == [0, 2, 4]
    battery_items = 6  # hand battery
    assert len(probes) == 3 * cfg.n_agents * battery_items
    assert all(p["score"] == 4.0 for p in probes)


def test_judgements_cover_journals_and_posts(tmp_path):
    cfg, run_dir, *_ = run(tmp_path)
    judgements = read_jsonl(run_dir / "judgements.jsonl")
    jj = [j for j in judgements if j["kind"] == "journal"]
    jp = [j for j in judgements if j["kind"] == "post"]
    assert len(jj) == 3 * cfg.n_agents            # checkpoints 0,2,4
    assert len(jp) == cfg.n_agents * cfg.rounds   # every agent post judged exactly once
    assert all(j["verdict"] == "absent" for j in judgements)


def test_deterministic_order_given_seed(tmp_path):
    _, _, fb1, _ = run(tmp_path, runs_dir=str(tmp_path / "r1"))
    _, _, fb2, _ = run(tmp_path, runs_dir=str(tmp_path / "r2"))
    order1 = [r.call_id for r in fb1.requests[:5]]
    calls1 = next((tmp_path / "r1").glob("*/calls.jsonl"))
    calls2 = next((tmp_path / "r2").glob("*/calls.jsonl"))
    agents1 = [json.loads(l)["agent"] for l in calls1.read_text().splitlines()[:3]]
    agents2 = [json.loads(l)["agent"] for l in calls2.read_text().splitlines()[:3]]
    assert agents1 == agents2
