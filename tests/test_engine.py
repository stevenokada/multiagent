import json
from pathlib import Path

import pytest

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


def test_identical_hf_agent_and_judge_share_one_loaded_model(tmp_path, monkeypatch):
    loaded = []
    def build(model, *args, **kwargs):
        backend = KindedFake()
        loaded.append(backend)
        return backend
    monkeypatch.setattr("mindvirus.engine.build_backend", build)
    cfg = make_cfg(tmp_path, agent_model=ModelConfig("hf", "same"),
                   judge_model=ModelConfig("hf", "same"), rounds=1, probe_every=1)
    directory = run_experiment(cfg)
    assert len(loaded) == 1
    assert {r.call_kind for r in loaded[0].requests} >= {"agent_turn", "judge", "probe"}
    calls = read_jsonl(directory / "calls.jsonl")
    assert len({r["call_id"] for r in calls}) == len(calls)


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


def test_post_judgement_round_matches_post_round(tmp_path):
    cfg, run_dir, *_ = run(tmp_path)
    judgements = read_jsonl(run_dir / "judgements.jsonl")
    board = read_jsonl(run_dir / "board.jsonl")
    board_posts = {(b["author"], b["text"], b["round"]) for b in board}
    post_judgements = [j for j in judgements if j["kind"] == "post"]
    assert post_judgements
    for j in post_judgements:
        assert 1 <= j["round"] <= cfg.rounds
        assert (j["agent"], j["text"], j["round"]) in board_posts


class FlakyFake(FakeBackend):
    """Agent backend that always raises for one persona; otherwise behaves like KindedFake."""

    def generate(self, req):
        if "Ruth" in req.system:
            raise RuntimeError("simulated persistent backend failure")
        self.requests.append(req)
        if req.call_kind == "probe":
            return GenResult(text="SCORE: 4\nREASON: neutral.")
        return GenResult(text=TURN)


def test_run_survives_persistent_agent_failure(tmp_path):
    cfg = make_cfg(tmp_path)
    agent_kb = FlakyFake()
    judge_fb = FakeBackend(default="ABSENT")
    run_dir = run_experiment(cfg, agent_backend=agent_kb, judge_backend=judge_fb)

    for name in ("config.yaml", "battery.json", "board.jsonl", "journals.jsonl",
                 "probes.jsonl", "judgements.jsonl", "calls.jsonl"):
        assert (run_dir / name).exists(), name

    board = read_jsonl(run_dir / "board.jsonl")
    agent_posts = [b for b in board if b["author"] != "MODERATOR"]
    authors = {b["author"] for b in agent_posts}
    assert "Ruth" not in authors           # the failing agent never posted
    assert authors                          # other agents' posts still exist

    probes = read_jsonl(run_dir / "probes.jsonl")
    ruth_probes = [p for p in probes if p["agent"] == "Ruth"]
    assert ruth_probes
    assert all(p["score"] is None and p["rationale"] is None and p["dist"] is None
              for p in ruth_probes)
    other_probes = [p for p in probes if p["agent"] != "Ruth"]
    assert any(p["score"] is not None for p in other_probes)


def turn_order(run_dir):
    return [(c["round"], c["agent"]) for c in read_jsonl(run_dir / "calls.jsonl")
            if c["kind"] == "agent_turn"]


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_matched_controls_preserve_all_turn_orders(tmp_path, seed):
    orders = []
    for n_patient_zero in (0, 1, 2):
        _, run_dir, *_ = run(tmp_path, seed=seed, n_patient_zero=n_patient_zero,
                            runs_dir=str(tmp_path / str(n_patient_zero)))
        orders.append(turn_order(run_dir))
    assert len(orders[0]) == 12  # all three agents in all four rounds
    assert orders[0] == orders[1] == orders[2]


def test_deterministic_order_given_seed(tmp_path):
    _, first, *_ = run(tmp_path, seed=42, runs_dir=str(tmp_path / "first"))
    _, repeat, *_ = run(tmp_path, seed=42, runs_dir=str(tmp_path / "repeat"))
    _, different, *_ = run(tmp_path, seed=43, runs_dir=str(tmp_path / "different"))
    assert turn_order(first) == turn_order(repeat)
    assert turn_order(first) != turn_order(different)


def test_engine_passes_seed_to_both_backends(tmp_path, monkeypatch):
    seeds = []

    def factory(model_cfg, run_dir, capture=None, *, seed=0):
        seeds.append(seed)
        return KindedFake() if len(seeds) == 1 else FakeBackend(default="ABSENT")

    monkeypatch.setattr("mindvirus.engine.build_backend", factory)
    run_experiment(make_cfg(tmp_path, seed=42))
    assert seeds == [42, 42]
