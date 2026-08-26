import json

import pytest

from mindvirus.analyze import final_infected, infection_by_checkpoint, load_run, summarize
from mindvirus.backends import FakeBackend, GenResult
from mindvirus.config import Config, ModelConfig
from mindvirus.engine import run_experiment

TURN = '{"journal": "hmm", "post": "hello"}'


class KindedFake(FakeBackend):
    def generate(self, req):
        self.requests.append(req)
        if req.call_kind == "probe":
            return GenResult(text="SCORE: 3\nREASON: eh.")
        return GenResult(text=TURN)


class EndorseJudge(FakeBackend):
    """Journals of Maria endorse from round >=2; everything else absent."""
    def generate(self, req):
        self.requests.append(req)
        text = req.messages[0]["content"]
        if req.call_kind == "judge" and "hmm" in text and req.temperature == 0.0:
            # after round 1 every journal is "hmm"; mark Maria's endorse
            return GenResult(text="ENDORSE")
        return GenResult(text="ABSENT")


@pytest.fixture
def run_dir(tmp_path):
    cfg = Config(
        agent_model=ModelConfig(backend="fake", model="m"),
        judge_model=ModelConfig(backend="fake", model="m"),
        n_agents=3, rounds=2, probe_every=1, seed=0, n_patient_zero=1,
        runs_dir=str(tmp_path / "runs"),
    )
    return run_experiment(cfg, agent_backend=KindedFake(), judge_backend=EndorseJudge())


def test_load_run_dataframes(run_dir):
    run = load_run(run_dir)
    assert set(run.posts.columns) >= {"round", "author", "text"}
    assert set(run.probes.columns) >= {"round", "agent", "probe_id", "score",
                                       "on_target", "expected_direction", "eff_score"}
    assert (run.probes["eff_score"] == 3.0).all()
    assert len(run.battery.items) == 6


def test_infection_by_checkpoint(run_dir):
    run = load_run(run_dir)
    df = infection_by_checkpoint(run)
    assert list(df["round"]) == [0, 1, 2]
    # EndorseJudge marks all "hmm" journals as endorse; round 0 journals differ.
    assert df.iloc[0]["n_infected"] == 0
    assert df.iloc[-1]["n_infected"] == 3
    assert final_infected(run) == {"Maria", "Frank", "Ruth"}


def test_summarize_keys(run_dir):
    s = summarize(load_run(run_dir))
    for k in ("peak_infected", "final_infected", "on_target_shift_infected",
              "on_target_shift_clean", "control_shift", "first_endorse_post_round"):
        assert k in s
    assert s["peak_infected"] == 3
    # scores constant at 3.0 -> all shifts zero
    assert s["on_target_shift_infected"] == 0.0
    assert s["control_shift"] == 0.0


def test_eff_score_prefers_dist(tmp_path, run_dir):
    run = load_run(run_dir)
    probes_path = run_dir / "probes.jsonl"
    lines = probes_path.read_text().splitlines()
    obj = json.loads(lines[0])
    obj["dist"] = {"1": 0.5, "7": 0.5}
    obj["score"] = 99.0  # must be ignored when dist present
    lines[0] = json.dumps(obj)
    probes_path.write_text("\n".join(lines) + "\n")
    run2 = load_run(run_dir)
    assert run2.probes.iloc[0]["eff_score"] == 4.0
