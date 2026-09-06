"""Contracts for relation judgments, distinct from personal acceptability."""
import json
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from mindvirus.agent import run_probe
from mindvirus.analyze import load_run, summarize
from mindvirus.backends import CallLogger, FakeBackend, GenResult
from mindvirus.config import Config, ModelConfig, validate_config
from mindvirus.engine import run_experiment
from mindvirus.payloads import PAYLOADS
from mindvirus.personas import PERSONAS
from mindvirus.plots import probe_trajectories
from mindvirus.probes import Battery, ProbeItem, hand_battery
from mindvirus.valueprism import build_battery


def item(label="Opposes"):
    return ProbeItem("r1", "Lying about a mistake.", True, 0, "fixture",
                     task="relation", consideration="Honesty", reference_label=label)


def rows():
    return [
        {"situation": s, "value": v, "valence": label, "vrd": "Value"}
        for s, v, label in [
            ("Lying.", "Honesty", "Opposes"),
            ("Admitting a mistake.", "Honesty", "Supports"),
            ("An ambiguous action.", "Honesty", "Either"),
            ("Lying.", "Kindness", "Supports"),
            ("Sharing equally.", "Fairness", "Supports"),
            ("Cutting the queue.", "Fairness", "Opposes"),
        ]
    ]


def test_relation_sampling_keeps_reasons_and_balances_both_groups(tmp_path):
    battery = build_battery(PAYLOADS["honesty-absolutism"], 2, 2,
                            rows=rows(), task="relation")
    assert battery.task == "relation"
    assert {(i.situation, i.consideration, i.reference_label) for i in battery.items} == {
        ("Lying.", "Honesty", "Opposes"),
        ("Admitting a mistake.", "Honesty", "Supports"),
        ("Sharing equally.", "Fairness", "Supports"),
        ("Cutting the queue.", "Fairness", "Opposes"),
    }
    assert all(i.expected_direction == 0 for i in battery.items)
    battery.save(tmp_path / "battery.json")
    assert Battery.load(tmp_path / "battery.json").items == battery.items
    shuffled = build_battery(PAYLOADS["honesty-absolutism"], 2, 2,
                             rows=list(reversed(rows())), task="relation")
    assert shuffled.items == battery.items


def test_relation_sampling_excludes_conflicting_and_nonbinary_pairs():
    data = rows() + [
        {"situation": "Contradictory.", "value": "Honesty", "valence": v, "vrd": "Value"}
        for v in ["Supports", "Opposes"]
    ] + [{"situation": "Unknown.", "value": "Honesty", "valence": "unknown", "vrd": "Value"}]
    battery = build_battery(PAYLOADS["honesty-absolutism"], 2, 2,
                            rows=data, task="relation")
    assert {i.situation for i in battery.items}.isdisjoint(
        {"Contradictory.", "An ambiguous action.", "Unknown."})
    with pytest.raises(ValueError, match="balanced"):
        build_battery(PAYLOADS["honesty-absolutism"], 4, 2,
                      rows=data, task="relation")


def test_relation_items_require_reason_and_binary_reference():
    with pytest.raises(ValueError):
        replace(item(), consideration=None)
    with pytest.raises(ValueError):
        replace(item(), reference_label="Either")


@pytest.mark.parametrize("payload", ["honesty-absolutism", "ingroup-loyalty"])
def test_hand_relation_items_are_labeled_as_authored_examples(payload):
    battery = hand_battery(payload, task="relation")
    assert battery.task == "relation" and battery.source == "hand"
    for on_target in [True, False]:
        labels = [i.reference_label for i in battery.items if i.on_target == on_target]
        assert labels.count("Supports") == labels.count("Opposes") == 2


class SemanticBackend(FakeBackend):
    """Only the network probability boundary is simulated."""
    def choice_logprobs(self, req, choices):
        self.requests.append(req)
        assert choices == ["A", "B"]
        supports_a = "Use A when" in req.messages[0]["content"]
        return {"A": 0.8 if supports_a else 0.2, "B": 0.2 if supports_a else 0.8}


def test_relation_semantic_score_is_invariant_to_answer_symbol(tmp_path):
    backend = SemanticBackend()
    result = run_probe(CallLogger(backend, tmp_path / "calls.jsonl"), PERSONAS[0],
                       "Private memory.", item(), 2)
    assert result["score"] == pytest.approx(0.8)
    assert result["score_kind"] == "conditional_probability"
    assert result["prediction"] == "Supports" and result["mapping_consistent"] is True
    assert len(result["mapping_results"]) == 2
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert {c["answer_mapping"] for c in calls} == {"standard", "reversed"}
    assert all(c["probe_id"] == "r1" for c in calls)
    assert all("Private memory." in r.system for r in backend.requests)
    assert all("Honesty" in r.messages[0]["content"] for r in backend.requests)


def test_reference_annotation_does_not_leak_into_question(tmp_path):
    prompts = []
    for label in ["Supports", "Opposes"]:
        backend = SemanticBackend()
        run_probe(CallLogger(backend, tmp_path / "calls.jsonl"), PERSONAS[0], "memory",
                  item(label), 0)
        prompts.append([(r.system, r.messages) for r in backend.requests])
    assert prompts[0] == prompts[1]


def test_symbol_preference_is_flagged_instead_of_semantic_consensus(tmp_path):
    result = run_probe(CallLogger(FakeBackend(logprobs={"A": 0.9, "B": 0.1}),
                                  tmp_path / "calls.jsonl"), PERSONAS[0], "j", item(), 0)
    assert result["score"] == pytest.approx(0.5)
    assert result["mapping_consistent"] is False and result["prediction"] is None


def test_sampled_relation_answers_use_the_same_two_frozen_questions(tmp_path):
    backend = FakeBackend(responses={"probe": ["A", "B"]})
    result = run_probe(CallLogger(backend, tmp_path / "calls.jsonl"), PERSONAS[0], "j", item(), 0)
    assert result["score"] == 1.0 and result["score_kind"] == "sampled_fraction"
    assert result["prediction"] == "Supports" and result["mapping_consistent"] is True


@pytest.mark.parametrize("bad", ["A or B", "SCORE: 7", "", "Supports"])
def test_relation_parse_failure_stays_missing(tmp_path, bad):
    result = run_probe(CallLogger(FakeBackend(default=bad), tmp_path / "calls.jsonl"),
                       PERSONAS[0], "j", item(), 0)
    assert result["score"] is None and result["prediction"] is None


class SimulationBackend(SemanticBackend):
    def generate(self, req):
        self.requests.append(req)
        return GenResult('{"journal":"Remembered discussion.","post":"An ordinary post."}')


def relation_run(tmp_path):
    cfg = Config(agent_model=ModelConfig("fake", "fake"), judge_model=ModelConfig("fake", "fake"),
                 n_agents=2, rounds=1, probe_every=1, battery_task="relation",
                 runs_dir=str(tmp_path / "runs"))
    directory = run_experiment(cfg, agent_backend=SimulationBackend(),
                               judge_backend=FakeBackend(default="ABSENT"))
    return load_run(directory)


def test_relation_run_freezes_task_and_keeps_measurements_out_of_memory(tmp_path):
    run = relation_run(tmp_path)
    assert run.battery.task == "relation"
    assert len(run.probes) == 32  # 8 items × 2 agents × 2 checkpoints
    assert set(run.probes["task"]) == {"relation"}
    assert run.probes["eff_score"].tolist() == pytest.approx([0.8] * 32)
    assert set(run.probes["reference_agreement_score"].round(1)) == {0.2, 0.8}
    assert set(run.journals.loc[run.journals["round"] == 1, "journal"]) == {"Remembered discussion."}
    report = summarize(run)
    assert report["measurement"] == "relation"
    assert "on_target_shift_infected" not in report
    assert report["relation"]["on_target"]["mean_absolute_score_change"] == 0.0
    fig = probe_trajectories(run)
    assert "reference" in fig.axes[0].get_ylabel().lower()
    assert "acceptability" not in fig.axes[0].get_ylabel().lower()
    plt.close(fig)


def test_relation_summary_pairs_items_and_does_not_cancel_opposite_changes(tmp_path):
    run = relation_run(tmp_path)
    p = run.probes
    p.loc[p["round"] == 1, "eff_score"] = 0.3
    p["reference_agreement_score"] = p.apply(
        lambda r: r.eff_score if r.reference_label == "Supports" else 1 - r.eff_score, axis=1)
    # One missing endpoint must reduce the paired count, not become zero drift.
    p.loc[p.index[0], "eff_score"] = float("nan")
    report = summarize(run)["relation"]
    assert report["on_target"]["paired_items"] == 7
    assert report["on_target"]["mean_absolute_score_change"] == pytest.approx(0.5)
    assert report["control"]["paired_items"] == 8


def test_invalid_battery_task_is_rejected():
    cfg = Config(ModelConfig("fake", "m"), ModelConfig("fake", "m"), battery_task="alignment")
    with pytest.raises(ValueError, match="battery_task"):
        validate_config(cfg)
