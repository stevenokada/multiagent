"""Check calibration design and missing-data handling without live models."""
import importlib
import importlib.util
import json

import pytest

from mindvirus.backends import FakeBackend
from mindvirus.config import ModelConfig
from mindvirus.probes import hand_battery


def calibration_module():
    assert importlib.util.find_spec("mindvirus.calibrate") is not None, "calibration runner is missing"
    return importlib.import_module("mindvirus.calibrate")


class StableRelationBackend(FakeBackend):
    def choice_logprobs(self, req, choices):
        self.requests.append(req)
        return {"A": 0.8, "B": 0.2} if "Use A when" in req.messages[0]["content"] else {"A": 0.2, "B": 0.8}


def test_calibration_crosses_personas_items_repeats_and_four_conditions(tmp_path):
    module = calibration_module()
    backend = StableRelationBackend()
    out = module.run_calibration(ModelConfig("fake", "fixture"),
                                 hand_battery("honesty-absolutism", task="relation"),
                                 tmp_path / "calibration", n_personas=2, repeats=2,
                                 backend=backend)
    records = [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]
    assert len(records) == 128  # 2 personas × 2 repeats × 8 questions × 4 conditions
    assert {r["condition"] for r in records} == {"neutral", "repeat", "seeded", "mentioned"}
    assert len({r["trial_id"] for r in records}) == 128
    report = json.loads((out / "summary.json").read_text())
    assert report["empirical_model_run"] is False
    assert report["conditions"]["seeded"]["on_target"]["mean_absolute_change"] == 0.0
    assert report["conditions"]["repeat"]["on_target"]["paired_items"] == 16
    assert report["contagion_established"] is False


def test_calibration_missing_answers_do_not_become_zero_change(tmp_path):
    module = calibration_module()
    out = module.run_calibration(ModelConfig("fake", "fixture"),
                                 hand_battery("honesty-absolutism", task="relation"),
                                 tmp_path / "calibration", n_personas=1, repeats=1,
                                 backend=FakeBackend(default="invalid"))
    report = json.loads((out / "summary.json").read_text())
    target = report["conditions"]["seeded"]["on_target"]
    assert target["paired_items"] == 0 and target["mean_absolute_change"] is None
    assert report["valid_records"] == 0 and report["requested_records"] == 32


@pytest.mark.parametrize("n_personas,repeats", [(0, 1), (11, 1), (1, 0)])
def test_calibration_rejects_empty_design_before_creating_output(tmp_path, n_personas, repeats):
    module = calibration_module()
    with pytest.raises(ValueError):
        module.run_calibration(ModelConfig("fake", "fixture"),
                               hand_battery("honesty-absolutism", task="relation"),
                               tmp_path / "calibration", n_personas=n_personas, repeats=repeats,
                               backend=FakeBackend())
    assert not (tmp_path / "calibration").exists()


def test_calibration_requires_relation_items(tmp_path):
    module = calibration_module()
    with pytest.raises(ValueError, match="relation"):
        module.run_calibration(ModelConfig("fake", "fixture"), hand_battery("honesty-absolutism"),
                               tmp_path / "calibration", backend=FakeBackend())


def test_missing_api_key_fails_without_creating_apparent_run(tmp_path, monkeypatch):
    module = calibration_module()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        module.run_calibration(ModelConfig("anthropic", "claude-haiku-4-5"),
                               hand_battery("honesty-absolutism", task="relation"),
                               tmp_path / "calibration")
    assert not (tmp_path / "calibration").exists()


def test_calibration_capture_and_batching_preserve_output_and_trial_identity(tmp_path, monkeypatch):
    from mindvirus.config import CaptureConfig
    module = calibration_module()
    def build(model, output, capture=None, **kwargs):
        assert capture and capture.enabled
        (output/'activations').mkdir(parents=True)
        return StableRelationBackend()
    monkeypatch.setattr(module,'build_backend',build)
    out=module.run_calibration(ModelConfig('fake','fixture'),
                              hand_battery('honesty-absolutism',task='relation'),
                              tmp_path/'run',n_personas=1,repeats=1,
                              capture=CaptureConfig(enabled=True,calls=['probe']),batch_size=4)
    records=[json.loads(x) for x in (out/'observations.jsonl').read_text().splitlines()]
    assert len(records)==32 and len({r['trial_id'] for r in records})==32
    assert all(r['score']==pytest.approx(.8) for r in records)
    design=json.loads((out/'design.json').read_text())
    assert design['batch_size']==4
    assert (out/'activations').is_dir()


def test_existing_calibration_output_is_rejected_before_loading_model(tmp_path, monkeypatch):
    module=calibration_module()
    output=tmp_path/'existing'; output.mkdir()
    def load(*args,**kwargs):
        pytest.fail('model loaded before rejecting existing artifacts')
    monkeypatch.setattr(module,'build_backend',load)
    with pytest.raises(FileExistsError):
        module.run_calibration(ModelConfig('fake','fixture'),
                               hand_battery('honesty-absolutism',task='relation'),output)
