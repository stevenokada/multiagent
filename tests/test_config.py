import pytest
from mindvirus.config import Config, ModelConfig, CaptureConfig, load_config, validate_config


def base_cfg(**over):
    kw = dict(
        agent_model=ModelConfig(backend="anthropic", model="claude-haiku-4-5"),
        judge_model=ModelConfig(backend="anthropic", model="claude-haiku-4-5"),
    )
    kw.update(over)
    return Config(**kw)


def test_defaults():
    cfg = base_cfg()
    assert cfg.n_agents == 10
    assert cfg.rounds == 15
    assert cfg.probe_every == 5
    assert cfg.feed_k == 25
    assert cfg.seed == 0
    assert cfg.payload_id == "honesty-absolutism"
    assert cfg.n_patient_zero == 1
    assert cfg.battery_source == "hand"
    assert cfg.agent_temperature == 1.0
    assert cfg.capture.enabled is False


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "n_agents: 4\nrounds: 3\nseed: 7\n"
        "agent_model: {backend: anthropic, model: claude-haiku-4-5}\n"
        "judge_model: {backend: anthropic, model: claude-haiku-4-5}\n"
        "capture: {enabled: false}\n"
    )
    cfg = load_config(p)
    assert cfg.n_agents == 4 and cfg.rounds == 3 and cfg.seed == 7
    assert cfg.agent_model.backend == "anthropic"


@pytest.mark.parametrize("over", [
    {"n_agents": 1}, {"n_agents": 11}, {"rounds": 0},
    {"probe_every": 0}, {"n_patient_zero": 10}, {"payload_id": "nope"},
    {"battery_source": "nope"},
])
def test_validate_rejects(over):
    with pytest.raises(ValueError):
        validate_config(base_cfg(**over))


def test_capture_requires_hf_backend():
    cfg = base_cfg(capture=CaptureConfig(enabled=True))
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_capture_positions_all_warns():
    cfg = base_cfg(
        agent_model=ModelConfig(backend="hf", model="Qwen/Qwen2.5-7B-Instruct"),
        capture=CaptureConfig(enabled=True, positions="all"),
    )
    with pytest.warns(UserWarning):
        validate_config(cfg)
