import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from mindvirus import plots
from tests.test_analyze import KindedFake, EndorseJudge  # reuse fakes
from mindvirus.analyze import load_run
from mindvirus.config import Config, ModelConfig
from mindvirus.engine import run_experiment


def make_run(tmp_path):
    cfg = Config(
        agent_model=ModelConfig(backend="fake", model="m"),
        judge_model=ModelConfig(backend="fake", model="m"),
        n_agents=3, rounds=2, probe_every=1, seed=0, n_patient_zero=1,
        runs_dir=str(tmp_path / "runs"),
    )
    return load_run(run_experiment(cfg, agent_backend=KindedFake(), judge_backend=EndorseJudge()))


def test_infection_curve_returns_figure(tmp_path):
    fig = plots.infection_curve(make_run(tmp_path))
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() and ax.get_ylabel()


def test_probe_trajectories_two_panels(tmp_path):
    fig = plots.probe_trajectories(make_run(tmp_path))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
