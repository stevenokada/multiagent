"""Figures from a loaded Run. All functions return matplotlib Figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

from mindvirus.analyze import Run, final_infected, infection_by_checkpoint


def infection_curve(run: Run):
    df = infection_by_checkpoint(run)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["round"], df["n_infected"], marker="o")
    n_agents = run.journals["agent"].nunique()
    ax.set_xlabel("round")
    ax.set_ylabel("infected agents (journal endorses payload)")
    ax.set_ylim(0, n_agents)
    ax.set_title(f"Infection curve — {run.battery.payload_id}")
    return fig


def probe_trajectories(run: Run):
    infected = final_infected(run)
    probes = run.probes.dropna(subset=["eff_score"]).copy()
    probes["group"] = probes["agent"].map(lambda a: "infected" if a in infected else "clean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, on_target, title in ((axes[0], True, "on-target items"),
                                 (axes[1], False, "control items")):
        sub = probes[probes["on_target"] == on_target]
        for group, g in sub.groupby("group"):
            m = g.groupby("round")["eff_score"].mean()
            ax.plot(m.index, m.values, marker="o", label=group)
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.legend()
    axes[0].set_ylabel("mean acceptability (1-7)")
    fig.suptitle(f"Probe trajectories — {run.battery.payload_id}")
    return fig
