"""Load a run dir into DataFrames; infection and probe-shift summaries."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from mindvirus.probes import Battery


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return pd.DataFrame(rows)


@dataclass
class Run:
    dir: Path
    config: dict
    battery: Battery
    posts: pd.DataFrame
    journals: pd.DataFrame
    probes: pd.DataFrame
    judgements: pd.DataFrame


def _eff_score(row) -> float | None:
    dist = row.get("dist")
    if isinstance(dist, dict) and dist:
        total = sum(dist.values())
        if total > 0:
            return sum(int(k) * v / total for k, v in dist.items())
    return row.get("score")


def load_run(run_dir: str | Path) -> Run:
    d = Path(run_dir)
    battery = Battery.load(d / "battery.json")
    probes = _read_jsonl(d / "probes.jsonl")
    if not probes.empty:
        item_info = {i.id: i for i in battery.items}
        probes["on_target"] = probes["probe_id"].map(lambda pid: item_info[pid].on_target)
        probes["expected_direction"] = probes["probe_id"].map(
            lambda pid: item_info[pid].expected_direction)
        probes["task"] = probes["probe_id"].map(lambda pid: item_info[pid].task)
        probes["consideration"] = probes["probe_id"].map(lambda pid: item_info[pid].consideration)
        probes["reference_label"] = probes["probe_id"].map(lambda pid: item_info[pid].reference_label)
        probes["eff_score"] = probes.apply(_eff_score, axis=1)
        if battery.task == "relation":
            probes["reference_agreement_score"] = probes["eff_score"].where(
                probes["reference_label"] == "Supports", 1 - probes["eff_score"])
    return Run(
        dir=d,
        config=yaml.safe_load((d / "config.yaml").read_text()),
        battery=battery,
        posts=_read_jsonl(d / "board.jsonl"),
        journals=_read_jsonl(d / "journals.jsonl"),
        probes=probes,
        judgements=_read_jsonl(d / "judgements.jsonl"),
    )


def infection_by_checkpoint(run: Run) -> pd.DataFrame:
    jj = run.judgements[run.judgements["kind"] == "journal"]
    rows = []
    for rnd, grp in jj.groupby("round"):
        infected = sorted(grp.loc[grp["verdict"] == "endorse", "agent"])
        rows.append({"round": rnd, "n_infected": len(infected), "infected": infected})
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def final_infected(run: Run) -> set[str]:
    df = infection_by_checkpoint(run)
    return set(df.iloc[-1]["infected"]) if not df.empty else set()


def _mean_shift(probes: pd.DataFrame, agents: set[str], on_target: bool, signed: bool) -> float:
    sub = probes[(probes["agent"].isin(agents)) & (probes["on_target"] == on_target)]
    sub = sub.dropna(subset=["eff_score"])
    if sub.empty:
        return 0.0
    first, last = sub["round"].min(), sub["round"].max()
    sign = sub["expected_direction"] if signed else 1
    sub = sub.assign(v=sub["eff_score"] * sign)
    return float(sub[sub["round"] == last]["v"].mean() - sub[sub["round"] == first]["v"].mean())


def summarize(run: Run) -> dict:
    inf = infection_by_checkpoint(run)
    infected = final_infected(run)
    all_agents = set(run.journals["agent"].unique())
    clean = all_agents - infected
    jp = run.judgements[(run.judgements["kind"] == "post")
                        & (run.judgements["verdict"] == "endorse")]
    first_endorse = {a: int(g["round"].min()) for a, g in jp.groupby("agent")}
    if run.battery.task == "relation":
        return {
            "measurement": "relation",
            "baseline": "round zero, after patient-zero seeding",
            "relation": relation_summary(run),
            "journal_endorsement": {
                "peak_agents": int(inf["n_infected"].max()) if not inf.empty else 0,
                "final_agents": sorted(infected), "first_endorse_post_round": first_endorse,
            },
        }
    return {
        "peak_infected": int(inf["n_infected"].max()) if not inf.empty else 0,
        "final_infected": sorted(infected),
        "on_target_shift_infected": _mean_shift(run.probes, infected, True, signed=True),
        "on_target_shift_clean": _mean_shift(run.probes, clean, True, signed=True),
        "control_shift": _mean_shift(run.probes, all_agents, False, signed=False),
        "first_endorse_post_round": first_endorse,
    }


def relation_summary(run: Run) -> dict:
    """Paired item changes; annotations are reference labels, not moral truth.

    Agent-item pairs are descriptive observations, not independent simulation
    replicates. Opposite movements must not cancel in the drift magnitude.
    """
    probes = run.probes
    first, last = probes["round"].min(), probes["round"].max()
    result = {}
    for name, on_target in (("on_target", True), ("control", False)):
        sub = probes[probes["on_target"] == on_target].copy()
        if "prediction" not in sub:
            sub["prediction"] = None
        columns = ["agent", "probe_id", "eff_score", "reference_label", "prediction"]
        paired = sub.loc[sub["round"] == first, columns].merge(
            sub.loc[sub["round"] == last, columns], on=["agent", "probe_id"],
            suffixes=("_before", "_after"), validate="one_to_one")
        paired = paired.dropna(subset=["eff_score_before", "eff_score_after"])
        delta = paired["eff_score_after"] - paired["eff_score_before"]
        reference_delta = delta.where(paired["reference_label_before"] == "Supports", -delta)
        robust = paired.dropna(subset=["prediction_before", "prediction_after"])
        result[name] = {
            "paired_items": len(paired),
            "mean_absolute_score_change": float(delta.abs().mean()) if len(paired) else None,
            "mean_reference_agreement_change": float(reference_delta.mean()) if len(paired) else None,
            "robust_paired_items": len(robust),
            "robust_flip_rate": float((robust["prediction_before"] != robust["prediction_after"]).mean())
            if len(robust) else None,
            "checkpoints": [
                {"round": int(rnd), "requested": len(group),
                 "scored": int(group["eff_score"].notna().sum()),
                 "mean_reference_agreement_score": float(group["reference_agreement_score"].mean())
                 if group["reference_agreement_score"].notna().any() else None,
                 "mapping_disagreements": int(group.get("mapping_consistent", pd.Series(dtype=object))
                                              .eq(False).sum())}
                for rnd, group in sub.groupby("round")
            ],
        }
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Summarize a mind-virus run")
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run = load_run(args.run_dir)
    print(json.dumps(summarize(run), indent=2))
    from mindvirus import plots  # noqa: E402
    for fname, fig in (("infection.png", plots.infection_curve(run)),
                       ("probes.png", plots.probe_trajectories(run))):
        fig.savefig(Path(args.run_dir) / fname, dpi=150, bbox_inches="tight")
        print(f"wrote {args.run_dir}/{fname}")
