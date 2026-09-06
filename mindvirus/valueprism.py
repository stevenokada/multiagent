"""Build a probe battery from the ValuePrism dataset (allenai/ValuePrism, gated)."""
from __future__ import annotations

import random
import hashlib
import json

from mindvirus.payloads import Payload
from mindvirus.probes import Battery, ProbeItem

DATASET = "allenai/ValuePrism"


def load_rows() -> list[dict]:
    """Network path: requires HF auth + accepted dataset terms. Not unit-tested."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, "values", split="train")
    cols = set(ds.column_names)
    # Be liberal about exact column names across dataset versions.
    sit = "situation" if "situation" in cols else "Situation"
    val = "value" if "value" in cols else "Value"
    vrd = "vrd" if "vrd" in cols else ("VRD" if "VRD" in cols else None)
    valence = "valence" if "valence" in cols else "Valence"
    return [
        {"situation": r[sit], "value": r[val],
         "vrd": (r[vrd] if vrd else "Value"), "valence": r[valence]}
        for r in ds
    ]


def build_battery(payload: Payload, n_on: int = 4, n_control: int = 4,
                  seed: int = 0, rows: list[dict] | None = None, *,
                  task: str = "acceptability") -> Battery:
    if task not in ("acceptability", "relation"):
        raise ValueError(f"unknown battery task {task!r}")
    if rows is None:
        rows = load_rows()
    if task == "relation":
        return _build_relation_battery(payload, rows, n_on, n_control, seed)
    axis = payload.target_axis.lower()
    value_rows = [r for r in rows if str(r.get("vrd", "Value")).lower() == "value"]

    tainted = {r["situation"] for r in value_rows if r["value"].lower() == axis}
    on_pool: dict[str, int] = {}
    for r in value_rows:
        if r["value"].lower() == axis and r["situation"] not in on_pool:
            direction = 1 if str(r["valence"]).lower().startswith("support") else -1
            on_pool[r["situation"]] = direction
    ctrl_pool = sorted({r["situation"] for r in value_rows} - tainted)

    if len(on_pool) < n_on or len(ctrl_pool) < n_control:
        raise ValueError(
            f"pool too small: {len(on_pool)} on-target / {len(ctrl_pool)} control "
            f"situations for axis {payload.target_axis!r}"
        )

    rng = random.Random(seed)
    on_pick = rng.sample(sorted(on_pool), n_on)
    ctrl_pick = rng.sample(ctrl_pool, n_control)

    items = [ProbeItem(f"vp-on-{i}", s, True, on_pool[s], "valueprism")
             for i, s in enumerate(on_pick)]
    items += [ProbeItem(f"vp-ctrl-{i}", s, False, -1, "valueprism")
              for i, s in enumerate(ctrl_pick)]
    return Battery(
        payload_id=payload.id, source="valueprism", items=items,
        meta={"n_on": n_on, "n_control": n_control, "seed": seed, "dataset": DATASET},
    )


def _build_relation_battery(payload: Payload, rows: list[dict], n_on: int,
                            n_control: int, seed: int) -> Battery:
    if any(n < 0 or n % 2 for n in (n_on, n_control)) or n_on + n_control == 0:
        raise ValueError("balanced relation groups require nonnegative even counts")
    axis = payload.target_axis.casefold()
    labels_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if str(row.get("vrd", "Value")).strip().casefold() != "value":
            continue
        situation, value = str(row["situation"]).strip(), str(row["value"]).strip()
        if situation and value:
            labels_by_pair.setdefault((situation, value), set()).add(
                str(row["valence"]).strip().casefold())
    # Exclude any control situation that names the target value anywhere.
    tainted = {s for s, v in labels_by_pair if v.casefold() == axis}
    candidates = [
        (s, v, next(iter(labels)).title())
        for (s, v), labels in sorted(labels_by_pair.items())
        if len(labels) == 1 and labels <= {"supports", "opposes"}
    ]
    rng = random.Random(seed)
    items = []
    for on_target, count in ((True, n_on), (False, n_control)):
        for label in ("Supports", "Opposes"):
            pool = [(s, v, y) for s, v, y in candidates if y == label and
                    ((v.casefold() == axis) if on_target else (s not in tainted))]
            if len(pool) < count // 2:
                raise ValueError(f"balanced relation pool too small: {len(pool)} {label} "
                                 f"rows in {'target' if on_target else 'control'} group")
            for situation, value, reference in rng.sample(pool, count // 2):
                digest = hashlib.sha256(json.dumps([situation, value, reference],
                                                   ensure_ascii=False).encode()).hexdigest()
                items.append(ProbeItem(f"vp-rel-{digest[:16]}", situation, on_target, 0,
                                       "valueprism", "relation", value, reference))
    return Battery(payload.id, "valueprism", items, {
        "task": "relation", "n_on": n_on, "n_control": n_control, "seed": seed,
        "dataset": DATASET, "reference_status": "dataset_annotation",
        "sampling_unit": "situation_consideration_pair", "balanced_by": "reference_label",
        "excluded": "non-value rows, nonbinary and conflicting pairs",
        "probe_training_overlap": "unverified",
        "answer_mappings": ["standard", "reversed"],
    })
