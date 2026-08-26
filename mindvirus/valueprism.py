"""Build a probe battery from the ValuePrism dataset (allenai/ValuePrism, gated)."""
from __future__ import annotations

import random

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
                  seed: int = 0, rows: list[dict] | None = None) -> Battery:
    if rows is None:
        rows = load_rows()
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
