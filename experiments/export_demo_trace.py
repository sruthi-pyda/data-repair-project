"""
experiments/export_demo_trace.py
--------------------------------
Export a **real** pipeline trace for the judge-facing demo site.

Planned in `docs/demo_site_plan.md` §2. The demo page replays this file, so
every number a judge sees comes from the actual pipeline running on the
actual data. Nothing on that page is invented, and nothing is computed from
thin air in JavaScript — except the decision arithmetic itself, which is
genuinely live and is the point of the demo.

WHAT IS EXPORTED, PER CELL
==========================
The full chain for a handful of cells, so the page can show one stage per
step with the reasoning visible:

    detection   which of the six checks fired, its score, the reason string
    repair      the top-k candidate values with their probabilities
    uncertainty raw confidence, self-consistency, entropy
    calibration raw confidence -> calibrated p, and which calibrator won
    q           P(this flagged cell is genuinely corrupted), per check type
    costs       the three expected costs under the shipped config
    decision    the argmin, and the two boundary values that decide it

LEAKAGE
=======
`CLAUDE.md` §6 forbids showing a clean value beside a dirty cell in the
decision demo. So the per-cell payload carries **no ground truth at all** —
not the clean value, not the correctness label. Ground-truth-derived figures
live in a separate `evaluation` block that the page renders only in its
clearly-labelled results section.

Run:  python experiments/export_demo_trace.py
Writes results/demo_trace.json. Deterministic: seed 42.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402
from src.risk.cost_model import CostConfig, RiskCalculator  # noqa: E402

enable_unicode_stdout()

SEED = 42
OUT_JSON = ROOT / "results" / "demo_trace.json"
COSTS_YAML = ROOT / "configs" / "costs.yaml"

#: Datasets the demo walks through. breast_cancer is first because it is the
#: only one that exercises the whole pipeline across corruption types.
DATASETS = ("breast_cancer", "smartfactory", "adult")

#: Cells exported per dataset. Enough to show variety, few enough that the
#: page stays breathable (`docs/demo_site_plan.md` §5).
N_CELLS = 6

#: Rows of the dirty table shipped for the "here is the data" act.
N_PREVIEW_ROWS = 8


def _num(x: Any) -> Any:
    """JSON-safe: NaN/inf become None rather than invalid JSON or 0.0."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if not math.isfinite(f) else f
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None or isinstance(x, (str, int)):
        return x
    return str(x)


def _sanitise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return _num(obj)


def pick_cells(bundle, n: int) -> List[int]:
    """
    Choose decided cells spanning as many detection methods as possible, so
    the demo shows variety rather than six copies of one check.
    """
    idx = bundle.decision_indices()
    by_method: Dict[str, List[int]] = {}
    for i in idx:
        by_method.setdefault(bundle.detections[i].detection_method, []).append(i)
    out: List[int] = []
    # round-robin across methods
    while len(out) < min(n, len(idx)):
        progressed = False
        for m in sorted(by_method):
            if by_method[m] and len(out) < n:
                out.append(by_method[m].pop(0))
                progressed = True
        if not progressed:
            break
    return out


def row_context(bundle, row_id: int, column: str) -> List[Dict[str, Any]]:
    """
    The rest of the row, which is what makes the conditional reasoning
    legible. The target cell is marked; no clean values appear.
    """
    row = bundle.dirty_df.iloc[row_id]
    out = []
    for c in bundle.dirty_df.columns:
        out.append({
            "column": str(c),
            "value": _num(row[c]),
            "is_target": bool(c == column),
        })
    return out


def export_dataset(dataset: str, n_cells: int) -> Optional[Dict[str, Any]]:
    t0 = time.time()
    bundle = run_pipeline(dataset, n_records=0, seed=SEED)
    costs = CostConfig.from_yaml(COSTS_YAML, dataset)
    calc = RiskCalculator(costs)

    picks = pick_cells(bundle, n_cells)
    if not picks:
        print(f"  {dataset:<14} no decided cells; skipped")
        return None

    cells: List[Dict[str, Any]] = []
    for i in picks:
        det = bundle.detections[i]
        rep = bundle.repair_results[i]
        unc = bundle.unc_records[i]
        p = float(bundle.calibrated_probs[i])
        q = float(bundle.corruption_probs[i]) if bundle.corruption_probs else 1.0
        ec = calc.compute_expected_costs(p, corruption_probability=q)

        cells.append({
            "row_id": int(det.row_id),
            "column": str(det.column),
            "dirty_value": _num(det.dirty_value),
            "detection": {
                "method": str(det.detection_method),
                "score": _num(det.detection_score),
                "error_type": str(det.error_type),
                "reason": str(det.reason),
            },
            "candidates": [
                {"value": _num(c.value), "probability": _num(c.probability)}
                for c in rep.candidates
            ],
            "repair_method": str(rep.repair_method),
            "uncertainty": {
                "raw_confidence": _num(unc.raw_confidence),
                "self_consistency": _num(unc.self_consistency),
                "entropy": _num(unc.entropy),
                "normalized_entropy": _num(unc.normalized_entropy),
                "n_candidates": _num(unc.n_candidates),
            },
            "calibration": {
                "raw": _num(unc.raw_confidence),
                "p": _num(p),
                "shift": _num(p - float(unc.raw_confidence)),
            },
            "q": _num(q),
            "expected_costs": {
                "auto": _num(ec.accept),
                "human": _num(ec.review),
                "abstain": _num(ec.leave),
            },
            "action": str(getattr(ec.best_action, "name", ec.best_action)),
            "context": row_context(bundle, int(det.row_id), str(det.column)),
        })

    # Column-level profile facts, for the "learn what normal looks like" step.
    columns = []
    for name in bundle.dirty_df.columns:
        prof = bundle.profile.columns.get(name) if hasattr(
            bundle.profile, "columns") else None
        kind = getattr(prof, "kind", None)
        columns.append({
            "name": str(name),
            "kind": str(getattr(kind, "value", kind)) if kind else "unknown",
            "n_distinct": int(bundle.dirty_df[name].nunique(dropna=True)),
        })

    preview = []
    for r in range(min(N_PREVIEW_ROWS, len(bundle.dirty_df))):
        preview.append({str(c): _num(bundle.dirty_df.iloc[r][c])
                        for c in bundle.dirty_df.columns})

    flagged_keys = {(int(bundle.unc_records[i].row_id),
                     str(bundle.unc_records[i].column))
                    for i in bundle.decision_indices()}

    out = {
        "dataset": dataset,
        "n_rows": int(len(bundle.dirty_df)),
        "n_cols": int(len(bundle.dirty_df.columns)),
        "n_cells": int(bundle.dirty_df.size),
        "n_flagged": int(bundle.n_detected_total),
        "n_decided": int(len(bundle.decision_indices())),
        "calibration_method": str(bundle.calibration_method),
        "calibration_note": (str(bundle.calibration_note)
                             if bundle.calibration_note else None),
        "detection_note": (str(bundle.detection_note)
                           if bundle.detection_note else None),
        "costs": {
            "wrong": _num(costs.c_wrong_repair),
            "human": _num(costs.c_human_review),
            "abstain": _num(costs.c_leave_error),
        },
        "columns": columns,
        "preview_rows": preview,
        "preview_flagged": sorted(
            [[r, c] for (r, c) in flagged_keys if r < N_PREVIEW_ROWS],
            key=lambda t: (t[0], t[1])),
        "cells": cells,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  {dataset:<14} {out['n_decided']:>6,} decided, "
          f"{len(cells)} cells exported, calibrator "
          f"{out['calibration_method']}   [{out['seconds']:.0f}s]")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export the demo trace")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--cells", type=int, default=N_CELLS)
    args = ap.parse_args(argv)

    print("Exporting a REAL pipeline trace for the demo site")
    print(f"seed {SEED}\n")
    datasets: List[Dict[str, Any]] = []
    for ds in args.datasets:
        try:
            r = export_dataset(ds, args.cells)
        except Exception as exc:
            print(f"  {ds:<14} FAILED {type(exc).__name__}: {exc}")
            continue
        if r:
            datasets.append(r)

    payload = {
        "generated_by": "experiments/export_demo_trace.py",
        "seed": SEED,
        "note": (
            "Every per-cell value here is real output from src/pipeline/"
            "runner.py. The demo page replays it and computes ONLY the "
            "decision arithmetic live. No per-cell ground truth is included: "
            "CLAUDE.md section 6 forbids showing a clean value beside a "
            "dirty cell in the decision demo."
        ),
        "datasets": datasets,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(_sanitise(payload), indent=1), encoding="utf-8")
    size = OUT_JSON.stat().st_size
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}  ({size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
