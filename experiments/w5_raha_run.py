"""
experiments/w5_raha_run.py
--------------------------
Run **Raha** (BigDaMa/raha, SIGMOD Most Reproducible Paper Award) on our
datasets, to produce the external baseline this project has never had.

Until now every baseline we beat was one we wrote ourselves. That is the
largest structural gap in the work: it establishes that our method beats our
own controls, not that it is competitive with the literature.

MUST be run with the dedicated interpreter, because `raha` pins
`pandas==2.1.0` while the project runs 2.3:

    .venv-raha/Scripts/python.exe experiments/w5_raha_run.py

It must also be a real file with a ``__main__`` guard: Raha uses
``multiprocessing``, and on Windows the spawn start method re-imports the
main module, which fails outright when that module is ``<stdin>``.

PROTOCOL, and it must be stated wherever these numbers appear:
Raha consumes a **20-label budget**, simulated from `clean.csv`. That is the
standard Raha / ED2 / SAGED benchmarking practice, but it means Raha sees 20
ground-truth labels per dataset while our label-free detectors see **zero**.
Raha's reported precision/recall/F1 are its own, computed by its own
evaluator over its own cell universe.
"""

from __future__ import annotations

import json
import os
import sys
import time

DATASETS = ["breast_cancer", "hospital", "rayyan", "beers"]
LABEL_BUDGET = 20


def main() -> int:
    import raha

    root = os.path.abspath("datasets")
    out = {}
    print(f"Raha external baseline  ·  labelling budget = {LABEL_BUDGET}")
    print(f"{'dataset':<16}{'P':>9}{'R':>9}{'F1':>9}{'flagged':>10}{'secs':>8}")
    print("-" * 61)

    for name in DATASETS:
        dd = {
            "name": name,
            "path": os.path.join(root, name, "dirty.csv"),
            "clean_path": os.path.join(root, name, "clean.csv"),
        }
        t0 = time.time()
        try:
            app = raha.Detection()
            app.LABELING_BUDGET = LABEL_BUDGET
            app.VERBOSE = False
            detection = app.run(dd)
            p, r, f = raha.Dataset(dd).get_data_cleaning_evaluation(detection)[:3]
            secs = time.time() - t0
            out[name] = {"precision": p, "recall": r, "f1": f,
                         "n_flagged": len(detection), "seconds": round(secs, 1)}
            print(f"{name:<16}{p:>9.4f}{r:>9.4f}{f:>9.4f}{len(detection):>10,}{secs:>8.0f}")
        except Exception as exc:  # a documented failure beats an invented number
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:<16}  FAILED  {type(exc).__name__}: {str(exc)[:40]}")

    with open("results/w5_raha_raw.json", "w", encoding="utf-8") as fh:
        json.dump({"labeling_budget": LABEL_BUDGET, "results": out}, fh, indent=2)
    print("\n[written] results/w5_raha_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
