"""
src/detection_eval/leaderboard.py
---------------------------------
The shared detection leaderboard. Every route in
``docs/research_program.md`` writes its result here, and §5 of that document
is filled in from it.

WHY A SHARED STORE RATHER THAN A TABLE IN EACH DOC
==================================================
Research program §5: *"no number enters by hand."* Routes run in parallel, in
separate sessions, and the comparison between them is the whole deliverable.
A hand-copied table drifts: one route reports PR-AUC on a 70/30 split, another
on the full dataset, a third reports the mean of three splits, and the
"comparison" compares nothing. So the only way a number reaches
``results/detection_leaderboard.md`` is :func:`record`, which takes the output
of :func:`src.detection_eval.harness.evaluate_detector` and cannot be given a
number by itself.

STORE FORMAT
============
``results/detection_leaderboard.json``::

    {"version": 1, "entries": [ {entry}, ... ]}

One entry per ``(method, dataset)``. A second :func:`record` for the same pair
**replaces** the first by default: re-running a route must update its row, not
append a duplicate that makes the leaderboard ambiguous about which number is
current. Pass ``replace=False`` to keep both (they are then distinguished only
by ``recorded_at``, so use it only deliberately).

Every metric is stored as an interval dict ``{mean, lo, hi, std, n}`` — the
shape :class:`src.experiments.aggregate.Interval` produces — so a single-run
number and a 5-split bootstrap live in the same field and the renderer can
always print the CI it has. ``n = 0`` means *no observation*: it renders as
``—`` and must never be read as 0.0.

Worked example
==============
    >>> from src.detection_eval.harness import RuleDetectorAdapter, evaluate_detector
    >>> from src.detection_eval.leaderboard import record, render_markdown
    >>> out = evaluate_detector(RuleDetectorAdapter(), "adult")      # doctest: +SKIP
    >>> record("rule_detector", "adult", out, notes="Phase 3 baseline")  # doctest: +SKIP
    >>> render_markdown()                                            # doctest: +SKIP
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.detection_eval.metrics import DetectionMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MD_PATH",
    "DEFAULT_STORE_PATH",
    "load_store",
    "record",
    "render_markdown",
]

DEFAULT_STORE_PATH = Path("results") / "detection_leaderboard.json"
DEFAULT_MD_PATH = Path("results") / "detection_leaderboard.md"
STORE_VERSION = 1

#: Marker for an undefined / unobserved quantity. Matches
#: ``src.experiments.aggregate.NOT_FITTED_MARKER`` in meaning: never print 0.0
#: for "we do not have this number" (CLAUDE.md §5.4).
MISSING = "—"

# ═══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ═══════════════════════════════════════════════════════════════════════════════


def _as_interval(value: Any) -> Dict[str, Any]:
    """
    Coerce a metric to the ``{mean, lo, hi, std, n}`` interval shape.

    ``None``                → ``n = 0`` placeholder (undefined, prints ``—``)
    ``float``               → degenerate interval, ``n = 1`` (a single run has
                              no interval; pretending otherwise would invent
                              precision)
    interval-shaped ``dict`` → passed through, with ``n`` defaulted to 1
    """
    if value is None:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "std": 0.0, "n": 0}
    if isinstance(value, dict):
        return {
            "mean": float(value.get("mean", 0.0)),
            "lo": float(value.get("lo", value.get("mean", 0.0))),
            "hi": float(value.get("hi", value.get("mean", 0.0))),
            "std": float(value.get("std", 0.0)),
            "n": int(value.get("n", 1)),
        }
    v = float(value)
    return {"mean": v, "lo": v, "hi": v, "std": 0.0, "n": 1}


def _normalise(metrics: Any) -> Dict[str, Any]:
    """
    Accept any of the three shapes a route might hand us and return a flat
    entry body.

    Accepted
    --------
    - the dict returned by :func:`src.detection_eval.harness.evaluate_detector`
      (has ``"intervals"`` and ``"full"``) — the normal path, CIs included;
    - a :class:`~src.detection_eval.metrics.DetectionMetrics` — a single run,
      no CI;
    - a plain dict of metric names to floats, with optional ``n`` / ``n_pos``
      / ``prevalence`` / ``k`` — the escape hatch for a method evaluated
      outside the harness, which must say so in ``notes``.
    """
    if isinstance(metrics, DetectionMetrics):
        metrics = metrics.to_dict()

    if not isinstance(metrics, dict):
        raise TypeError(
            "metrics must be an evaluate_detector() dict, a DetectionMetrics, "
            f"or a dict of metric values; got {type(metrics).__name__}."
        )

    if "intervals" in metrics:
        intervals = metrics.get("intervals") or {}
        full = metrics.get("full") or {}
        stats = metrics.get("label_stats") or {}
        body: Dict[str, Any] = {
            "n": int(full.get("n", stats.get("n_cells", 0))),
            "n_pos": int(full.get("n_pos", stats.get("n_corrupted", 0))),
            "prevalence": float(full.get("prevalence", stats.get("prevalence", 0.0))),
            "k": int(full.get("k", 0)),
            "n_splits": int(metrics.get("n_splits", 0)),
            "seed": int(metrics.get("seed", 0)),
            "split_fraction": float(metrics.get("split_fraction", 0.0)),
            "full": {k: full.get(k) for k in DetectionMetrics.SCORE_FIELDS},
            "score_structure": metrics.get("score_structure"),
        }
        for field in DetectionMetrics.SCORE_FIELDS:
            body[field] = _as_interval(intervals.get(field))
        return body

    body = {
        "n": int(metrics.get("n", 0)),
        "n_pos": int(metrics.get("n_pos", 0)),
        "prevalence": float(metrics.get("prevalence", 0.0)),
        "k": int(metrics.get("k", 0)),
        "n_splits": int(metrics.get("n_splits", 1)),
        "seed": int(metrics.get("seed", 0)),
        "split_fraction": float(metrics.get("split_fraction", 0.0)),
        "full": {k: metrics.get(k) for k in DetectionMetrics.SCORE_FIELDS},
        "score_structure": metrics.get("score_structure"),
    }
    for field in DetectionMetrics.SCORE_FIELDS:
        body[field] = _as_interval(metrics.get(field))
    return body


# ═══════════════════════════════════════════════════════════════════════════════
# Store I/O
# ═══════════════════════════════════════════════════════════════════════════════


def load_store(store_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Read the JSON store, returning an empty one if the file does not exist.

    A corrupt or unexpected-version file raises rather than being silently
    reset: the store is the only record of what every route measured.
    """
    path = Path(store_path or DEFAULT_STORE_PATH)
    if not path.exists():
        return {"version": STORE_VERSION, "entries": []}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError(f"Malformed leaderboard store: {path}")
    if int(data.get("version", STORE_VERSION)) != STORE_VERSION:
        raise ValueError(
            f"Leaderboard store version {data.get('version')} != {STORE_VERSION} "
            f"({path}). Migrate it deliberately rather than overwriting."
        )
    return data


def record(
    name: str,
    dataset: str,
    metrics: Any,
    notes: str = "",
    store_path: Optional[Path | str] = None,
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Add (or update) one ``(method, dataset)`` row in the store.

    Parameters
    ----------
    name : str
        Method label, e.g. ``"rule_detector"``, ``"conditional_gbm"``.
    dataset : str
        ``"adult"`` / ``"breast_cancer"`` / ``"smartfactory"``.
    metrics : dict | DetectionMetrics
        Normally the dict from
        :func:`src.detection_eval.harness.evaluate_detector`. See
        :func:`_normalise` for the other accepted shapes.
    notes : str
        What the method is and, crucially, what it does **not** do
        (research program §2: every route states its limitation). Kept on the
        row so the leaderboard cannot be read without it.
    store_path : path, optional
        Default ``results/detection_leaderboard.json``.
    replace : bool
        True (default): replace an existing row for the same
        ``(name, dataset)``.

    Returns
    -------
    dict
        The entry written.
    """
    if not str(name).strip():
        raise ValueError("name must be a non-empty string.")
    if not str(dataset).strip():
        raise ValueError("dataset must be a non-empty string.")

    path = Path(store_path or DEFAULT_STORE_PATH)
    store = load_store(path)

    entry = {
        "method": str(name),
        "dataset": str(dataset),
        "notes": str(notes),
        "recorded_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
        **_normalise(metrics),
    }

    entries: List[Dict[str, Any]] = list(store["entries"])
    if replace:
        entries = [
            e
            for e in entries
            if not (e.get("method") == entry["method"] and e.get("dataset") == entry["dataset"])
        ]
    entries.append(entry)
    store["entries"] = entries
    store["version"] = STORE_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, sort_keys=False)
        fh.write("\n")

    logger.info(
        "Leaderboard: recorded %s on %s (PR-AUC %s).",
        entry["method"],
        entry["dataset"],
        _fmt_interval(entry.get("pr_auc"), with_ci=False),
    )
    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt_interval(iv: Optional[Dict[str, Any]], with_ci: bool = True) -> str:
    """Render an interval dict. ``n = 0`` → ``—``; ``n = 1`` → bare mean."""
    if not iv or int(iv.get("n", 0)) == 0:
        return MISSING
    mean = float(iv["mean"])
    if not with_ci or int(iv["n"]) < 2:
        return f"{mean:.4f}"
    return f"{mean:.4f} [{float(iv['lo']):.4f}, {float(iv['hi']):.4f}]"


def _sort_key(entry: Dict[str, Any]) -> tuple:
    """
    PR-AUC descending, undefined last, method name as the tie-break.

    Undefined sorts last rather than as 0.0 so a method that could not be
    scored never appears to have lost to one scoring 0.0001.
    """
    iv = entry.get("pr_auc") or {}
    has = int(iv.get("n", 0)) > 0
    return (0 if has else 1, -float(iv.get("mean", 0.0)) if has else 0.0, str(entry.get("method", "")))


def render_markdown(
    store_path: Optional[Path | str] = None,
    out_path: Optional[Path | str] = None,
) -> str:
    """
    Write ``results/detection_leaderboard.md`` and return its text.

    One table per dataset, rows sorted by PR-AUC descending. Each table prints
    the dataset's prevalence in its heading, because PR-AUC is meaningless
    without it: a random scorer scores ≈ prevalence.

    Returns
    -------
    str
        The markdown written (also returned so a caller can print it).
    """
    store = load_store(store_path)
    entries: List[Dict[str, Any]] = list(store.get("entries", []))
    out = Path(out_path or DEFAULT_MD_PATH)

    lines: List[str] = [
        "# Detection leaderboard",
        "",
        "Generated by `src/detection_eval/leaderboard.py` "
        "(`render_markdown()`). **Do not edit by hand** — "
        "`docs/research_program.md` §5: no number enters by hand.",
        "",
        "Primary metric is **PR-AUC** (average precision). Read it against the",
        "prevalence in each heading: a random scorer scores ≈ prevalence, so",
        "0.08 is ~32x random on adult and ~1.07x random on breast_cancer.",
        "`—` means *undefined / not measured*, never 0.0.",
        "",
        "`best-F1` is an **oracle** threshold chosen with knowledge of the labels;",
        "it is an upper bound, comparable across methods but not an operating point.",
        "`P@k` is precision in the top *k* cells with *k = n_pos* (R-precision),",
        "computed as the expectation under random tie-breaking.",
        "",
    ]

    if not entries:
        lines += ["_No entries recorded yet._", ""]
    else:
        for dataset in sorted({str(e.get("dataset", "")) for e in entries}):
            rows = sorted(
                [e for e in entries if str(e.get("dataset", "")) == dataset], key=_sort_key
            )
            prevalence = float(rows[0].get("prevalence", 0.0))
            n_cells = int(rows[0].get("n", 0))
            n_pos = int(rows[0].get("n_pos", 0))
            lines += [
                f"## {dataset}",
                "",
                f"{n_cells:,} cells, {n_pos:,} corrupted, "
                f"prevalence {prevalence:.5f} ({100 * prevalence:.3f} %)",
                "",
                "| method | PR-AUC [95% CI] | ROC-AUC | best-F1 | "
                "P@k (k=n_pos) | prevalence | splits | notes |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for e in rows:
                n_splits = int(e.get("n_splits", 0))
                lines.append(
                    "| {method} | {pr} | {roc} | {f1} | {pk} | {prev} | "
                    "{ns} | {notes} |".format(
                        method=str(e.get("method", "")),
                        pr=_fmt_interval(e.get("pr_auc")),
                        roc=_fmt_interval(e.get("roc_auc"), with_ci=False),
                        f1=_fmt_interval(e.get("best_f1"), with_ci=False),
                        pk=_fmt_interval(e.get("precision_at_k"), with_ci=False),
                        prev=f"{float(e.get('prevalence', 0.0)):.5f}",
                        ns=(str(n_splits) if n_splits else MISSING),
                        notes=str(e.get("notes", "")).replace("|", "\\|").replace("\n", " "),
                    )
                )
            lines.append("")

    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("Leaderboard rendered to %s (%d entries).", out, len(entries))
    return text
