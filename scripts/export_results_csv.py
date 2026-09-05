"""Export a Modal OM-LEU run to two CSVs (per-seed and side-by-side summary).

    python scripts/export_results_csv.py <dataset>/results/<run_tag> [--out DIR]

Writes ``results_by_seed.csv`` (one row per method x seed, with every
metric the leaderboard carries) and ``results_summary.csv`` (one row per
method: mean/sd over seeds, the gap to the method of interest, and the
paired significance tests from ``aggregate/significance``).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

METRICS = ("top1", "top3", "top5", "mrr", "test_nll", "brier", "ece", "n_params")


def _seed_dirs(run_dir: Path) -> list[tuple[int, Path, Path]]:
    """``(seed, seed_dir, omleu_dir)`` for the Modal layout
    (``<run>/seed_<s>/{omleu,baselines}``) or a single local run
    (``<run>/{records.pkl,...}`` + ``<run>/baselines``)."""
    out: list[tuple[int, Path, Path]] = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_")[1])
        except ValueError:
            continue
        out.append((seed, seed_dir, seed_dir / "omleu"))
    if not out and (run_dir / "baselines").is_dir():
        seed = 0
        for lb in run_dir.glob("baselines/baselines_leaderboard_*seed*.json"):
            tail = lb.stem.rsplit("seed", 1)[-1]
            if tail.isdigit():
                seed = int(tail)
        out.append((seed, run_dir, run_dir))
    return out


def load_run(run_dir: Path, focus: str = "OM-LEU") -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for seed, seed_dir, omleu_dir in _seed_dirs(run_dir):
        lb = next(iter(seed_dir.glob("baselines/baselines_leaderboard_*.json")), None)
        if lb is None:
            continue
        omleu_metrics: dict = {}
        mt = omleu_dir / "metrics_test.json"
        if mt.exists():
            omleu_metrics = json.loads(mt.read_text())
        for r in json.loads(lb.read_text()):
            if r.get("status") != "ok":
                continue
            row = {"seed": seed, "method": r["name"], "n_test": len(r.get("per_event_nll") or [])}
            for k in METRICS:
                row[k] = r.get(k)
            if r["name"] == focus and omleu_metrics:
                row["ece"] = omleu_metrics.get("ece_val")
                row["n_params"] = omleu_metrics.get("n_params")
            rows.append(row)
    sig_path = run_dir / "aggregate" / "significance" / f"pairwise_vs_{focus}.json"
    sig = json.loads(sig_path.read_text()) if sig_path.exists() else {}
    return rows, sig


def summarize(rows: list[dict], sig: dict, focus: str = "OM-LEU") -> list[dict]:
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    focus_rows = {r["seed"]: r for r in by_method.get(focus, [])}
    out: list[dict] = []
    pairs = sig.get("pairs", {}) if isinstance(sig, dict) else {}
    for method, rs in by_method.items():
        row: dict = {"method": method, "n_seeds": len(rs), "seeds": "|".join(str(r["seed"]) for r in rs)}
        for k in ("top1", "top3", "top5", "mrr", "test_nll", "brier", "ece"):
            vals = [float(r[k]) for r in rs if r.get(k) is not None]
            row[f"{k}_mean"] = statistics.mean(vals) if vals else None
            row[f"{k}_sd"] = statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        # Same-seed gaps to the focus method (positive = focus better for
        # NLL/Brier, negative = focus better for Top-1/MRR).
        for k, sign in (("top1", -1), ("mrr", -1), ("test_nll", +1), ("brier", +1)):
            deltas = [
                float(r[k]) - float(focus_rows[r["seed"]][k])
                for r in rs if r["seed"] in focus_rows and r.get(k) is not None
            ]
            row[f"delta_{k}_vs_{focus}_mean"] = statistics.mean(deltas) if deltas else None
        p = pairs.get(method) or {}
        row["paired_dNLL_mean"] = p.get("delta_mean")
        row["paired_dNLL_ci_lo"] = p.get("delta_ci_lo")
        row["paired_dNLL_ci_hi"] = p.get("delta_ci_hi")
        row["p_delta_gt_zero"] = p.get("p_delta_gt_zero")
        row["wilcoxon_p"] = p.get("wilcoxon_p_value")
        row["mcnemar_p"] = p.get("mcnemar_p_value")
        row["n_params"] = rs[0].get("n_params")
        out.append(row)
    out.sort(key=lambda r: (r["test_nll_mean"] is None, r["test_nll_mean"] or 0.0))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)) for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--focus", default="OM-LEU")
    a = p.parse_args(argv)
    rows, sig = load_run(a.run_dir, a.focus)
    if not rows:
        raise SystemExit(f"no leaderboard rows under {a.run_dir}")
    summary = summarize(rows, sig, a.focus)
    out = a.out or (a.run_dir / "aggregate")
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["seed"], r["method"]))
    write_csv(out / "results_by_seed.csv", rows)
    write_csv(out / "results_summary.csv", summary)
    print(f"wrote {out / 'results_by_seed.csv'} ({len(rows)} rows)")
    print(f"wrote {out / 'results_summary.csv'} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
