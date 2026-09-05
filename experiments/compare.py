"""Compare all variant results against the references (original OM-LEU, OM-LEU +
numeric residual, and the literature models under methods/results).

    venv/bin/python experiments/compare.py            # prints + writes experiments/results/summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness.data import MAIN_REPO, load_bundle, results_dir  # noqa: E402

DATASETS = ["swissmetro", "optima", "lpmc"]
SEEDS = [7, 11, 13]


def paired(d):
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    root = results_dir()
    refs = {}
    for ds in DATASETS:
        for s in SEEDS:
            try:
                b = load_bundle(ds, s)
            except FileNotFoundError:
                continue
            refs[(ds, s)] = b.refs
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = mdir.name
        for ds in DATASETS:
            recs = []
            for s in SEEDS:
                f = mdir / f"{ds}_seed{s}.json"
                if f.exists():
                    r = json.load(open(f))
                    if "nll" in r:
                        recs.append(r)
            if not recs:
                continue
            top1 = np.mean([r["top1"] for r in recs]); nll = np.mean([r["nll"] for r in recs])
            brier = np.mean([r["brier"] for r in recs]); ece = np.mean([r["ece"] for r in recs])
            d_num, d_om = [], []
            for r in recs:
                ref = refs.get((ds, r["seed"]), {})
                pe = np.array(r["per_event_nll"])
                if "numres_test_nll" in ref and len(ref["numres_test_nll"]) == len(pe):
                    d_num.append(pe - ref["numres_test_nll"])
                if "omleu_test_nll" in ref and len(ref["omleu_test_nll"]) == len(pe):
                    d_om.append(pe - ref["omleu_test_nll"])
            pn = paired(np.concatenate(d_num)) if d_num else None
            po = paired(np.concatenate(d_om)) if d_om else None
            rows.append((name, ds, len(recs), top1, nll, brier, ece, pn, po))
    # literature references
    lit = {}
    for ds in DATASETS:
        for m in ("mnl", "gbdt_full_features", "asu_dnn"):
            fs = list((MAIN_REPO / "methods" / "results" / ds / m).glob("seed_*.json"))
            rs = [json.load(open(f)) for f in fs]
            rs = [r for r in rs if "nll" in r]
            if rs:
                lit[(ds, m)] = (np.mean([r["top1"] for r in rs]), np.mean([r["nll"] for r in rs]))
    lines = ["# Variant comparison (test, mean over seeds; ΔNLL paired per event, 95 % bootstrap CI)", ""]
    for ds in DATASETS:
        lines += [f"## {ds}", "", "| model | seeds | Top-1 | NLL | Brier | ECE | ΔNLL vs OM-LEU+numres | ΔNLL vs OM-LEU |", "|---|---|---|---|---|---|---|---|"]
        ref_line = " · ".join(f"{m}: {v[0]*100:.1f}% / {v[1]:.3f}" for (d, m), v in lit.items() if d == ds)
        for name, d, n, top1, nll, brier, ece, pn, po in sorted([r for r in rows if r[1] == ds], key=lambda r: r[4]):
            f = lambda p: "" if p is None else f"{p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]"
            lines.append(f"| {name} | {n} | {top1*100:.1f}% | {nll:.4f} | {brier:.3f} | {ece:.3f} | {f(pn)} | {f(po)} |")
        lines += ["", f"references (methods/): {ref_line}", ""]
    text = "\n".join(lines)
    (root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
