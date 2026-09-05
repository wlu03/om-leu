"""Build the combined comparison table: OM-LEU and the ML / LLM baselines from
``<dataset>/results/<run_tag>/aggregate/results_summary.csv`` plus the
literature choice models from ``methods/results/<dataset>/<method>/seed_*.json``.

    venv/bin/python methods/make_table.py            # all datasets
    venv/bin/python methods/make_table.py --datasets lpmc

Writes methods/results/comparison_table.md and comparison_table.csv, plus
per-dataset paired ΔNLL against OM-LEU (per-event, pooled over seeds) for
the literature models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from methods.common.data import run_tag  # noqa: E402
from methods.run_methods import LABELS, METHODS  # noqa: E402

GROUP_OF = {
    "OM-LEU": "OM-LEU (ours)",
    "ST-MLP": "ML baseline (repo)", "LASSO-MNL": "ML baseline (repo)", "Popularity": "ML baseline (repo)",
    "Bayesian-ARD": "ML baseline (repo)", "DUET": "ML baseline (repo)", "GradientBoosting": "ML baseline (repo)",
    "MLP": "ML baseline (repo)", "Delphos": "ML baseline (repo)", "RandomForest": "ML baseline (repo)",
}


def _repo_rows(dataset: str) -> pd.DataFrame:
    path = REPO_ROOT / dataset / "results" / run_tag(dataset) / "aggregate" / "results_summary.csv"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        name = r["method"]
        group = GROUP_OF.get(name, "LLM ranker (repo)" if "Llama" in name or "Claude" in name else "ML baseline (repo)")
        label = name.replace("-Llama-3.3-70B-Instruct-FP8-dynamic", " Llama-3.3-70B ranker")
        rows.append(dict(dataset=dataset, group=group, method=label, n_seeds=int(r["n_seeds"]),
                         top1=r["top1_mean"], top1_sd=r["top1_sd"], top3=r["top3_mean"], mrr=r["mrr_mean"],
                         nll=r["test_nll_mean"], nll_sd=r["test_nll_sd"], brier=r["brier_mean"],
                         ece=r.get("ece_mean", np.nan), n_params=r.get("n_params", np.nan),
                         paired_dnll=r.get("paired_dNLL_mean", np.nan),
                         paired_ci_lo=r.get("paired_dNLL_ci_lo", np.nan), paired_ci_hi=r.get("paired_dNLL_ci_hi", np.nan),
                         wilcoxon_p=r.get("wilcoxon_p", np.nan), mcnemar_p=r.get("mcnemar_p", np.nan)))
    return pd.DataFrame(rows)


def _omleu_per_event(dataset: str) -> dict:
    """Per-seed OM-LEU per-event NLL and top-1 flags (test split, record order)."""
    out = {}
    base = REPO_ROOT / dataset / "results" / run_tag(dataset)
    for sd in sorted(base.glob("seed_*")):
        pe = json.load(open(sd / "omleu" / "test_per_event.json"))["per_event"]
        out[int(sd.name.split("_")[1])] = (np.array([p["nll"] for p in pe]), np.array([p["top1_correct"] for p in pe], dtype=int))
    return out


def _paired(dnll: np.ndarray, dtop: np.ndarray, seed: int = 0, B: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(dnll)
    boots = np.array([dnll[rng.integers(0, n, n)].mean() for _ in range(B)])
    from scipy.stats import wilcoxon
    try:
        wp = float(wilcoxon(dnll).pvalue)
    except ValueError:
        wp = np.nan
    # McNemar on top-1 disagreements (exact binomial)
    from scipy.stats import binomtest
    b = int(((dtop == 1)).sum()); c = int(((dtop == -1)).sum())
    mp = float(binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else np.nan
    return dict(paired_dnll=float(dnll.mean()), paired_ci_lo=float(np.percentile(boots, 2.5)),
                paired_ci_hi=float(np.percentile(boots, 97.5)), wilcoxon_p=wp, mcnemar_p=mp)


OMLEU_VARIANTS = {
    "numres_los_z": "OM-LEU + numeric residual (LOS + person shifters, pre-fit)",
    "numres_los": "OM-LEU + numeric residual (LOS only, pre-fit)",
    "numres_los_z_gate": "OM-LEU + numeric residual, gated semantic branch",
}


def _variant_rows(dataset: str) -> pd.DataFrame:
    """OM-LEU retrained variants under <dataset>/results/<run_tag>_<variant>/seed_*/."""
    om = _omleu_per_event(dataset)
    rows = []
    base = REPO_ROOT / dataset / "results"
    for variant, label in OMLEU_VARIANTS.items():
        d = base / f"{run_tag(dataset)}_{variant}"
        seeds = sorted(d.glob("seed_*/metrics_test.json"))
        if not seeds:
            continue
        recs, ro, dn, dt = [], [], [], []
        for f in seeds:
            seed = int(f.parent.name.split("_")[1])
            m = json.load(open(f))
            recs.append(m)
            r = f.parent / "metrics_test_residual_only.json"
            if r.exists():
                ro.append(json.load(open(r)))
            pe = f.parent / "test_per_event.json"
            if pe.exists() and seed in om:
                pev = json.load(open(pe))["per_event"]
                onll, otop = om[seed]
                if len(pev) == len(onll):
                    dn.append(np.array([p["nll"] for p in pev]) - onll)
                    dt.append(np.array([int(p["top1_correct"]) for p in pev]) - otop)
        g = lambda rs, k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
        paired = _paired(np.concatenate(dn), np.concatenate(dt)) if dn else {}
        rows.append(dict(dataset=dataset, group="OM-LEU (ours)", method=label, n_seeds=len(recs),
                         top1=g(recs, "top1")[0], top1_sd=g(recs, "top1")[1], top3=g(recs, "top3")[0],
                         mrr=g(recs, "mrr_val")[0], nll=g(recs, "nll_val")[0], nll_sd=g(recs, "nll_val")[1],
                         brier=g(recs, "brier_val")[0], ece=g(recs, "ece_val")[0], **paired))
        if ro and variant == "numres_los_z":
            rows.append(dict(dataset=dataset, group="OM-LEU (ours)", method="  residual only (stage-1 MNL inside OM-LEU)",
                             n_seeds=len(ro), top1=g(ro, "top1")[0], top1_sd=g(ro, "top1")[1], top3=g(ro, "top3")[0],
                             mrr=g(ro, "mrr_val")[0], nll=g(ro, "nll_val")[0], nll_sd=g(ro, "nll_val")[1],
                             brier=g(ro, "brier_val")[0], ece=g(ro, "ece_val")[0]))
    return pd.DataFrame(rows)


def _method_rows(dataset: str) -> pd.DataFrame:
    om = _omleu_per_event(dataset)
    rows = []
    for method in METHODS:
        files = sorted((REPO_ROOT / "methods" / "results" / dataset / method).glob("seed_*.json"))
        recs = [json.load(open(f)) for f in files]
        recs = [r for r in recs if "top1" in r]
        if not recs:
            skipped = [json.load(open(f)).get("skipped") for f in files]
            rows.append(dict(dataset=dataset, group="Literature choice model (methods/)", method=LABELS[method],
                             n_seeds=0, note=(skipped[0] if skipped and skipped[0] else "not run")))
            continue
        agg = lambda k: (np.mean([r[k] for r in recs]), np.std([r[k] for r in recs]))
        dn, dt = [], []
        for r in recs:
            if r["seed"] in om:
                onll, otop = om[r["seed"]]
                if len(onll) == len(r["per_event_nll"]):
                    dn.append(np.array(r["per_event_nll"]) - onll)
                    dt.append(np.array(r["per_event_top1"]) - otop)
        paired = _paired(np.concatenate(dn), np.concatenate(dt)) if dn else {}
        rows.append(dict(dataset=dataset, group="Literature choice model (methods/)", method=LABELS[method],
                         n_seeds=len(recs), top1=agg("top1")[0], top1_sd=agg("top1")[1], top3=agg("top3")[0],
                         mrr=agg("mrr")[0], nll=agg("nll")[0], nll_sd=agg("nll")[1], brier=agg("brier")[0],
                         ece=agg("ece")[0], n_params=np.mean([r["n_params"] for r in recs]), **paired))
    return pd.DataFrame(rows)


def _fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def _md(df: pd.DataFrame, dataset: str) -> str:
    lines = [f"### {dataset}", "",
             "| Group | Method | Top-1 | Top-3 | MRR | NLL | Brier | ECE | ΔNLL vs OM-LEU [95% CI] | Wilcoxon p | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"OM-LEU (ours)": 0, "Literature choice model (methods/)": 1, "ML baseline (repo)": 2, "LLM ranker (repo)": 3}
    d = df[df.dataset == dataset].copy()
    d["_o"] = d.group.map(order)
    d = d.sort_values(["_o", "nll"], na_position="last")
    for _, r in d.iterrows():
        if r.get("n_seeds", 0) == 0:
            lines.append(f"| {r.group} | {r.method} | n/a | | | | | | {r.get('note', '')} | | |")
            continue
        top1 = f"{_fmt(r.top1, pct=True)} ± {100 * r.top1_sd:.1f}" if not np.isnan(r.get("top1_sd", np.nan)) else _fmt(r.top1, pct=True)
        nll = f"{_fmt(r.nll)} ± {r.nll_sd:.3f}" if not np.isnan(r.get("nll_sd", np.nan)) else _fmt(r.nll)
        d_ = "" if np.isnan(r.get("paired_dnll", np.nan)) else f"{r.paired_dnll:+.3f} [{r.paired_ci_lo:+.3f}, {r.paired_ci_hi:+.3f}]"
        wp = "" if np.isnan(r.get("wilcoxon_p", np.nan)) else ("<1e-6" if r.wilcoxon_p < 1e-6 else f"{r.wilcoxon_p:.3g}")
        mp = "" if np.isnan(r.get("mcnemar_p", np.nan)) else ("<1e-6" if r.mcnemar_p < 1e-6 else f"{r.mcnemar_p:.3g}")
        lines.append(f"| {r.group} | {r.method} | {top1} | {_fmt(r.top3, pct=True)} | {_fmt(r.mrr)} | {nll} | {_fmt(r.brier)} | {_fmt(r.ece)} | {d_} | {wp} | {mp} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["swissmetro", "optima", "lpmc"])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "methods" / "results")
    args = ap.parse_args()
    frames = []
    for ds in args.datasets:
        frames.append(_repo_rows(ds))
        frames.append(_variant_rows(ds))
        frames.append(_method_rows(ds))
    df = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "comparison_table.csv", index=False)
    md = ["# Combined comparison: OM-LEU vs literature choice models vs repo ML / LLM baselines", "",
          "Same per-seed train/val/test splits for every row (3 seeds; mean ± sd over seeds). ΔNLL is the paired",
          "per-event difference (method − OM-LEU) pooled over seeds with a 2000-draw bootstrap CI; Wilcoxon on",
          "per-event NLL; McNemar (exact) on top-1 hits. Repo rows come from the OM-LEU pipeline's own",
          "significance report. Literature models see numeric level of service + person covariates; repo ML",
          "baselines see only price/popularity columns (see docs/optima_lpmc.md).", ""]
    for ds in args.datasets:
        md.append(_md(df, ds))
    (args.out / "comparison_table.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
