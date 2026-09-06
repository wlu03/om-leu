"""Ablation table for OM-LEU 2: every ``abl_*`` / ``combo_*`` variant against ``combo_full``
(paired per-event ΔNLL pooled over seeds, 2000-draw bootstrap CI).

    venv/bin/python experiments/ablation_table.py            # prints + writes experiments/results/ablations.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness.data import results_dir  # noqa: E402

DATASETS = ["swissmetro", "optima", "lpmc"]
SEEDS = [7, 11, 13]
REF = "combo_full"
ROWS = [  # (variant, label)
    ("combo_full", "full model (reference)"),
    ("combo_struct_cal", "A11 no semantic mixture (temperature only)"),
    ("combo_struct", "A12 no stage 3 at all"),
    ("abl_members1", "A10 one semantic member instead of 5"),
    ("abl_shuffled_sentences", "B1 shuffled sentences (other event, same alternative)"),
    ("abl_random_embeddings", "B2 random embeddings"),
    ("abl_altid_sentences", "B3 alternative-identity-only 'sentences'"),
    ("abl_no_person", "A1 no person random effects"),
    ("abl_no_taste", "A2 no TasteNet tastes"),
    ("abl_no_intercepts", "A3 no functional intercepts"),
    ("abl_linear_struct", "A4 linear MNL structural part"),
    ("abl_restarts1", "A5 one restart instead of 5"),
    ("abl_no_boost", "A6 no boosted residual"),
    ("abl_boost_nomono", "A7 booster without monotone constraints"),
    ("abl_boost_additive", "A8 additive booster (one feature per tree)"),
    ("abl_tau1_insample", "A9 in-sample offset, tau = 1"),
    ("combo_struct_insample", "A9b in-sample offset, tau selected"),
    ("combo_full_ncat", "A13 + concept-head residual"),
    ("abl_no_hist", "C1 no history features"),
    ("abl_cold_start", "C4 cold-start protocol (test persons removed from train)"),
    ("abl_cold_start_struct", "C4b cold-start, structural only"),
]


def load(variant: str, ds: str):
    out = {}
    for s in SEEDS:
        f = results_dir() / variant / f"{ds}_seed{s}.json"
        if f.exists():
            r = json.load(open(f))
            if "nll" in r:
                out[s] = r
    return out


def paired(a: dict, b: dict):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = np.concatenate([np.array(a[s]["per_event_nll"]) - np.array(b[s]["per_event_nll"]) for s in seeds])
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main() -> None:
    lines = ["# OM-LEU 2 ablations (test, mean over seeds; ΔNLL paired per event vs the full model, 95 % bootstrap CI; * = CI excludes 0)", ""]
    for ds in DATASETS:
        ref = load(REF, ds)
        lines += [f"## {ds}", "", "| ablation | seeds | Top-1 | NLL | Brier | ECE | ΔNLL vs full | π | τ |", "|---|---|---|---|---|---|---|---|---|"]
        for variant, label in ROWS:
            r = load(variant, ds)
            if not r:
                continue
            m = lambda k: np.mean([x[k] for x in r.values()])
            pi = np.mean([x.get("extra", {}).get("pi", np.nan) for x in r.values()])
            tau = np.mean([x.get("extra", {}).get("tau", np.nan) for x in r.values()])
            p = paired(r, ref) if variant != REF else None
            dtxt = "" if p is None else f"{p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]{'*' if (p[1] > 0 or p[2] < 0) else ''}"
            lines.append(f"| {label} | {len(r)} | {m('top1')*100:.1f}% | {m('nll'):.4f} | {m('brier'):.3f} | {m('ece'):.3f} | {dtxt} | "
                         f"{'' if np.isnan(pi) else f'{pi:.2f}'} | {'' if np.isnan(tau) else f'{tau:.2f}'} |")
        lines.append("")
    text = "\n".join(lines)
    (results_dir() / "ablations.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
