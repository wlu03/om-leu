"""Identified outcome channel: each variant against its own shuffled control.

The contrast that matters is variant vs its OWN shuffled control, not vs a common
baseline: the control shuffles the same (erased, residual-trained) embeddings, so the
difference isolates event-specific content in the generated text and cannot be explained
by anything the erasure left in place.

    venv/bin/python experiments/identified_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness.data import load_bundle, results_dir  # noqa: E402

DATASETS = ["swissmetro", "optima", "lpmc", "amazon"]
SEEDS = [7, 11, 13]
PAIRS = [("id_plain", "id_plain_shuffled", "no orthogonalisation"),
         ("id_erase", "id_erase_shuffled", "erasure"),
         ("id_resid", "id_resid_shuffled", "residual offset"),
         ("id_erase_resid", "id_erase_resid_shuffled", "erasure + residual offset")]
REFERENCE = "abl_cold_start_struct"          # person split, no language channel


def load(variant, dataset, seed):
    f = results_dir() / variant / f"{dataset}_seed{seed}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return None if "error" in d else d


def clusters_for(dataset, seed, n_events):
    """Cluster label per test event, for the bootstrap; falls back to one cluster per event."""
    try:
        b = load_bundle(dataset, seed)
        from experiments.models.omleu2 import cold_start_view
        bv = cold_start_view(b, seed)
        te = bv.idx("test")
        if len(te) == n_events:
            return b.person[te].numpy()
    except Exception:
        pass
    return np.arange(n_events)


def paired(a, b, cl, draws=2000, seed=0):
    d = np.asarray(b) - np.asarray(a)          # control minus variant: positive = variant better
    uniq = sorted(set(cl.tolist()))
    pos = {c: i for i, c in enumerate(uniq)}
    idx = [[] for _ in uniq]
    for i, c in enumerate(cl):
        idx[pos[c]].append(i)
    sums = np.array([d[i].sum() for i in idx])
    cnts = np.array([len(i) for i in idx], dtype=float)
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, len(uniq), (draws, len(uniq)))
    boot = sums[dr].sum(1) / np.maximum(cnts[dr].sum(1), 1e-12)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def main():
    print("Identified outcome channel, person-level split, weight stacked out of fold.")
    print("Each variant against its own shuffled control; positive favours the real sentences.\n")
    print(f"{'dataset':11s} {'variant':26s} {'NLL':>8s} {'control':>8s} {'gain [95% CI]':>28s}  seeds")
    for ds in DATASETS:
        ref = [load(REFERENCE, ds, s) for s in SEEDS]
        ref = [r["nll"] for r in ref if r]
        if ref:
            print(f"{ds:11s} {'no language channel':26s} {np.mean(ref):8.4f} {'--':>8s} {'--':>28s}  {len(ref)}")
        for var, ctrl, label in PAIRS:
            ev_v, ev_c, cls, n_ok, nll_v, nll_c = [], [], [], 0, [], []
            for s in SEEDS:
                dv, dc = load(var, ds, s), load(ctrl, ds, s)
                if not dv or not dc:
                    continue
                pv, pc = np.array(dv["per_event_nll"]), np.array(dc["per_event_nll"])
                if len(pv) != len(pc):
                    continue
                ev_v.append(pv); ev_c.append(pc)
                cls.append(np.array([f"s{s}:{c}" for c in clusters_for(ds, s, len(pv))]))
                nll_v.append(pv.mean()); nll_c.append(pc.mean()); n_ok += 1
            if not n_ok:
                continue
            m, lo, hi = paired(np.concatenate(ev_v), np.concatenate(ev_c), np.concatenate(cls))
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"{ds:11s} {label:26s} {np.mean(nll_v):8.4f} {np.mean(nll_c):8.4f} "
                  f"{m:+.4f}{star} [{lo:+.4f}, {hi:+.4f}]  {n_ok}")
        print()


if __name__ == "__main__":
    main()
