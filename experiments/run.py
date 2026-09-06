"""Run registered variants on the exported bundles.

    venv/bin/python experiments/run.py --models hybrid_base --datasets optima --seeds 7
    venv/bin/python experiments/run.py --models mnl_only hybrid_base            # all datasets, 3 seeds

Writes <results_dir>/<model>/<dataset>_seed<s>.json (metrics + per-event
nll/top1 + fit info). results_dir is the MAIN checkout's experiments/results
unless OMLEU_EXPERIMENTS_RESULTS is set, so worktrees write to one place.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness.data import load_bundle, results_dir  # noqa: E402
from experiments.harness.train import evaluate  # noqa: E402
from experiments.models import REGISTRY, get  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--datasets", nargs="+", default=["optima", "lpmc", "swissmetro"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default=None, help="results sub-folder name (default: model name)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    for dataset in args.datasets:
        for seed in args.seeds:
            b = load_bundle(dataset, seed)
            for name in args.models:
                out = results_dir() / (args.tag or name) / f"{dataset}_seed{seed}.json"
                if out.exists() and not args.force:
                    print(f"[{dataset} s{seed}] {name}: exists"); continue
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    model, run = get(name)(b, seed)
                    info = run(model, b, seed)
                    m = evaluate(model, getattr(model, "data_view", None) or b, "test")   # data views re-split
                except Exception:
                    err = traceback.format_exc(); print(f"[{dataset} s{seed}] {name}: FAILED\n{err}")
                    out.write_text(json.dumps({"model": name, "dataset": dataset, "seed": seed, "error": err}))
                    continue
                rec = {"model": name, "dataset": dataset, "seed": seed, **m, **info}
                out.write_text(json.dumps(rec, indent=1, default=float))
                print(f"[{dataset} s{seed}] {name:24s} top1={m['top1']*100:5.1f}%  nll={m['nll']:.4f}  brier={m['brier']:.4f}"
                      f"  ece={m['ece']:.3f}  ({info.get('fit', {}).get('seconds', 0):.0f}s, best_ep={info.get('fit', {}).get('best_epoch')})", flush=True)


if __name__ == "__main__":
    main()
