"""Run the ablations in ablation/<name>/ on the exported bundles.

    venv/bin/python ablation/run.py --ablations all --protocols chrono person --pi val
    venv/bin/python ablation/run.py --ablations full_model plain_nn_sentences --datasets optima --seeds 7 --pi oof

One JSON per (ablation, protocol, pi estimator, dataset, seed) in ablation/<name>/results/.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import torch

from _common import DATASETS, SEEDS, build, discover, result_path  # noqa: E402
from experiments.harness.data import load_bundle  # noqa: E402
from experiments.harness.train import evaluate  # noqa: E402


def main() -> None:
    mods = discover()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", nargs="+", default=["all"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--protocols", nargs="+", default=["chrono", "person"])
    ap.add_argument("--pi", nargs="+", default=["val"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = list(mods) if args.ablations == ["all"] else args.ablations
    torch.set_num_threads(args.threads)
    for dataset in args.datasets:
        for seed in args.seeds:
            b = load_bundle(dataset, seed)
            for protocol in args.protocols:
                for pi in args.pi:
                    for name in names:
                        out = result_path(name, protocol, pi, dataset, seed)
                        if out.exists() and not args.force:
                            print(f"[{dataset} s{seed} {protocol} pi={pi}] {name}: exists", flush=True); continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        t0 = time.time()
                        try:
                            model, run = build(mods[name], protocol, pi)(b, seed)
                            info = run(model, b, seed)
                            m = evaluate(model, getattr(model, "data_view", None) or b, "test")
                        except Exception:
                            err = traceback.format_exc(); print(f"[{dataset} s{seed} {protocol} pi={pi}] {name}: FAILED\n{err}", flush=True)
                            out.write_text(json.dumps({"ablation": name, "protocol": protocol, "pi_fit": pi, "dataset": dataset, "seed": seed, "error": err}))
                            continue
                        s3 = info.get("extra", {}).get("stage3", {})
                        rec = {"ablation": name, "protocol": protocol, "pi_fit": pi, "dataset": dataset, "seed": seed, **m, **info,
                               "sem_only": s3.get("sem_only_test"), "sem_only_per_event_nll": s3.pop("sem_only_per_event_nll", None)}
                        out.write_text(json.dumps(rec, indent=1, default=float))
                        so = rec["sem_only"]["nll"] if rec["sem_only"] else float("nan")
                        print(f"[{dataset} s{seed} {protocol} pi={pi}] {name:26s} nll={m['nll']:.4f} sem_only={so:.4f} "
                              f"pi={info['extra'].get('pi', 0):.3f} top1={m['top1']*100:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
