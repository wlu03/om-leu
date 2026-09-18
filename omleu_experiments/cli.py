"""Single command interface: audit, prepare, run, evaluate, report.

    venv/bin/python -m omleu_experiments.cli audit    --datasets optima lpmc swissmetro
    venv/bin/python -m omleu_experiments.cli prepare  --datasets optima --sources template identity
    venv/bin/python -m omleu_experiments.cli run      --suite smoke --datasets optima --seeds 7
    venv/bin/python -m omleu_experiments.cli evaluate --artifact-root <path>
    venv/bin/python -m omleu_experiments.cli report   --artifact-root <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from . import artifacts as art
from .protocol import ordering_support
from .matrix import SUITES, VARIANTS   # importing the matrix registers every experiment
from .runner import ORDER_KIND, Variant, cluster_ids, person_strings, run_variant


def _artifact_root(args) -> Path:
    r = args.artifact_root or os.environ.get("OMLEU_ARTIFACT_ROOT")
    if not r:
        raise SystemExit("--artifact-root or OMLEU_ARTIFACT_ROOT is required")
    return Path(r)


def cmd_audit(args):
    from experiments.harness.data import load_bundle
    out = {}
    for ds in args.datasets:
        b = load_bundle(ds, 7)
        persons, dates, _ = person_strings(b)
        ov = np.array([np.datetime64(d).astype("datetime64[s]").astype(float) for d in dates])
        cl = cluster_ids(b, persons)
        out[ds] = {"events": b.N, "alternatives": b.J, "sentence_slots": b.K, "embedding_dim": b.d,
                   "persons": len(set(persons.tolist())), "clusters": len(set(cl.tolist())),
                   "ordering": ordering_support(ov, ORDER_KIND.get(ds, "none"), persons),
                   "covariates": b.P, "attributes": b.meta["alt_feature_names"]}
        print(json.dumps({ds: out[ds]}, indent=1, default=str))
    return out


def cmd_prepare(args):
    from experiments.harness.data import load_bundle
    from .sentences import build_source
    root = _artifact_root(args)
    for ds in args.datasets:
        for seed in args.seeds:
            b = load_bundle(ds, seed)
            for src in args.sources:
                E = build_source(b, src, seed=seed, cache_dir=root / "_sentences")
                print(f"{ds} seed{seed} {src}: {tuple(E.shape)}")


def cmd_run(args):
    root = _artifact_root(args)
    names = args.variants or SUITES[args.suite]
    cache = {}
    for ds in args.datasets:
        for seed in args.seeds:
            for proto in args.protocols:
                for name in names:
                    v = VARIANTS[name]
                    try:
                        r = run_variant(ds, seed, proto, v, root, folds=args.folds, members=args.members,
                                        smoke=args.smoke, numeric_cache=cache)
                        tag = " (cached)" if r.get("reused_existing_artifact") else ""
                        print(f"[{ds} s{seed} {proto}] {name:34s} nll={r['metrics_final']['nll']:.4f} "
                              f"numeric={r['metrics_numeric_only']['nll']:.4f} pi={r['calibration']['pi']:.3f}{tag}", flush=True)
                    except Exception as e:
                        status = "blocked_data" if isinstance(e, RuntimeError) and "unsupported" in str(e) else "failed"
                        art.record_status(root, {"status": status, "dataset": ds, "protocol": proto, "variant": name,
                                                 "master_seed": seed, "reason": str(e)[:400]})
                        print(f"[{ds} s{seed} {proto}] {name:34s} {status}: {str(e)[:160]}", flush=True)
                        if args.traceback:
                            traceback.print_exc()


def cmd_evaluate(args):
    from .evaluate import evaluate_all
    print(evaluate_all(_artifact_root(args)))


def cmd_report(args):
    from .evaluate import write_report
    from .report import write as write_contrasts
    print(write_report(_artifact_root(args), Path(args.out) if args.out else None))
    print(write_contrasts(_artifact_root(args)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omleu_experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(datasets=["optima", "lpmc", "swissmetro"], seeds=[7, 11, 13])
    for name, fn in (("audit", cmd_audit), ("prepare", cmd_prepare), ("run", cmd_run),
                     ("evaluate", cmd_evaluate), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--artifact-root", default=None)
        p.add_argument("--datasets", nargs="+", default=common["datasets"])
        p.add_argument("--seeds", nargs="+", type=int, default=common["seeds"])
        if name == "prepare":
            p.add_argument("--sources", nargs="+", default=["template", "identity"])
        if name == "run":
            p.add_argument("--suite", default="smoke", choices=sorted(SUITES))
            p.add_argument("--variants", nargs="+", default=None)
            p.add_argument("--protocols", nargs="+", default=["person_disjoint"])
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--members", type=int, default=5)
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--traceback", action="store_true")
        if name == "report":
            p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    main()
