"""OM-LEU + numeric level-of-service residual: replay-train OM-LEU from a
saved ``records.pkl`` (cached outcomes + embeddings, no LLM calls) with the
Strategy B tabular residual extended by the *numeric* alternative attributes
and, optionally, alternative-specific person shifters — i.e. the same
information the MNL under ``methods/mnl`` receives.

The numeric features travel in a side channel ``rec["alt_numeric"]`` that is
never rendered into the LLM prompt and does not enter the outcomes-cache key,
so the cached outcome sentences are reused verbatim. Residual columns are
``num:<name>``; alternative-specific coefficients are obtained by
interaction columns ``num:los_<feature>@<alt>`` (attribute × alt indicator)
and ``num:z_<name>@<alt>`` (standardised person covariate × alt indicator,
non-reference alternatives only), so the linear residual is exactly an
alternative-specific MNL on top of the semantic utility.

    venv/bin/python scripts/retrain_omleu_numeric.py --dataset optima --seed 7 --variant los_z
    venv/bin/python scripts/retrain_omleu_numeric.py --dataset lpmc --seeds 7 11 13 --variant los

Variants: ``los`` (numeric attributes only), ``los_z`` (attributes + person
shifters), ``base`` (original 7 residual features; sanity replay).
Outputs: ``<dataset>/results/<run_tag>_numres_<variant>/seed_<s>/`` with the
retrain script's metrics_test.json, test_logits.npz, test_per_event.json.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from methods.common.data import load_dataset, run_tag, records_path  # noqa: E402

ORIGINAL_MODEL_ID = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
PROMPT_VERSION = "v7_modechoice_anchored"
K = 5
BASE_FEATURES = ["price", "log1p_price", "price_rank", "popularity_count",
                 "log1p_popularity_count", "is_repeat", "log1p_purchase_count"]


def augment(bundle: dict, ds, variant: str) -> tuple[dict, list[str]]:
    """Attach ``alt_numeric`` to every record; return (bundle, feature names)."""
    alt_idx = {a: i for i, a in enumerate(ds.alts)}
    tr = ds.splits["train"]
    # keep only (feature, alt) columns that vary in train
    los_cols = [(f, a) for f, fname in enumerate(ds.alt_feature_names) for a in ds.alts
                if tr.X[:, alt_idx[a], f].std() > 1e-8]
    z_cols = [(p, a) for p in range(ds.P) for a in ds.alts[1:]] if variant == "los_z" else []
    asc_cols = list(ds.alts[1:])  # alternative-specific constants (reference = first alt)
    names = [f"num:asc@{a}" for a in asc_cols] + \
            [f"num:los_{ds.alt_feature_names[f]}@{a}" for f, a in los_cols] + \
            [f"num:z_{ds.z_names[p]}@{a}" for p, a in z_cols]
    for split in ("train", "val", "test"):
        s = ds.splits[split]
        recs = bundle[split]
        assert len(recs) == s.n, (split, len(recs), s.n)
        for i, rec in enumerate(recs):
            per_alt = []
            for alt_name in rec["choice_asins"]:
                j = alt_idx[alt_name]
                d = {f"asc@{a}": (1.0 if a == alt_name else 0.0) for a in asc_cols}
                for f, a in los_cols:
                    d[f"los_{ds.alt_feature_names[f]}@{a}"] = float(s.X[i, j, f]) if a == alt_name else 0.0
                for p, a in z_cols:
                    d[f"z_{ds.z_names[p]}@{a}"] = float(s.Z[i, p]) if a == alt_name else 0.0
                per_alt.append(d)
            rec["alt_numeric"] = per_alt
    return bundle, names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["swissmetro", "optima", "lpmc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--variant", default="los_z", choices=["base", "los", "los_z"])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=None, help="override train.lr of the joint stage")
    ap.add_argument("--override", action="append", default=[],
                    help="config override 'dotted.key=value' (e.g. regularizers.head_variance=0), repeatable")
    ap.add_argument("--semantic-gate-init", type=float, default=None,
                    help="learnable gate on the semantic utility (0.0 = start at residual-only)")
    ap.add_argument("--residual-lr-multiplier", type=float, default=None)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--no-prefit", action="store_true", help="skip the stage-1 MNL pre-fit of the residual")
    args = ap.parse_args()

    tag = run_tag(args.dataset)
    for seed in args.seeds:
        src = records_path(args.dataset, seed)
        seed_dir = src.parent.parent  # <dataset>/results/<tag>/seed_<s>
        out_dir = REPO_ROOT / args.dataset / "results" / f"{tag}_numres_{args.variant}{args.tag_suffix}" / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = pickle.load(open(src, "rb"))
        # prompt version / K of the original run (they are part of the cache key)
        rc = {}
        rc_path = seed_dir / "run_config.json"
        if rc_path.exists():
            import json
            rc = json.load(open(rc_path))
        prompt_version = str(rc.get("prompt_version") or PROMPT_VERSION)
        k_outcomes = int(rc.get("K") or K)
        features = list(BASE_FEATURES)
        if args.variant != "base":
            ds = load_dataset(args.dataset, seed)
            bundle, extra = augment(bundle, ds, args.variant)
            features += extra
        rec_path = out_dir / "records_numeric.pkl"
        rec_path.write_bytes(pickle.dumps(bundle))
        cfg = yaml.safe_load(args.config.read_text()) or {}
        cfg.setdefault("model", {}).setdefault("tabular_residual", {})
        cfg["model"]["tabular_residual"]["enabled"] = True
        cfg["model"]["tabular_residual"]["features"] = features
        if args.lr is not None:
            cfg.setdefault("train", {})["lr"] = float(args.lr)
        for ov in args.override:
            key, val = ov.split("=", 1)
            node = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = yaml.safe_load(val)
        cfg_path = out_dir / "config_numeric.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        env = {**os.environ,
               "RETRAIN_CACHE_ONLY_MODEL_ID": ORIGINAL_MODEL_ID,
               "OUTCOMES_CACHE_PATH": str(seed_dir / "cache" / "outcomes.sqlite"),
               "EMBEDDINGS_CACHE_PATH": str(seed_dir / "cache" / "embeddings.sqlite")}
        cmd = [sys.executable, "-m", "scripts.retrain_with_records", "--records", str(rec_path),
               "--config", str(cfg_path), "--output-dir", str(out_dir), "--seed", str(seed),
               "--n-epochs", str(args.n_epochs), "--tabular-residual", "true",
               "--K", str(k_outcomes), "--prompt-version-cascade", prompt_version]
        if not args.no_prefit:
            cmd += ["--residual-prefit"]
        if args.semantic_gate_init is not None:
            cmd += ["--semantic-gate-init", str(args.semantic_gate_init)]
        if args.residual_lr_multiplier is not None:
            cmd += ["--residual-lr-multiplier", str(args.residual_lr_multiplier)]
        print(f"[{args.dataset} s{seed} {args.variant}] {len(features)} residual features -> {out_dir}", flush=True)
        log = out_dir / "retrain.log"
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        tail = "".join(open(log).readlines()[-4:])
        print(tail if rc == 0 else f"FAILED rc={rc}; see {log}\n" + "".join(open(log).readlines()[-25:]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
