"""Run the literature choice models under ``methods/`` on the same per-seed
splits as the OM-LEU runs and write metrics to ``methods/results/``.

    venv/bin/python methods/run_methods.py --datasets swissmetro optima lpmc --seeds 7 11 13
    venv/bin/python methods/run_methods.py --datasets optima --methods mnl iclv_hybrid_choice --seeds 7

Outputs: methods/results/<dataset>/<method>/seed_<s>.json with metrics
(same functions as the OM-LEU pipeline), per-event NLL / top-1 flags for
paired tests, the fitted-model summary (coefficients, VOT, nest scales,
latent-variable loadings) and timing.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from methods.common.data import apply_person_split_from_file, load_dataset, write_person_split  # noqa: E402  (numpy/pandas only)

# LightGBM and torch each bundle their own libomp; loading both in one process
# segfaults on macOS. The boosting methods therefore run in a torch-free child
# process that writes probabilities to an .npz which the parent evaluates.
BOOSTING = {"rumboost", "gbdt_full_features"}

METHODS = [
    "mnl", "nested_logit", "mixed_logit", "iclv_hybrid_choice",
    "l_mnl", "asu_dnn", "tastenet_mnl", "rumboost", "gbdt_full_features",
]
LABELS = {
    "mnl": "MNL (LOS + socio-demographics)",
    "nested_logit": "Nested logit",
    "mixed_logit": "Mixed logit (panel, lognormal time/cost)",
    "iclv_hybrid_choice": "ICLV hybrid choice (latent attitude)",
    "l_mnl": "L-MNL (Sifringer et al. 2020)",
    "asu_dnn": "ASU-DNN (Wang et al. 2020)",
    "tastenet_mnl": "TasteNet-MNL (Han et al. 2022)",
    "rumboost": "RUMBoost (Salvadé & Hillel 2024)",
    "gbdt_full_features": "GBDT, full features (Hillel et al. 2021)",
}


def _child(dataset: str, seed: int, method: str, out: Path, protocol: str = "historical",
           with_history: str = "0") -> None:
    # This child also runs the boosting models, and LightGBM cannot share a process with torch
    # on this machine.  The person-disjoint assignment is therefore computed by the parent and
    # read from a file here, so the child never imports torch.
    ds = load_dataset(dataset, seed, with_history=with_history == "1")
    if protocol == "person":
        ds = apply_person_split_from_file(ds, dataset, seed)
    mod = importlib.import_module(f"methods.{method}.model")
    res = mod.run(ds, seed)
    np.savez(out, probs=np.asarray(res["probs_test"]), n_params=int(res["n_params"]),
             fit=json.dumps(res.get("fit", {}), default=float), extra=json.dumps(res.get("extra", {}), default=float))


def _run_one(dataset: str, seed: int, method: str, ds, out: Path) -> dict:
    if method in BOOSTING:
        tmp = out.with_suffix(".npz")
        subprocess.run([sys.executable, __file__, "--_child", dataset, str(seed), method, str(tmp),
                        os.environ.get("METHODS_PROTOCOL", "historical"),
                        os.environ.get("METHODS_HISTORY", "0")],
                       check=True, cwd=str(REPO_ROOT))
        z = np.load(tmp, allow_pickle=False)
        res = {"probs_test": z["probs"], "n_params": int(z["n_params"]),
               "fit": json.loads(str(z["fit"])), "extra": json.loads(str(z["extra"]))}
        tmp.unlink()
        return res
    mod = importlib.import_module(f"methods.{method}.model")
    return mod.run(ds, seed)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_child":
        _child(sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5]),
               sys.argv[6] if len(sys.argv) > 6 else "historical",
               sys.argv[7] if len(sys.argv) > 7 else "0")
        return
    from methods.common.metrics import evaluate  # imports torch; keep out of the child

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["swissmetro", "optima", "lpmc"])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "methods" / "results")
    ap.add_argument("--force", action="store_true", help="re-run even if the result json exists")
    ap.add_argument("--protocol", default="historical", choices=["historical", "person"],
                    help="historical: the chronological within-person split shipped with the records. "
                         "person: re-partition so every event of a respondent lies in one split, using "
                         "the same draw as the proposed model's person-disjoint protocol.")
    ap.add_argument("--with-history", action="store_true",
                    help="give the reference models the two per-alternative history columns the "
                         "proposed model's structural stage uses, so the comparison is not "
                         "confounded by unequal information")
    args = ap.parse_args()

    for dataset in args.datasets:
        for seed in args.seeds:
            ds = load_dataset(dataset, seed, with_history=args.with_history)
            if args.with_history:
                os.environ["METHODS_HISTORY"] = "1"
            if args.protocol == "person":
                write_person_split(ds, dataset, seed)
                ds = apply_person_split_from_file(ds, dataset, seed)
                os.environ["METHODS_PROTOCOL"] = "person"
            for method in args.methods:
                out = args.out / dataset / method / f"seed_{seed}.json"
                if out.exists() and not args.force:
                    print(f"[{dataset} s{seed}] {method}: exists, skip")
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    res = _run_one(dataset, seed, method, ds, out)
                except Exception:  # keep going; record the failure
                    err = traceback.format_exc()
                    print(f"[{dataset} s{seed}] {method}: FAILED\n{err}")
                    out.write_text(json.dumps({"dataset": dataset, "seed": seed, "method": method, "error": err}, indent=1))
                    continue
                record = {"dataset": dataset, "seed": seed, "method": method, "label": LABELS[method]}
                if "skipped" in res:
                    record["skipped"] = res["skipped"]
                    print(f"[{dataset} s{seed}] {method}: skipped ({res['skipped']})")
                else:
                    te = ds.splits["test"]
                    m = evaluate(np.asarray(res["probs_test"]), te.y, n_params=int(res["n_params"]), n_train=ds.splits["train"].n)
                    record.update(m)
                    record["n_test"] = te.n
                    record["fit"] = res.get("fit", {})
                    record["extra"] = res.get("extra", {})
                    print(f"[{dataset} s{seed}] {method:22s} top1={m['top1']*100:5.1f}%  nll={m['nll']:.4f}  brier={m['brier']:.4f}  "
                          f"params={res['n_params']}  ({record['fit'].get('seconds', 0):.0f}s)")
                out.write_text(json.dumps(record, indent=1, default=float))


if __name__ == "__main__":
    main()
