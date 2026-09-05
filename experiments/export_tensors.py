"""Export everything an OM-LEU variant needs as one .npz per (dataset, seed),
replayed from the cached Llama outcomes + mpnet embeddings (no LLM calls):

  E        (N, J, K, d) float16   outcome-sentence embeddings, record alt order
  Xnum     (N, J, F)   float32    numeric level of service (methods/common/data units)
  Xhist    (N, J, 2)   float32    per-alt history: is_repeat, log1p(purchase_count)
  z_d      (N, p)      float32    OM-LEU's bucketed person vector
  Z        (N, P)      float32    numeric person / trip covariates (standardised on train)
  alt_idx  (N, J)      int64      canonical alternative index of each slot
  y        (N,)        int64      chosen slot
  person   (N,)        int64      person index (panel key)
  split    (N,)        int8       0 train / 1 val / 2 test
  omleu_test_nll, omleu_test_top1  (n_test,)  original OM-LEU per-event test numbers (pairing)
  numres_test_nll, numres_test_top1 (n_test,)  OM-LEU + numeric residual (los_z) if available

Usage: venv/bin/python experiments/export_tensors.py --datasets optima lpmc swissmetro --seeds 7 11 13
Output: experiments/data/<dataset>_seed<s>.npz  (+ .json with names)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from methods.common.data import load_dataset, records_path, run_tag  # noqa: E402


class _CacheOnly:
    model_id = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"

    def generate(self, *a, **k):
        raise RuntimeError("outcomes cache miss; refusing to call an LLM")

    complete = generate


def export(dataset: str, seed: int, out_dir: Path) -> Path:
    from src.data.batching import assemble_batch
    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter
    from src.outcomes.encode import SentenceTransformersEncoder

    ds = load_dataset(dataset, seed)
    rp = records_path(dataset, seed)
    seed_dir = rp.parent.parent
    rc = json.load(open(seed_dir / "run_config.json"))
    bundle = pickle.load(open(rp, "rb"))
    enc = SentenceTransformersEncoder(model_id="sentence-transformers/all-mpnet-base-v2", max_length=64, pooling="mean")
    oc = OutcomesCache(seed_dir / "cache" / "outcomes.sqlite")
    ec = EmbeddingsCache(seed_dir / "cache" / "embeddings.sqlite")
    alt_of = {a: i for i, a in enumerate(ds.alts)}
    people = sorted({p for s in ds.splits.values() for p in s.person})
    pidx = {p: i for i, p in enumerate(people)}

    parts = {k: [] for k in ("E", "Xnum", "Xhist", "z_d", "Z", "alt_idx", "y", "person", "split")}
    for si, split in enumerate(("train", "val", "test")):
        recs = bundle[split]
        s = ds.splits[split]
        assert len(recs) == s.n
        b = assemble_batch(recs, adapter=None, llm_client=_CacheOnly(), encoder=enc, outcomes_cache=oc,
                           embeddings_cache=ec, K=int(rc["K"]), seed=seed,
                           prompt_version_cascade=(rc["prompt_version"],), diversity_filter=diversity_filter,
                           tabular_feature_names=None)
        E = b.E.numpy().astype(np.float16)
        alt_idx = np.array([[alt_of[a] for a in r["choice_asins"]] for r in recs], dtype=np.int64)
        Xnum = np.stack([s.X[i, alt_idx[i], :] for i in range(s.n)]).astype(np.float32)
        Xhist = np.array([[[float(a.get("is_repeat") or 0.0), float(np.log1p(float(a.get("purchase_count") or 0.0)))]
                           for a in r["alt_texts"]] for r in recs], dtype=np.float32)
        y = b.c_star.numpy().astype(np.int64)
        assert (alt_idx[np.arange(s.n), y] == s.y).all(), "chosen slot mismatch"
        parts["E"].append(E); parts["Xnum"].append(Xnum); parts["Xhist"].append(Xhist)
        parts["z_d"].append(b.z_d.numpy().astype(np.float32)); parts["Z"].append(s.Z.astype(np.float32))
        parts["alt_idx"].append(alt_idx); parts["y"].append(y)
        parts["person"].append(np.array([pidx[p] for p in s.person], dtype=np.int64))
        parts["split"].append(np.full(s.n, si, dtype=np.int8))
    arrays = {k: np.concatenate(v, 0) for k, v in parts.items()}
    # pairing references
    pe = json.load(open(seed_dir / "omleu" / "test_per_event.json"))["per_event"]
    arrays["omleu_test_nll"] = np.array([p["nll"] for p in pe], dtype=np.float32)
    arrays["omleu_test_top1"] = np.array([int(p["top1_correct"]) for p in pe], dtype=np.int8)
    nr = REPO_ROOT / dataset / "results" / f"{run_tag(dataset)}_numres_los_z" / f"seed_{seed}" / "test_per_event.json"
    if nr.exists():
        pe2 = json.load(open(nr))["per_event"]
        arrays["numres_test_nll"] = np.array([p["nll"] for p in pe2], dtype=np.float32)
        arrays["numres_test_top1"] = np.array([int(p["top1_correct"]) for p in pe2], dtype=np.int8)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dataset}_seed{seed}.npz"
    np.savez_compressed(out, **arrays)
    meta = {"dataset": dataset, "seed": seed, "alts": ds.alts, "alt_feature_names": ds.alt_feature_names,
            "z_names": ds.z_names, "hist_names": ["is_repeat", "log1p_purchase_count"], "nests": ds.nests,
            "K": int(rc["K"]), "prompt_version": rc["prompt_version"], "n_persons": len(people),
            "time_idx": 0, "cost_idx": 1, "run_tag": run_tag(dataset)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} N={len(arrays['y'])} J={arrays['E'].shape[1]} K={arrays['E'].shape[2]} d={arrays['E'].shape[3]}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["optima", "lpmc", "swissmetro"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "data")
    args = ap.parse_args()
    for d in args.datasets:
        for s in args.seeds:
            export(d, s, args.out)


if __name__ == "__main__":
    main()
