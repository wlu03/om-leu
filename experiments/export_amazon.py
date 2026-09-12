"""Export the Amazon purchase-choice run as a bundle the experiments/ and ablation/ code can read.

Amazon differs from the mode-choice datasets in one structural way: the alternatives are
products drawn from a catalogue, not a fixed set of travel modes.  There is therefore no
alternative-specific constant and no per-alternative taste; every slot is the same kind of
object and the model is a conditional logit on the product's own attributes plus the
respondent's history with it.  That is expressed here by a single canonical alternative
(``alt_idx`` all zero), which makes the alternative-specific blocks of Stage 1 and Stage 2
collapse to one shared block.

Price takes the place of cost and carries the sign constraint.  There is no time attribute;
the slot required by the structural model is filled with a constant column, which the model's
own ``time_valid`` mask zeroes because it has no training variance.

    venv/bin/python experiments/export_amazon.py --run-dir amazon/results/fixture_qwen8b --seed 7
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


class _CacheOnly:
    """Refuses to generate: everything must already be in the outcomes cache."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, *a, **k):
        raise RuntimeError("outcomes cache miss; refusing to call a generator during export")

    complete = generate


def _f(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except Exception:
        return default


def export(run_dir: Path, seed: int, out_dir: Path, model_id: str) -> Path:
    from src.data.batching import assemble_batch
    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter
    from src.outcomes.encode import SentenceTransformersEncoder

    rec_path = next(p for p in (run_dir / "records.pkl", run_dir / f"seed_{seed}" / "omleu" / "records.pkl")
                    if p.exists())
    cache_dir = next(p for p in (run_dir / "cache", run_dir / f"seed_{seed}" / "cache",
                                 REPO_ROOT / "amazon" / "cache") if p.exists())
    rc_path = next((p for p in (run_dir / "run_config.json", run_dir / f"seed_{seed}" / "run_config.json")
                    if p.exists()), None)
    rc = json.load(open(rc_path)) if rc_path else {}
    K = int(rc.get("K", 3)); pv = rc.get("prompt_version", "v1")
    records = pickle.load(open(rec_path, "rb"))
    enc = SentenceTransformersEncoder(model_id="sentence-transformers/all-mpnet-base-v2",
                                      max_length=64, pooling="mean")
    oc = OutcomesCache(cache_dir / "outcomes.sqlite")
    ec = EmbeddingsCache(cache_dir / "embeddings.sqlite")

    people = sorted({r["customer_id"] for s in records.values() for r in records[s] if True} if False
                    else {r["customer_id"] for s in ("train", "val", "test") for r in records[s]})
    pidx = {p: i for i, p in enumerate(people)}
    parts = {k: [] for k in ("E", "Xnum", "Xhist", "z_d", "Z", "alt_idx", "y", "person", "split")}
    for si, split in enumerate(("train", "val", "test")):
        recs = records[split]
        if not recs:
            continue
        b = assemble_batch(recs, adapter=None, llm_client=_CacheOnly(model_id), encoder=enc,
                           outcomes_cache=oc, embeddings_cache=ec, K=K, seed=seed,
                           prompt_version_cascade=(pv,), diversity_filter=diversity_filter,
                           tabular_feature_names=None)
        J = len(recs[0]["choice_asins"])
        price = np.array([[_f(a.get("price")) for a in r["alt_texts"]] for r in recs], dtype=np.float32)
        pop = np.array([[np.log1p(_f(a.get("popularity_count"))) for a in r["alt_texts"]] for r in recs], dtype=np.float32)
        zero = np.zeros_like(price)
        Xnum = np.stack([price, zero, pop], -1)                       # price, (unused time slot), log popularity
        Xhist = np.array([[[_f(a.get("is_repeat")), np.log1p(_f(a.get("purchase_count")))]
                           for a in r["alt_texts"]] for r in recs], dtype=np.float32)
        parts["E"].append(b.E.numpy().astype(np.float16))
        parts["Xnum"].append(Xnum); parts["Xhist"].append(Xhist)
        parts["z_d"].append(b.z_d.numpy().astype(np.float32))
        parts["Z"].append(b.z_d.numpy().astype(np.float32))           # standardised below on train
        parts["alt_idx"].append(np.zeros((len(recs), J), dtype=np.int64))   # one canonical alternative
        parts["y"].append(b.c_star.numpy().astype(np.int64))
        parts["person"].append(np.array([pidx[r["customer_id"]] for r in recs], dtype=np.int64))
        parts["split"].append(np.full(len(recs), si, dtype=np.int8))
    out = {k: np.concatenate(v, 0) for k, v in parts.items()}
    tr = out["split"] == 0
    mu, sd = out["Z"][tr].mean(0), out["Z"][tr].std(0)
    out["Z"] = ((out["Z"] - mu) / np.where(sd > 1e-6, sd, 1.0)) * (sd > 1e-6)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"amazon_seed{seed}.npz"
    np.savez_compressed(npz, **out)
    meta = {"dataset": "amazon", "seed": seed, "alts": ["product"],
            "alt_feature_names": ["price", "unused_time_slot", "log1p_popularity"],
            "z_names": [f"z{i}" for i in range(out["Z"].shape[1])],
            "hist_names": ["is_repeat", "log1p_purchase_count"], "nests": None, "K": K,
            "prompt_version": pv, "n_persons": len(people), "time_idx": 1, "cost_idx": 0,
            "run_tag": run_dir.name,
            "note": ("one canonical alternative: products have no alternative-specific constant. "
                     "The time slot is a constant column and is masked out by the model's own "
                     "time_valid check.")}
    (out_dir / f"amazon_seed{seed}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {npz}: {out['y'].shape[0]} events, J={out['alt_idx'].shape[1]}, K={K}, "
          f"persons={len(people)}, splits={np.bincount(out['split'])}")
    return npz


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "data")
    ap.add_argument("--model-id", default="qwen3-vl:8b-instruct")
    a = ap.parse_args()
    export(a.run_dir, a.seed, a.out_dir, a.model_id)
