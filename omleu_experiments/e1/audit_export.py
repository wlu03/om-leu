"""Cached-generation audit: mechanical scan plus a label-blind human-annotation export."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.harness.data import load_bundle
from omleu_experiments.contracts import AXES
from omleu_experiments.runner import shared_records_path

from .validators import numeric_support_report, unsupported_claim_scan


def load_cached_sentences(dataset: str, seed: int, rows: List[int]) -> Dict[Tuple[int, int], List[str]]:
    """Read the cached consequences for the given bundle rows (no generation)."""
    import json as _json

    from methods.common.data import run_tag
    from experiments.harness.data import MAIN_REPO
    from src.data.batching import assemble_batch
    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter
    from src.outcomes.encode import SentenceTransformersEncoder

    class CacheOnly:
        model_id = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"

        def generate(self, *a, **k):
            raise RuntimeError("outcomes cache miss; refusing to call a generator")
        complete = generate

    rp = shared_records_path(dataset, seed)
    seed_dir = rp.parent.parent
    rc = _json.load(open(seed_dir / "run_config.json"))
    recs = pickle.load(open(rp, "rb"))
    allr = recs["train"] + recs["val"] + recs["test"]
    enc = SentenceTransformersEncoder(model_id="sentence-transformers/all-mpnet-base-v2", max_length=64, pooling="mean")
    oc = OutcomesCache(seed_dir / "cache" / "outcomes.sqlite")
    ec = EmbeddingsCache(seed_dir / "cache" / "embeddings.sqlite")
    out: Dict[Tuple[int, int], List[str]] = {}
    B = 32
    for s in range(0, len(rows), B):
        chunk = [allr[i] for i in rows[s:s + B]]
        b = assemble_batch(chunk, adapter=None, llm_client=CacheOnly(), encoder=enc, outcomes_cache=oc,
                           embeddings_cache=ec, K=int(rc["K"]), seed=seed,
                           prompt_version_cascade=(rc["prompt_version"],), diversity_filter=diversity_filter,
                           tabular_feature_names=None)
        for n, i in enumerate(rows[s:s + B]):
            for j in range(len(chunk[n]["choice_asins"])):
                out[(i, j)] = list(b.outcomes_nested[n][j])
    return out


def supplied_quantities(bundle, row: int, slot: int) -> Dict[str, float]:
    names = bundle.meta["alt_feature_names"]
    d = {n: float(bundle.Xnum[row, slot, f]) for f, n in enumerate(names)}
    for h, hn in enumerate(bundle.meta["hist_names"]):
        d[hn] = float(bundle.Xhist[row, slot, h])
    return d


def audit(dataset: str, seed: int, n_events: int, out_dir: Path, *, rng_seed: int = 0) -> Dict:
    """Stratified, label-blind export plus the mechanical scan of the same sample."""
    b = load_bundle(dataset, seed)
    rng = np.random.default_rng(rng_seed)
    strata = b.y.numpy()                       # stratify by the chosen alternative to cover the space
    rows: List[int] = []
    per = max(1, n_events // max(1, len(set(strata.tolist()))))
    for v in sorted(set(strata.tolist())):
        idx = np.where(strata == v)[0]
        rows += rng.choice(idx, size=min(per, len(idx)), replace=False).tolist()
    rows = sorted(set(rows))
    texts = load_cached_sentences(dataset, seed, rows)
    supplied = {k: supplied_quantities(b, *k) for k in texts}
    report = numeric_support_report(b, texts, supplied)
    items = []
    for (i, j), sents in texts.items():
        for k, s in enumerate(sents):
            items.append({"annotation_id": f"{dataset}-{seed}-{i}-{j}-{k}", "dataset": dataset,
                          "alternative": b.meta["alts"][int(b.alt_idx[i, j])], "axis": AXES[k] if k < len(AXES) else str(k),
                          "text": s, "supplied_fields": supplied[(i, j)],
                          "support": None, "factually_correct": None, "preference_contamination": None,
                          "ambiguous": None, "annotator": None,
                          "note": "the chosen alternative is deliberately not shown to the annotator"})
    rng.shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"audit_export_{dataset}_seed{seed}.json").write_text(json.dumps(items, indent=1))
    (out_dir / f"audit_scan_{dataset}_seed{seed}.json").write_text(json.dumps(report, indent=1, default=float))
    return report
