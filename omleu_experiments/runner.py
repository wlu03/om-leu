"""One evaluation of one variant under one protocol.

Nesting (§5 of the experiment programme):

1. the test partition is reserved and never used for any fitting or selection;
2. outer development folds give held-out predictions of the *entire* pipeline;
3. inside an outer training partition, early stopping and tau use an inner
   cluster-disjoint split, and Stage-2 offsets are cross-fitted on inner folds;
4. pi and the temperature are fitted on the pooled out-of-fold development predictions;
5. the base procedures are refitted on the whole development partition and the locked
   test partition is predicted once.

The pooled development predictions used in step 4 are reported as calibration inputs,
never as an evaluation of the fitted mixture.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from experiments.harness.data import Bundle, load_bundle

from . import artifacts as art
from .calibrate import apply_calibration, fit_mixture
from .contracts import Prediction, stable_hash
from .metrics import all_metrics, event_nll
from .numeric import fit_numeric
from .protocol import historical_chronological, ordering_support, person_disjoint, rolling_origin
from .readers import READERS
from .sentences import build_source
from .views import fold_view, inner_split, replace_embeddings, restandardise_covariates


@dataclass
class Variant:
    name: str
    reader: Optional[str] = "preserved"       # None -> numeric only
    reader_kw: Dict = field(default_factory=dict)
    sentence_source: str = "llm"
    source_kw: Dict = field(default_factory=dict)
    use_boost: bool = True
    group: str = "core"
    notes: str = ""

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


def cluster_ids(b: Bundle, persons: Sequence[str]) -> np.ndarray:
    """Household when the identifier encodes it (LPMC ``h<household>_p<person>``),
    otherwise the respondent."""
    out = []
    for p in persons:
        p = str(p)
        out.append(p.split("_p")[0] if p.startswith("h") and "_p" in p else p)
    return np.asarray(out)


def shared_records_path(dataset: str, seed: int) -> Path:
    """Raw records live in the original checkout and are shared read-only by every
    worktree; the worktree copy of the path is used when it exists."""
    from experiments.harness.data import MAIN_REPO
    from methods.common.data import records_path, run_tag
    local = records_path(dataset, seed)
    if local.exists():
        return local
    return MAIN_REPO / dataset / "results" / run_tag(dataset) / f"seed_{seed}" / "omleu" / "records.pkl"


def person_strings(b: Bundle) -> np.ndarray:
    import pickle
    recs = pickle.load(open(shared_records_path(b.dataset, b.seed), "rb"))
    allr = recs["train"] + recs["val"] + recs["test"]
    assert len(allr) == b.N, f"records {len(allr)} != bundle rows {b.N}"
    return np.asarray([str(r["customer_id"]) for r in allr]), np.asarray([str(r["order_date"]) for r in allr]), allr


ORDER_KIND = {"lpmc": "calendar", "swissmetro": "task_index", "optima": "synthetic"}


def make_partition(b: Bundle, protocol: str, master_seed: int, persons, order_values, clusters, folds: int):
    if protocol == "person_disjoint":
        return person_disjoint(persons, clusters, master_seed, n_folds=folds)
    if protocol == "temporal":
        sup = ordering_support(order_values, ORDER_KIND.get(b.dataset, "none"), persons)
        if not sup["supported"]:
            raise RuntimeError(f"temporal protocol unsupported for {b.dataset}: {sup['reason']}")
        return rolling_origin(order_values, clusters, n_folds=folds)
    if protocol == "historical_chronological":
        return historical_chronological(b.split.numpy(), clusters, n_folds=folds)
    raise ValueError(protocol)


def run_variant(dataset: str, master_seed: int, protocol: str, variant: Variant, artifact_root: Path,
                *, folds: int = 5, members: int = 5, smoke: bool = False, owner: str = "coordinator",
                numeric_cache: Optional[Dict] = None) -> Dict:
    t0 = time.time()
    b = load_bundle(dataset, master_seed if master_seed in (7, 11, 13) else 7)
    persons, order_dates, _ = person_strings(b)
    order_values = np.array([np.datetime64(d).astype("datetime64[s]").astype(float) for d in order_dates])
    clusters = cluster_ids(b, persons)
    part = make_partition(b, protocol, master_seed, persons, order_values, clusters, folds)
    person_effects = protocol != "person_disjoint"          # never fit an effect for a held-out respondent
    cfg_hash = stable_hash({"v": variant.config_hash, "p": part.manifest_hash, "f": folds, "m": members,
                            "s": master_seed, "smoke": smoke})
    out_dir = art.artifact_dir(artifact_root, dataset, protocol, variant.name, master_seed, cfg_hash)
    if not art.claim(out_dir, owner):
        raise RuntimeError(f"artifact directory owned by another worker: {out_dir}")

    # sentence source (controls draw donors only from development rows)
    if variant.reader is None:
        E = None
    else:
        E = build_source(b, variant.sentence_source, seed=master_seed, cache_dir=artifact_root / "_sentences",
                         admissible_rows=part.train_dev, **variant.source_kw)
    bs = b if E is None else replace_embeddings(b, E)

    numeric_cache = {} if numeric_cache is None else numeric_cache
    dev_rows, test_rows = part.train_dev, part.test
    y = b.y.numpy()
    oof_num, oof_sem, oof_rows = [], [], []
    for k, fold in enumerate(part.dev_folds):
        fit_rows = np.setdiff1d(dev_rows, fold)
        if protocol == "temporal":                       # forward-only: fit strictly before the block
            fit_rows = fit_rows[order_values[fit_rows] < order_values[fold].min()]
            if len(fit_rows) < 50:
                continue
        f_rows, v_rows = inner_split(fit_rows, clusters, master_seed * 31 + k)
        nk = stable_hash({"d": dataset, "s": master_seed, "p": part.manifest_hash, "k": k,
                          "pe": person_effects, "bo": variant.use_boost})
        if nk not in numeric_cache:
            numeric_cache[nk] = fit_numeric(b, fit_rows=f_rows, val_rows=v_rows, predict_rows=fold,
                                            seed=master_seed, view_tag=f"{protocol}{part.manifest_hash[:8]}f{k}",
                                            person_effects=person_effects, use_boost=variant.use_boost)
        nf = numeric_cache[nk]
        oof_num.append(nf.logits); oof_rows.append(fold)
        if variant.reader is not None:
            reader = READERS[variant.reader](**variant.reader_kw); reader.n_members = members
            bv_sem = restandardise_covariates(fold_view(bs, f_rows, v_rows, fold))
            reader.fit(bv_sem, master_seed * 17 + k)
            oof_sem.append(reader.logprobs(bv_sem, fold))
    rows_oof = np.concatenate(oof_rows)
    U_oof = np.concatenate(oof_num, 0)
    Q_oof = np.concatenate(oof_sem, 0) if oof_sem else None
    cal = fit_mixture(U_oof, Q_oof, y[rows_oof])
    cal_numeric_only = fit_mixture(U_oof, None, y[rows_oof], fit_pi=False)

    # refit on the whole development partition, then predict the locked test partition once
    f_rows, v_rows = inner_split(dev_rows, clusters, master_seed * 31 + 999)
    nk = stable_hash({"d": dataset, "s": master_seed, "p": part.manifest_hash, "k": "final",
                      "pe": person_effects, "bo": variant.use_boost})
    if nk not in numeric_cache:
        numeric_cache[nk] = fit_numeric(b, fit_rows=f_rows, val_rows=v_rows, predict_rows=test_rows,
                                        seed=master_seed, view_tag=f"{protocol}{part.manifest_hash[:8]}final",
                                        person_effects=person_effects, use_boost=variant.use_boost)
    U_test = numeric_cache[nk].logits
    Q_test = None
    if variant.reader is not None:
        reader = READERS[variant.reader](**variant.reader_kw); reader.n_members = members
        bv_sem = restandardise_covariates(fold_view(bs, f_rows, v_rows, test_rows))
        reader.fit(bv_sem, master_seed * 17 + 999)
        Q_test = reader.logprobs(bv_sem, test_rows)
    lp_final = apply_calibration(U_test, Q_test, cal)
    lp_numeric = apply_calibration(U_test, None, cal_numeric_only)
    yt = y[test_rows]
    m_final = all_metrics(lp_final, yt)
    m_numeric = all_metrics(lp_numeric, yt)
    m_sem = all_metrics(Q_test, yt) if Q_test is not None else None

    preds = [Prediction(dataset=dataset, event_id=str(int(r)), person_id=str(persons[r]), cluster_id=str(clusters[r]),
                        split="test", fold=None, master_seed=master_seed, variant=variant.name, protocol=protocol,
                        alt_ids=[b.meta["alts"][int(a)] for a in b.alt_idx[r]], avail=[True] * b.J,
                        numeric_logits=[float(x) for x in U_test[i]],
                        semantic_logprobs=None if Q_test is None else [float(x) for x in Q_test[i]],
                        final_logprobs=[float(x) for x in lp_final[i]], chosen_index=int(yt[i]),
                        calibration={"pi": cal["pi"], "a": cal["a"]},
                        lineage={"config_hash": cfg_hash, "partition": part.manifest_hash}).to_json()
             for i, r in enumerate(test_rows)]
    payload = {
        "status": "completed", "dataset": dataset, "protocol": protocol, "variant": variant.name,
        "variant_config": asdict(variant), "master_seed": master_seed, "config_hash": cfg_hash,
        "partition": {"hash": part.manifest_hash, "notes": part.notes, "n_dev": int(len(dev_rows)),
                      "n_test": int(len(test_rows)), "n_folds": len(part.dev_folds),
                      "n_test_clusters": int(len(set(clusters[test_rows].tolist())))},
        "calibration": cal, "calibration_numeric_only": cal_numeric_only,
        "metrics_final": m_final, "metrics_numeric_only": m_numeric, "metrics_semantic_only": m_sem,
        "smoke": smoke, "members": members, "seconds": time.time() - t0,
        "environment": art.environment_fingerprint(Path(__file__).resolve().parents[1]),
    }
    art.write_atomic(out_dir / "result.json", payload)
    art.write_atomic(out_dir / "predictions.json", preds)
    art.record_status(artifact_root, {"status": "completed", "dataset": dataset, "protocol": protocol,
                                      "variant": variant.name, "master_seed": master_seed,
                                      "artifact": str(out_dir), "nll": m_final["nll"], "smoke": smoke})
    return payload
