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
from .calibrate import apply_calibration, apply_conditional_gate, fit_conditional_gate, fit_mixture
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
    train_fraction: float = 1.0         # E6 learning curves: nested share of training clusters
    calibrator: str = "global"          # "global" scalar pi, or "gate" (E7 conditional gate)
    gate_l2: float = 1.0
    group: str = "core"
    notes: str = ""

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


def subsample_clusters(rows: np.ndarray, clusters: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Nested subset of the training clusters: the 10% set is inside the 25% set, and so on.

    Whole respondents are kept or dropped; events are never subsampled inside a respondent,
    and the development and test partitions are untouched.
    """
    if fraction >= 1.0:
        return rows
    uniq = np.array(sorted(set(clusters[rows].tolist())))
    order = np.random.default_rng(seed).permutation(len(uniq))
    keep = set(uniq[order[:max(1, int(round(fraction * len(uniq))))]].tolist())
    return rows[np.array([c in keep for c in clusters[rows]])]


def gate_context(b: Bundle, rows: np.ndarray) -> np.ndarray:
    """Small, predeclared context for the conditional gate.

    Admissible history length, an evidence/missingness indicator (how many of the recorded
    attributes of the chosen-set alternatives are absent), and the size of the choice set.
    No label, no channel correctness, no test-set quantity.
    """
    hist = b.Xhist[rows].sum(dim=(1, 2)).numpy()
    missing = (b.Xnum[rows] == 0).float().mean(dim=(1, 2)).numpy()
    size = np.full(len(rows), float(b.J))
    C = np.stack([np.log1p(hist), missing, size], 1)
    mu, sd = C.mean(0, keepdims=True), C.std(0, keepdims=True) + 1e-6
    return (C - mu) / sd


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
                numeric_cache: Optional[Dict] = None, skip_existing: bool = True) -> Dict:
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
    if skip_existing and (out_dir / "result.json").exists():
        import json as _json
        done = _json.loads((out_dir / "result.json").read_text())
        done["reused_existing_artifact"] = True
        return done
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
        fit_rows = subsample_clusters(fit_rows, clusters, variant.train_fraction, master_seed * 101)
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
    gate_cal = None
    if variant.calibrator == "gate":
        if Q_oof is None:
            raise RuntimeError("the conditional gate needs a semantic channel")
        gate_cal = fit_conditional_gate(U_oof, Q_oof, y[rows_oof], gate_context(b, rows_oof), l2=variant.gate_l2)

    # refit on the whole development partition, then predict the locked test partition once
    dev_fit_rows = subsample_clusters(dev_rows, clusters, variant.train_fraction, master_seed * 101)
    f_rows, v_rows = inner_split(dev_fit_rows, clusters, master_seed * 31 + 999)
    nk = stable_hash({"d": dataset, "s": master_seed, "p": part.manifest_hash, "k": "final",
                      "pe": person_effects, "bo": variant.use_boost})
    if nk not in numeric_cache:
        numeric_cache[nk] = fit_numeric(b, fit_rows=f_rows, val_rows=v_rows, predict_rows=test_rows,
                                        seed=master_seed, view_tag=f"{protocol}{part.manifest_hash[:8]}final",
                                        person_effects=person_effects, use_boost=variant.use_boost)
    U_test = numeric_cache[nk].logits
    Q_test = None
    behavioural = None
    if variant.reader is not None:
        reader = READERS[variant.reader](**variant.reader_kw); reader.n_members = members
        bv_sem = restandardise_covariates(fold_view(bs, f_rows, v_rows, test_rows))
        reader.fit(bv_sem, master_seed * 17 + 999)
        Q_test = reader.logprobs(bv_sem, test_rows)
        if hasattr(reader, "behavioural_report"):
            # held-out paraphrase families and edit magnitudes; a behavioural check, not a
            # counterfactual choice observation
            behavioural = reader.behavioural_report(bv_sem, test_rows)
    if gate_cal is not None:
        lp_final = apply_conditional_gate(U_test, Q_test, gate_context(b, test_rows), gate_cal)
    else:
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
        "label_budget": {"train_fraction": variant.train_fraction,
                         "training_events_used": int(len(dev_fit_rows)),
                         "training_clusters_used": int(len(set(clusters[dev_fit_rows].tolist()))),
                         "calibration_events": int(len(rows_oof)),
                         "note": "calibration and early-stopping labels are counted separately from fitting labels"},
        "partition": {"hash": part.manifest_hash, "notes": part.notes, "n_dev": int(len(dev_rows)),
                      "n_test": int(len(test_rows)), "n_folds": len(part.dev_folds),
                      "n_test_clusters": int(len(set(clusters[test_rows].tolist())))},
        "calibration": cal, "calibration_numeric_only": cal_numeric_only, "calibration_gate": gate_cal,
        "metrics_final": m_final, "metrics_numeric_only": m_numeric, "metrics_semantic_only": m_sem,
        "behavioural": behavioural, "reader_info": getattr(reader, "info", None) if variant.reader else None,
        "smoke": smoke, "members": members, "seconds": time.time() - t0,
        "environment": art.environment_fingerprint(Path(__file__).resolve().parents[1]),
    }
    art.write_atomic(out_dir / "result.json", payload)
    art.write_atomic(out_dir / "predictions.json", preds)
    art.write_atomic(out_dir / "oof_development_predictions.json", {
        "note": "held-out development predictions used to fit pi and the temperature; these are "
                "calibration inputs and are never reported as an evaluation of the fitted mixture",
        "rows": [int(r) for r in rows_oof], "y": [int(v) for v in y[rows_oof]],
        "cluster": [str(clusters[r]) for r in rows_oof],
        "numeric_logits": U_oof.tolist(),
        "semantic_logprobs": None if Q_oof is None else Q_oof.tolist()})
    art.record_status(artifact_root, {"status": "completed", "dataset": dataset, "protocol": protocol,
                                      "variant": variant.name, "master_seed": master_seed,
                                      "artifact": str(out_dir), "nll": m_final["nll"], "smoke": smoke})
    return payload
