"""Sentence sources for the semantic channel.

``llm``       the cached frozen-LLM consequences already exported in the bundle
``template``  deterministic factual sentences built from exactly the generation allowlist
``shuffled``  a donor event's sentences for the same dataset, alternative and axis
``identity``  alternative-identity text only (no attributes, no person)
``random``    seeded random vectors with the documented normalisation of the encoder

Template and control tensors are cached under the artifact root, content-addressed by
the fields that determine them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from experiments.harness.data import Bundle

from .contracts import AXES, stable_hash

ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"


def _encode(texts: Sequence[str]) -> np.ndarray:
    from src.outcomes.encode import SentenceTransformersEncoder
    enc = SentenceTransformersEncoder(model_id=ENCODER_ID, max_length=64, pooling="mean")
    out = []
    B = 256
    for s in range(0, len(texts), B):
        out.append(np.asarray(enc.encode(list(texts[s:s + B]))))
    V = np.concatenate(out, 0).astype(np.float32)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)   # L2, as the pipeline does
    return V


def template_sentences(b: Bundle, i: int, j: int) -> Dict[str, str]:
    """Deterministic factual sentences: recorded quantities only, one per axis, with
    'not recorded' where the dataset has no value.  No preference judgement, no
    invented precision."""
    names = b.meta["alt_feature_names"]
    val = {n: float(b.Xnum[i, j, f]) for f, n in enumerate(names)}
    alt = b.meta["alts"][int(b.alt_idx[i, j])]
    hist = {n: float(b.Xhist[i, j, h]) for h, n in enumerate(b.meta["hist_names"])}

    def g(*keys, default=None):
        for k in keys:
            if k in val:
                return val[k]
        return default
    t = g("time_h"); c = g("cost_chf", "cost_gbp", "cost_100chf")
    wait = g("waiting_h", "headway_h"); walk = g("walking_h", "access_h")
    tr = g("transfers", "interchanges"); dist = g("distance_km")
    cur = {"cost_chf": "CHF", "cost_gbp": "GBP", "cost_100chf": "100 CHF"}
    ccur = next((v for k, v in cur.items() if k in val), "")
    s = {
        "financial": (f"Recorded out-of-pocket cost for {alt}: {c:.2f} {ccur}." if c is not None
                      else f"Out-of-pocket cost for {alt}: not recorded."),
        "time": (f"Recorded in-vehicle or travel time for {alt}: {t * 60:.0f} minutes." if t is not None
                 else f"Travel time for {alt}: not recorded."),
        "comfort": (f"Recorded distance for {alt}: {dist:.1f} km." if dist not in (None, 0.0)
                    else f"No comfort attribute is recorded for {alt}."),
        "convenience": (f"Recorded access or walking time for {alt}: {walk * 60:.0f} minutes; "
                        f"transfers: {tr:.0f}." if walk is not None and tr is not None
                        else (f"Recorded access or walking time for {alt}: {walk * 60:.0f} minutes."
                              if walk is not None else f"No access attribute is recorded for {alt}.")),
        "reliability": (f"Recorded waiting or headway time for {alt}: {wait * 60:.0f} minutes." if wait is not None
                        else f"No waiting or headway attribute is recorded for {alt}."),
    }
    if hist.get("is_repeat", 0.0) > 0:
        s["convenience"] += f" This respondent chose {alt} on {hist.get('log1p_purchase_count', 0.0):.2f} log-earlier occasions."
    return s


def identity_sentences(b: Bundle, i: int, j: int) -> Dict[str, str]:
    alt = b.meta["alts"][int(b.alt_idx[i, j])]
    return {a: f"This alternative is {alt}." for a in AXES}


def build_source(b: Bundle, kind: str, *, seed: int = 0, cache_dir: Optional[Path] = None,
                 admissible_rows: Optional[np.ndarray] = None,
                 builder: Optional[Callable] = None, tag: str = "") -> torch.Tensor:
    """Return an (N, J, K, d) embedding tensor for the requested sentence source."""
    key = stable_hash({"ds": b.dataset, "seed": b.seed, "kind": kind, "s": seed, "tag": tag,
                       "enc": ENCODER_ID, "n": b.N, "K": b.K})
    path = None if cache_dir is None else Path(cache_dir) / f"sent_{b.dataset}_s{b.seed}_{kind}{tag}_{key[:12]}.npy"
    if path is not None and path.exists():
        return torch.from_numpy(np.load(path)).float()
    if kind in ("llm", "none"):
        # "none" belongs to a channel that reads no text at all (the numeric auxiliary);
        # the tensor is passed through unchanged and ignored by that reader
        return b.E.float()
    if kind in ("template", "identity", "custom"):
        fn = builder or (template_sentences if kind == "template" else identity_sentences)
        texts, order = [], []
        for i in range(b.N):
            for j in range(b.J):
                d = fn(b, i, j)
                for k, a in enumerate(AXES):
                    texts.append(d[a]); order.append((i, j, k))
        uniq = sorted(set(texts))
        pos = {t: n for n, t in enumerate(uniq)}
        V = _encode(uniq)
        E = np.zeros((b.N, b.J, b.K, V.shape[1]), dtype=np.float32)
        for (i, j, k), t in zip(order, texts):
            E[i, j, k] = V[pos[t]]
    elif kind == "shuffled":
        # donor events drawn from the admissible fitting partition, matched on dataset,
        # canonical alternative and axis, never the recipient person, never by the label
        rng = np.random.default_rng(seed)
        pool = np.arange(b.N) if admissible_rows is None else np.asarray(admissible_rows)
        by_alt = {}
        for i in pool:
            for j in range(b.J):
                by_alt.setdefault(int(b.alt_idx[i, j]), []).append((int(i), j))
        E = np.zeros(tuple(b.E.shape), dtype=np.float32)
        for i in range(b.N):
            for j in range(b.J):
                cand = by_alt.get(int(b.alt_idx[i, j]), [])
                cand = [(ii, jj) for ii, jj in cand if b.person[ii] != b.person[i]] or cand
                ii, jj = cand[rng.integers(0, len(cand))]
                E[i, j] = b.E[ii, jj].numpy()
    elif kind == "random":
        g = torch.Generator().manual_seed(seed)
        V = torch.randn(b.N, b.J, b.K, b.d, generator=g)
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)          # same L2 normalisation
        E = V.numpy().astype(np.float32)
    else:
        raise ValueError(kind)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy"); np.save(tmp, E); tmp.replace(path)
    return torch.from_numpy(E).float()
