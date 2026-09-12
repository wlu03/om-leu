"""Uniform-floor control: how much of a mixture's gain is information and how much is a floor?

Mixing any distribution into a sharp model bounds the per-event loss from below, because
p_i(y) >= (1 - pi) s_i(y).  A channel that predicts at chance therefore still lowers the
negative log-likelihood.  The trained controls already in ``ablation/`` (shuffled sentences,
alternative-identity text, random embeddings) do not isolate this, because each of them is
still a fitted model producing a non-uniform distribution.  This control replaces the semantic
channel with the uniform distribution at the *same* fitted pi, so the difference is exactly what
the sentences contribute beyond the floor.

    venv/bin/python ablation/uniform_floor_control.py lpmc optima swissmetro amazon
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402
from experiments.harness.data import load_bundle  # noqa: E402
from experiments.models.omleu2 import _log_mean_exp  # noqa: E402


def run(dataset: str, seed: int = 7, protocol: str = "person", pi_fit: str = "val") -> dict:
    mods = _common.discover()
    b = load_bundle(dataset, seed)
    extra = {"boost_off": True} if dataset == "amazon" else None
    model, run_fn = _common.build(mods["full_model"], protocol, pi_fit, extra)(b, seed)
    run_fn(model, b, seed)
    m0, run0 = _common.build(mods["no_sentences"], protocol, pi_fit, extra)(b, seed)
    run0(m0, b, seed)
    bb = getattr(model, "data_view", None) or b
    te = bb.idx("test"); y = bb.y[te].numpy(); n = np.arange(len(y))
    with torch.no_grad():
        pi = float(model.pi)
        s = torch.softmax(float(model.log_a.exp()) * model.U[te], -1).double().numpy()
        q = _log_mean_exp(torch.stack([m(model.sem_view or bb, te) for m in model.members], 1), 1).exp().double().numpy()
        mix = model(bb, te).exp().double().numpy()
        b0 = getattr(m0, "data_view", None) or b
        num = m0(b0, b0.idx("test")).exp().double().numpy()
    J = s.shape[1]
    nll = lambda p: float(-np.log(np.clip(p[n, y], 1e-300, 1)).mean())
    unif = (1 - pi) * s + pi * np.full_like(s, 1.0 / J)
    return {"dataset": dataset, "protocol": protocol, "pi": pi, "chance": float(np.log(J)),
            "numeric_only": nll(num), "sentences_alone": nll(q), "mixture": nll(mix),
            "uniform_floor_same_pi": nll(unif), "sentences_beyond_the_floor": nll(unif) - nll(mix),
            "total_mixture_gain": nll(num) - nll(mix)}


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["lpmc", "optima", "swissmetro", "amazon"]):
        r = run(ds)
        print(f"{r['dataset']:11s} pi={r['pi']:.3f} numeric {r['numeric_only']:.4f} | sentences alone "
              f"{r['sentences_alone']:.4f} (chance {r['chance']:.4f}) | mixture {r['mixture']:.4f} | "
              f"uniform floor {r['uniform_floor_same_pi']:.4f} | beyond the floor "
              f"{r['sentences_beyond_the_floor']:+.4f} of {r['total_mixture_gain']:+.4f}")
