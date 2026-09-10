"""Monotonicity probe for the complete pipeline.

For a mixture ``p_j = (1 - pi(x)) q_j(x) + pi(x) r_j(x)`` the derivative with respect to an
attribute is

    dp_j/dx = (1 - pi) dq_j/dx + pi dr_j/dx + (r_j - q_j) dpi/dx,

so a sign constraint inside one component does not certify the mixture, and when the gate
depends on an attribute the last term has to be included.  The probe measures all three
terms empirically instead of asserting a guarantee.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def gate_monotonicity_probe(p_lo: np.ndarray, p_hi: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                            r_lo: np.ndarray, r_hi: np.ndarray, pi_lo: np.ndarray, pi_hi: np.ndarray,
                            slot: int) -> Dict[str, float]:
    """Compare the change in the final probability of ``slot`` when one attribute of that
    alternative is made worse, and decompose it into the three terms above."""
    d_final = p_hi[:, slot] - p_lo[:, slot]
    pi_bar = 0.5 * (pi_lo + pi_hi)
    t_numeric = (1 - pi_bar) * (q_hi[:, slot] - q_lo[:, slot])
    t_semantic = pi_bar * (r_hi[:, slot] - r_lo[:, slot])
    t_gate = (0.5 * (r_hi[:, slot] + r_lo[:, slot]) - 0.5 * (q_hi[:, slot] + q_lo[:, slot])) * (pi_hi - pi_lo)
    return {"violation_rate": float((d_final > 1e-9).mean()),
            "mean_change": float(d_final.mean()),
            "term_numeric": float(t_numeric.mean()), "term_semantic": float(t_semantic.mean()),
            "term_gate": float(t_gate.mean()),
            "decomposition_residual": float(np.abs(d_final - (t_numeric + t_semantic + t_gate)).mean())}
