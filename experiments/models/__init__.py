"""Variant registry: name -> factory(bundle, seed) -> (model, run_fn).

A run_fn(model, bundle, seed) performs the full training protocol (prefit +
joint fit, or whatever the variant needs) and returns a dict with at least
``fit`` (FitResult-like dict) and ``extra`` (anything worth reporting).
Register new variants by adding a module under experiments/models/ and an
entry in REGISTRY (see baseline.py for the pattern).
"""
from importlib import import_module

REGISTRY = {
    "mnl_only": ("experiments.models.baseline", "build_mnl_only"),
    "hybrid_base": ("experiments.models.baseline", "build_hybrid_base"),
    "pref_dual": ("experiments.models.pref", "build_pref_dual"),
    "pref_dual_sem": ("experiments.models.pref", "build_pref_dual_sem"),
    "pref_mix": ("experiments.models.pref", "build_pref_mix"),
    "pref_mix_sem": ("experiments.models.pref", "build_pref_mix_sem"),
    "pref_mnl_cal": ("experiments.models.pref", "build_pref_mnl_cal"),
    "pref_mnl_unif": ("experiments.models.pref", "build_pref_mnl_unif"),
}


def get(name: str):
    mod, fn = REGISTRY[name]
    return getattr(import_module(mod), fn)
