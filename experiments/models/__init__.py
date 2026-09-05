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
    # heterogeneous / panel-aware structural part (experiments/models/hetero.py)
    "hetero_v1": ("experiments.models.hetero", "build_hetero_v1"),
    "hetero_v2": ("experiments.models.hetero", "build_hetero_v2"),
}


def get(name: str):
    mod, fn = REGISTRY[name]
    return getattr(import_module(mod), fn)
