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
    # numeric-aware concept heads (experiments/models/ncat.py)
    "ncat_v2": ("experiments.models.ncat", "build_ncat_v2"),
    "ncat_v3": ("experiments.models.ncat", "build_ncat_v3"),
    # preference-aligned semantic ensemble mixed at probability level (experiments/models/pref.py)
    "pref_dual": ("experiments.models.pref", "build_pref_dual"),
    "pref_dual_sem": ("experiments.models.pref", "build_pref_dual_sem"),
    "pref_mix": ("experiments.models.pref", "build_pref_mix"),
    "pref_mix_sem": ("experiments.models.pref", "build_pref_mix_sem"),
    "pref_mnl_cal": ("experiments.models.pref", "build_pref_mnl_cal"),
    "pref_mnl_unif": ("experiments.models.pref", "build_pref_mnl_unif"),
}
# boosted-residual variants (experiments/models/boost.py)
REGISTRY.update({name: ("experiments.models.boost", f"build_{name}") for name in (
    "boost_v2", "boost_v2_sem", "boost_v3", "boost_v3_sem")})
# OM-LEU 2: hetero_v1 structural part -> boosted residual -> semantic mixture (experiments/models/omleu2.py)
REGISTRY.update({name: ("experiments.models.omleu2", f"build_{name}") for name in (
    "combo_struct", "combo_struct_cal", "combo_full", "combo_full_ncat", "combo_struct_ncat", "combo_struct_insample")})


def get(name: str):
    mod, fn = REGISTRY[name]
    return getattr(import_module(mod), fn)
