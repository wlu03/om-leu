"""Emit every LaTeX table in the paper from the saved per-run artifacts.

Run from the repository root:

    venv/bin/python paper/make_tables.py

Sources
-------
``ablation/<variant>/results/*.json``  the ablation programme (4 datasets, 3 settings)
``<artifact-root>/**/result.json``     the nested consequence programme (E1-E7)

Nothing is typed by hand: every number in the paper is written by this script.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
_env = os.environ.get("OMLEU_ARTIFACT_ROOT", "").strip()
AR = Path(_env) if _env else None
if AR is None or not (AR / "status_ledger.jsonl").exists():
    cands = sorted(ROOT.parent.glob("*-consequence-*-artifacts"))
    AR = cands[-1] if cands else None

DS4 = ["swissmetro", "optima", "lpmc", "amazon"]
DS3 = ["swissmetro", "optima", "lpmc"]
PRETTY = {"swissmetro": "Swissmetro", "optima": "Optima", "lpmc": "LPMC", "amazon": "Amazon"}


# ----------------------------------------------------------------- loaders
def load_ablation():
    R = defaultdict(list)
    for f in glob.glob(str(ROOT / "ablation/*/results/*.json")):
        d = json.load(open(f))
        if "error" in d:
            continue
        R[(d["dataset"], d["protocol"], d["pi_fit"], d["ablation"])].append(d)
    return R


def load_consequence():
    R = defaultdict(list)
    if AR is None:
        return R
    for f in glob.glob(str(AR / "**/result.json"), recursive=True):
        if "superseded" in f:
            continue
        d = json.load(open(f))
        if d.get("smoke"):
            continue
        R[(d["dataset"], d["protocol"], d["variant"])].append(d)
    return R


AB, CQ = load_ablation(), load_consequence()


def ab(dataset, protocol, pi, variant, field="nll", sem=False):
    rs = AB.get((dataset, protocol, pi, variant), [])
    if not rs:
        return None
    if sem:
        vals = [r["sem_only"][field] for r in rs if r.get("sem_only")]
    else:
        vals = [r[field] for r in rs if field in r]
    return (float(np.mean(vals)), len(vals)) if vals else None


def cq(dataset, protocol, variant, key=("metrics_final", "nll")):
    rs = CQ.get((dataset, protocol, variant), [])
    if not rs:
        return None
    vals = []
    for r in rs:
        v = r
        for k in key:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if v is not None:
            vals.append(v)
    return (float(np.mean(vals)), len(vals)) if vals else None


def fmt(x, nd=3):
    return "--" if x is None else f"{x[0]:.{nd}f}"


def fmtd(x, ref, nd=3, star=False):
    if x is None or ref is None:
        return "--"
    d = ref[0] - x[0]
    return f"{d:+.{nd}f}" + ("$^{*}$" if star else "")


def write(name, body):
    (OUT / name).write_text(body)
    print(f"  wrote paper/tables/{name}")


# ----------------------------------------------------------------- tables
def table_main():
    """Ensemble, semantic and structural contributions under the person-disjoint protocol."""
    rows = [
        ("numeric only", "no_sentences", "numeric_only"),
        ("\\quad + second numeric channel", "plain_nn_numeric", "numeric_auxiliary"),
        ("\\quad + random-vector channel", None, "preserved_random"),
        ("\\quad + identity-text channel", None, "preserved_identity"),
        ("\\quad + shuffled consequences", None, "preserved_shuffled"),
        ("\\quad + templated facts", None, "preserved_template"),
        ("\\quad + LLM consequences", "full_model", "preserved_llm"),
    ]
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "channel added to the numeric model & " + " & ".join(PRETTY[d] for d in DS4) + " \\\\", "\\midrule"]
    for label, av, cv in rows:
        cells = []
        for d in DS4:
            v = cq(d, "person_disjoint", cv) if d != "amazon" else None
            if v is None and av is not None:
                v = ab(d, "person", "oof", av)
            cells.append(fmt(v))
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("main.tex", "\n".join(L))


def table_structural():
    """Designed readers against ordinary networks given the same information."""
    rows = [("shipped reader (attention, heads, person weights)", "preserved_llm"),
            ("axis-preserving reader (\\S\\ref{sec:axis})", "axis_llm"),
            ("\\quad + anchor supervision", "axis_llm_anchored"),
            ("two-layer network, mean-pooled sentences", "plain_mean_llm"),
            ("two-layer network, axis slots kept", "plain_slots_llm"),
            ("two-layer network, slots + covariates", "plain_slots_z_llm"),
            ("two-layer network, all inputs", "plain_all_inputs_llm")]
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "sentence reader & " + " & ".join(PRETTY[d] for d in DS3) + " \\\\", "\\midrule"]
    for label, v in rows:
        L.append(f"{label} & " + " & ".join(fmt(cq(d, "person_disjoint", v)) for d in DS3) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("structural.tex", "\n".join(L))


def table_knockouts():
    """Knock-outs inside the designed reader: channel alone, then inside the mixture."""
    rows = [("personalised head weights", "no_person_weights"),
            ("\\quad learned but shared weights", "global_weights"),
            ("five attribute heads", "single_head"),
            ("salience attention", "no_salience"),
            ("low-rank projection", "no_projection"),
            ("auxiliary probe loss", "no_probe"),
            ("sentence-slot dropout", "no_slot_dropout"),
            ("five-member ensemble", "single_member")]
    ref_s = {d: ab(d, "person", "oof", "full_model", sem=True) for d in DS3}
    ref_m = {d: ab(d, "person", "oof", "full_model") for d in DS3}
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "& \\multicolumn{3}{c}{channel alone} & \\multicolumn{3}{c}{inside the mixture} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "component removed & " + " & ".join(PRETTY[d][:4] for d in DS3) * 2 + " \\\\", "\\midrule"]
    for label, v in rows:
        a = [fmtd(ab(d, "person", "oof", v, sem=True), ref_s[d]) for d in DS3]
        m = [fmtd(ab(d, "person", "oof", v), ref_m[d]) for d in DS3]
        L.append(f"{label} & " + " & ".join(a + m) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("knockouts.tex", "\n".join(L))


def table_protocol():
    """Person-disjoint against time-respecting evaluation on the one dataset that supports both."""
    rows = [("numeric only", "numeric_only"), ("second numeric channel", "numeric_auxiliary"),
            ("shuffled consequences", "preserved_shuffled"), ("templated facts", "preserved_template"),
            ("LLM consequences", "preserved_llm"), ("axis reader on consequences", "axis_llm")]
    L = ["\\begin{tabular}{lcc}", "\\toprule",
         "LPMC & person-disjoint & time-respecting \\\\", "\\midrule"]
    for label, v in rows:
        L.append(f"{label} & {fmt(cq('lpmc','person_disjoint',v))} & {fmt(cq('lpmc','temporal',v))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("protocol.tex", "\n".join(L))


def table_improvements():
    """The four proposed improvements, each against its own control."""
    L = ["\\begin{tabular}{llcc}", "\\toprule",
         "proposal & comparison & baseline & proposed \\\\", "\\midrule"]
    pairs = [("grounding by construction (E1)", "LPMC, templates vs LLM text",
              cq("lpmc", "person_disjoint", "preserved_llm"), cq("lpmc", "person_disjoint", "preserved_template")),
             ("consistency training (E3)", "LPMC, no loss vs both losses",
              cq("lpmc", "person_disjoint", "e3_consistency_none"), cq("lpmc", "person_disjoint", "e3_consistency_both")),
             ("complementarity training (E4)", "LPMC, independent vs mixture-aware",
              cq("lpmc", "person_disjoint", "preserved_llm"), cq("lpmc", "person_disjoint", "mixture_aware_preserved")),
             ("conditional gate (E7)", "LPMC, global vs conditional weight",
              cq("lpmc", "person_disjoint", "gate_global_llm"), cq("lpmc", "person_disjoint", "gate_conditional_llm"))]
    for name, comp, a, b in pairs:
        L.append(f"{name} & {comp} & {fmt(a)} & {fmt(b)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("improvements.tex", "\n".join(L))


def table_transfer():
    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "target & source & frozen scorer & fine-tuned & fitted on target \\\\", "\\midrule"]
    for t in DS3:
        base = cq(t, "person_disjoint", "axis_llm")
        for s in DS3:
            if s == t:
                continue
            fr = cq(t, "person_disjoint", f"transfer_from_{s}")
            ft = cq(t, "person_disjoint", f"transfer_from_{s}_finetune")
            if fr is None:
                continue
            L.append(f"{PRETTY[t]} & {PRETTY[s]} & {fmt(fr)} & {fmt(ft)} & {fmt(base)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("transfer.tex", "\n".join(L))


def table_curve():
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "training households & axis reader, LLM text & axis reader, templates & shipped reader \\\\", "\\midrule"]
    for frac, n in ((10, 45), (25, 114), (50, 227), (100, 454)):
        a = cq("lpmc", "person_disjoint", f"lc_axis_llm_{frac:03d}")
        b = cq("lpmc", "person_disjoint", f"lc_axis_template_{frac:03d}")
        c = cq("lpmc", "person_disjoint", f"lc_preserved_llm_{frac:03d}")
        L.append(f"{n} ({frac}\\%) & {fmt(a)} & {fmt(b)} & {fmt(c)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("curve.tex", "\n".join(L))


def table_pi():
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "channel & " + " & ".join(PRETTY[d] for d in DS3) + " \\\\", "\\midrule"]
    for label, v in [("LLM consequences", "preserved_llm"), ("templated facts", "preserved_template"),
                     ("shuffled consequences", "preserved_shuffled"), ("identity text", "preserved_identity"),
                     ("random vectors", "preserved_random"), ("second numeric channel", "numeric_auxiliary")]:
        L.append(f"{label} & " + " & ".join(fmt(cq(d, "person_disjoint", v, ("calibration", "pi")), 2) for d in DS3) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("pi.tex", "\n".join(L))


def table_benchmarks():
    """Nine reference specifications under the paper's own protocol, with matched features."""
    import glob as _g
    import numpy as _np
    LABEL = {"mnl": "Multinomial logit", "nested_logit": "Nested logit", "mixed_logit": "Mixed logit",
             "iclv_hybrid_choice": "ICLV hybrid choice", "l_mnl": "L-MNL \\citep{sifringer2020enhancing}",
             "asu_dnn": "ASU-DNN \\citep{wang2020deep}",
             "tastenet_mnl": "TasteNet-MNL \\citep{han2022tastenet}",
             "rumboost": "RUMBoost \\citep{salvade2024rumboost}",
             "gbdt_full_features": "Gradient boosting \\citep{hillel2021systematic}"}
    R = {}
    for f in _g.glob(str(ROOT / "methods/results_person_hist/*/*/seed_*.json")):
        d = json.load(open(f))
        if "nll" not in d:
            continue
        parts = Path(f).parts
        R.setdefault(parts[-3], {}).setdefault(parts[-2], []).append(d["nll"])
    if not R:
        return
    order = ["mnl", "nested_logit", "mixed_logit", "iclv_hybrid_choice", "l_mnl", "asu_dnn",
             "tastenet_mnl", "rumboost", "gbdt_full_features"]
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "specification & " + " & ".join(PRETTY[d] for d in DS3) + " \\\\", "\\midrule"]
    for m in order:
        cells = []
        for d in DS3:
            v = R.get(d, {}).get(m)
            cells.append(f"{_np.mean(v):.3f}" if v else "--")
        if set(cells) == {"--"}:
            continue
        L.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    for label, v in (("proposed, attribute channel only", "no_sentences"),
                     ("proposed, with the language channel", "full_model")):
        cells = [fmt(ab(d, "person", "oof", v), 3) for d in DS3]
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("benchmarks.tex", "\n".join(L))


def table_identification():
    """Each identified-channel variant against its OWN shuffled control."""
    import subprocess
    PAIRS = [("id_plain", "id_plain_shuffled", "no orthogonalisation"),
             ("id_erase", "id_erase_shuffled", "erasure"),
             ("id_resid", "id_resid_shuffled", "residual offset"),
             ("id_erase_resid", "id_erase_resid_shuffled", "erasure + residual offset")]
    import numpy as _np
    res_dir = ROOT / "experiments" / "results"

    def ev(variant, ds, seed):
        f = res_dir / variant / f"{ds}_seed{seed}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        return None if "error" in d else _np.array(d["per_event_nll"])

    def cl(ds, seed, n):
        try:
            import sys as _s
            _s.path.insert(0, str(ROOT))
            from experiments.harness.data import load_bundle
            from experiments.models.omleu2 import cold_start_view
            b = load_bundle(ds, seed); bv = cold_start_view(b, seed); te = bv.idx("test")
            if len(te) == n:
                return _np.array([f"s{seed}:{int(p)}" for p in b.person[te].numpy()])
        except Exception:
            pass
        return _np.array([f"s{seed}:{i}" for i in range(n)])

    def paired(a, b_, c, draws=2000):
        d = b_ - a
        uniq = sorted(set(c.tolist())); pos = {x: i for i, x in enumerate(uniq)}
        idx = [[] for _ in uniq]
        for i, x in enumerate(c):
            idx[pos[x]].append(i)
        sums = _np.array([d[i].sum() for i in idx]); cnts = _np.array([len(i) for i in idx], dtype=float)
        rng = _np.random.default_rng(0); dr = rng.integers(0, len(uniq), (draws, len(uniq)))
        bt = sums[dr].sum(1) / _np.maximum(cnts[dr].sum(1), 1e-12)
        lo, hi = _np.quantile(bt, [0.025, 0.975])
        return float(d.mean()), float(lo), float(hi)

    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "dataset & channel & NLL & its control & difference [95\\% CI] \\\\", "\\midrule"]
    any_row = False
    for ds in DS4:
        first = True
        for var, ctrl, label in PAIRS:
            A, B, C, n = [], [], [], 0
            for seed in (7, 11, 13):
                a, b_ = ev(var, ds, seed), ev(ctrl, ds, seed)
                if a is None or b_ is None or len(a) != len(b_):
                    continue
                A.append(a); B.append(b_); C.append(cl(ds, seed, len(a))); n += 1
            if not n:
                continue
            a, b_, c = _np.concatenate(A), _np.concatenate(B), _np.concatenate(C)
            m, lo, hi = paired(a, b_, c)
            star = "$^{*}$" if (lo > 0 or hi < 0) else ""
            name = PRETTY[ds] if first else ""
            L.append(f"{name} & {label} & {a.mean():.4f} & {b_.mean():.4f} & "
                     f"{m:+.4f}{star} [{lo:+.4f}, {hi:+.4f}] \\\\")
            first = False; any_row = True
        if not first:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    if any_row:
        write("identification.tex", "\n".join(L))


def table_floor():
    """The uniform-floor control: how much of the mixture gain is information."""
    f = OUT / "floor_control.json"
    if not f.exists():
        return
    rows = {r["dataset"]: r for r in json.load(open(f))}
    L = ["\\begin{tabular}{lccccc}", "\\toprule",
         "& numeric only & channel alone & chance & mixture & uniform floor \\\\", "\\midrule"]
    for d in DS4:
        r = rows.get(d)
        if r is None:
            continue
        L.append(f"{PRETTY[d]} & {r['numeric_only']:.3f} & {r['sentences_alone']:.3f} & "
                 f"{r['chance']:.3f} & {r['mixture']:.3f} & {r['uniform_floor_same_pi']:.3f} \\\\")
    L += ["\\midrule", "sentences beyond the floor & " +
          " & ".join(f"{rows[d]['sentences_beyond_the_floor']:+.3f}" for d in DS4 if d in rows)
          + " & & \\\\" if False else
          "\\multicolumn{6}{l}{\\emph{sentences beyond the floor: " +
          ", ".join(f"{PRETTY[d]} {rows[d]['sentences_beyond_the_floor']:+.3f}" for d in DS4 if d in rows) +
          "}} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    write("floor.tex", "\n".join(L))


if __name__ == "__main__":
    print(f"artifact root: {AR}")
    for fn in (table_main, table_structural, table_knockouts, table_protocol,
               table_improvements, table_transfer, table_curve, table_pi, table_floor,
               table_identification, table_benchmarks):
        fn()
