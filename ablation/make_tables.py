"""Tables from ablation/<name>/results/*.json -> ablation/<name>/results.md and ablation/README.md.

    venv/bin/python ablation/make_tables.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ABL_DIR, DATASETS, REFERENCE, discover, fmt_delta, load_results, paired_delta  # noqa: E402

PROTO_LABEL = {"chrono": "chronological within-person split", "person": "person-level split (no test person in training)"}
PI_LABEL = {"val": "π fitted on the validation split", "oof": "π stacked out of fold"}
ORDER_BASE = ["no_sentences", "plain_nn_numeric", "plain_nn_sentences", "plain_nn_sentences_flat", "plain_nn_sentences_person",
              "plain_nn_all_inputs", "full_model"]
ORDER_DESIGN = ["full_model", "no_person_weights", "single_head", "no_salience", "person_attention", "no_projection", "no_probe",
                "no_slot_dropout", "single_member"]


def collect():
    """(protocol, pi, dataset, name) -> list of records (one per seed)."""
    R = defaultdict(list)
    for name in discover():
        for r in load_results(name):
            R[(r["protocol"], r["pi_fit"], r["dataset"], name)].append(r)
    return R


def pooled(recs, key):
    """Concatenate a per-event list over seeds (sorted by seed) or return None."""
    recs = sorted(recs, key=lambda r: r["seed"])
    vals = [r.get(key) for r in recs]
    if any(v is None for v in vals) or not vals:
        return None, [r["seed"] for r in recs]
    return list(np.concatenate([np.asarray(v) for v in vals])), [r["seed"] for r in recs]


def cell_sem(R, proto, pi, ds, name):
    recs = R.get((proto, pi, ds, name), [])
    if not recs or recs[0].get("sem_only") is None:
        return "—", "—"
    nll = np.mean([r["sem_only"]["nll"] for r in recs])
    a, sa = pooled(recs, "sem_only_per_event_nll")
    b, sb = pooled(R.get((proto, pi, ds, REFERENCE), []), "sem_only_per_event_nll")
    d = paired_delta(a, b) if (a is not None and b is not None and sa == sb) else None
    return f"{nll:.4f}", fmt_delta(d)


def cell_mix(R, proto, pi, ds, name):
    recs = R.get((proto, pi, ds, name), [])
    if not recs:
        return "—", "—", "—"
    nll = np.mean([r["nll"] for r in recs])
    pis = [r.get("extra", {}).get("pi", 0.0) for r in recs]
    a, sa = pooled(recs, "per_event_nll")
    b, sb = pooled(R.get((proto, pi, ds, REFERENCE), []), "per_event_nll")
    d = paired_delta(a, b) if (a is not None and b is not None and sa == sb) else None
    return f"{nll:.4f}", f"{np.mean(pis):.2f}", fmt_delta(d)


def table(R, proto, pi, names, mods, kind):
    head = "| variant | " + " | ".join(f"{ds}: NLL | Δ vs ref" + (" | π" if kind == "mix" else "") for ds in DATASETS) + " |"
    sep = "|---|" + "|".join("---|---" + ("|---" if kind == "mix" else "") for _ in DATASETS) + "|"
    rows = [head, sep]
    for n in names:
        if n not in mods:
            continue
        cells = []
        for ds in DATASETS:
            if kind == "sem":
                nll, d = cell_sem(R, proto, pi, ds, n); cells += [nll, d]
            else:
                nll, p, d = cell_mix(R, proto, pi, ds, n); cells += [nll, d, p]
        label = mods[n].LABEL + (" **(reference)**" if n == REFERENCE else "")
        rows.append(f"| `{n}` {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def decomposition(R, proto, pi):
    """How much comes from the LLM sentences and how much from the designed model (mean NLL over seeds)."""
    def nll(name, sem=False):
        out = []
        for ds in DATASETS:
            recs = R.get((proto, pi, ds, name), [])
            if not recs:
                out.append(None); continue
            out.append(np.mean([r["sem_only"]["nll"] if sem else r["nll"] for r in recs]) if (not sem or recs[0].get("sem_only")) else None)
        return out
    rows = [
        ("no LLM, no designed sentence model: structural utility + residual", nll("no_sentences")),
        ("no LLM, no structure: plain MLP on numeric inputs", nll("plain_nn_numeric", sem=True)),
        ("LLM sentences, no structure: plain MLP on the sentence embedding", nll("plain_nn_sentences", sem=True)),
        ("LLM sentences + z_i, no structure: plain MLP", nll("plain_nn_sentences_person", sem=True)),
        ("all inputs, no structure: plain MLP on [sentences, z_i, x_ij, h_ij]", nll("plain_nn_all_inputs", sem=True)),
        ("LLM sentences + designed sentence model (alone)", nll("full_model", sem=True)),
        ("plain sentence MLP mixed with the structural model", nll("plain_nn_sentences")),
        ("full OM-LEU 2 (designed sentence model mixed with the structural model)", nll("full_model")),
    ]
    out = ["| model | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for lab, v in rows:
        out.append(f"| {lab} | " + " | ".join("—" if x is None else f"{x:.4f}" for x in v) + " |")
    return "\n".join(out)


def per_seed(recs):
    rows = ["| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: (r["protocol"], r["pi_fit"], DATASETS.index(r["dataset"]), r["seed"])):
        so = r["sem_only"]["nll"] if r.get("sem_only") else float("nan")
        ex = r.get("extra", {})
        rows.append(f"| {r['protocol']} | {r['pi_fit']} | {r['dataset']} | {r['seed']} | {r['nll']:.4f} | {ex.get('pi', 0):.3f} | "
                    f"{ex.get('temp_a', 1):.2f} | {so:.4f} | {r['top1']*100:.1f}% |")
    return "\n".join(rows)


def main():
    mods = discover()
    R = collect()
    combos = sorted({(p, q) for (p, q, _, _) in R}, key=lambda t: (t[0] != "chrono", t[1] != "val"))
    md = ["# Ablations: LLM sentences vs the designed model\n",
          "Every row is OM-LEU 2 with one change, three seeds per dataset. **Sentence-only NLL** is the sentence channel "
          "evaluated alone (five members averaged in probability, no structural model). **Mixture NLL** is the full system "
          "p = (1−π)·q + π·p̄ with that sentence channel. Δ is the paired per-event ΔNLL against `full_model` pooled over "
          "seeds with a 2000-draw bootstrap 95 % CI; `*` = CI excludes 0; negative = better than the reference. "
          "Plain-MLP baselines use exactly the same consequence sentences and embeddings as the full model.\n",
          "Code: `ablation/<name>/model.py`; per-run JSON: `ablation/<name>/results/`; maths: `ablation/<name>/breakdown.md`; "
          "runner: `ablation/run.py`; this file: `ablation/make_tables.py`.\n"]
    for proto, pi in combos:
        md.append(f"\n## {PROTO_LABEL[proto]} — {PI_LABEL[pi]}\n")
        md.append("### Where the improvement comes from (mean test NLL over seeds)\n")
        md.append(decomposition(R, proto, pi) + "\n")
        md.append("### A. Same sentences, unstructured baselines — sentence channel alone\n")
        md.append(table(R, proto, pi, ORDER_BASE, mods, "sem") + "\n")
        md.append("### A'. Same baselines mixed with the structural model\n")
        md.append(table(R, proto, pi, ORDER_BASE, mods, "mix") + "\n")
        md.append("### B. Knock-outs inside the designed sentence model — sentence channel alone\n")
        md.append(table(R, proto, pi, ORDER_DESIGN, mods, "sem") + "\n")
        md.append("### B'. Same knock-outs mixed with the structural model\n")
        md.append(table(R, proto, pi, ORDER_DESIGN, mods, "mix") + "\n")
    (ABL_DIR / "README.md").write_text("\n".join(md))
    for name, mod in mods.items():
        recs = load_results(name)
        out = [f"# `{name}`: {mod.LABEL}\n", f"Config overrides: `{mod.CONFIG}`  \nMaths: `breakdown.md`\n"]
        for proto, pi in combos:
            out.append(f"\n## {PROTO_LABEL[proto]} — {PI_LABEL[pi]}\n")
            out.append(table(R, proto, pi, [REFERENCE, name] if name != REFERENCE else [name], mods, "sem").replace("| variant |", "| sentence channel alone |") + "\n")
            out.append(table(R, proto, pi, [REFERENCE, name] if name != REFERENCE else [name], mods, "mix").replace("| variant |", "| mixed with the structural model |") + "\n")
        out.append("\n## Per seed\n\n" + per_seed(recs) + "\n")
        (ABL_DIR / name / "results.md").write_text("\n".join(out))
    print("wrote ablation/README.md and", len(mods), "results.md files")


if __name__ == "__main__":
    main()
