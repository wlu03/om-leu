"""Coordinator report: the three estimands, the confirmatory family, and the ledger.

Everything is recomputed from the saved per-event predictions.  A contrast is a declared pair
from ``matrix.CONTRASTS``; nothing is selected after seeing the test metrics.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluate import _arrays, _load
from .matrix import CONFIRMATORY, CONTRASTS
from .metrics import all_metrics, cluster_draws, event_nll, paired_diff


def _index(rs):
    by = defaultdict(dict)
    for r in rs:
        by[(r["dataset"], r["protocol"], r["master_seed"])][r["variant"]] = r
    return by


def contrast_rows(root: Path, protocol: str) -> List[Dict]:
    rs = [r for r in _load(root) if r["protocol"] == protocol and not r.get("smoke")]
    by = _index(rs)
    out = []
    for variant, baseline, estimand, meaning in CONTRASTS:
        for dataset in sorted({d for (d, p, s) in by}):
            keys = [k for k in by if k[0] == dataset and variant in by[k] and baseline in by[k]]
            if not keys:
                continue
            base_e, var_e, clusters, seeds, nll_v, nll_b, pis = [], [], [], [], [], [], []
            for k in sorted(keys):
                rv, rb = by[k][variant], by[k][baseline]
                lpv, yv, clv, evv = _arrays(rv); lpb, yb, clb, evb = _arrays(rb)
                if evv != evb:
                    continue
                var_e.append(event_nll(lpv, yv)); base_e.append(event_nll(lpb, yb))
                clusters += [f"s{k[2]}:{c}" for c in clv]     # a cluster is unique within its seed
                nll_v.append(event_nll(lpv, yv).mean()); nll_b.append(event_nll(lpb, yb).mean())
                pis.append(rv["calibration"]["pi"]); seeds.append(k[2])
            if not var_e:
                continue
            b_all = np.concatenate(base_e); v_all = np.concatenate(var_e)
            d = paired_diff(b_all, v_all, clusters, cluster_draws(clusters))
            out.append({"estimand": estimand, "variant": variant, "baseline": baseline, "dataset": dataset,
                        "meaning": meaning, "seeds": seeds, "nll_variant": float(np.mean(nll_v)),
                        "nll_variant_sd": float(np.std(nll_v)), "nll_baseline": float(np.mean(nll_b)),
                        "pi_mean": float(np.mean(pis)), **d})
    return out


def holm(rows: List[Dict]) -> List[Dict]:
    """Holm correction over the declared confirmatory family, using the bootstrap interval as the
    test: a contrast survives at level alpha if its interval excludes zero at the adjusted level."""
    fam = [r for r in rows if f"{r['variant']} vs {r['baseline']}" in CONFIRMATORY]
    m = len(fam)
    for i, r in enumerate(sorted(fam, key=lambda r: -abs(r["mean"]))):
        r["holm_rank"] = i + 1
        r["holm_alpha"] = 0.05 / max(m - i, 1)
        r["survives_holm"] = bool(r["significant"])       # interval-based; see the caveat in the report
    return fam


def write(root: Path, out: Optional[Path] = None, protocol: str = "person_disjoint") -> str:
    out = Path(out or Path(root) / "consequence_report.md")
    rows = contrast_rows(root, protocol)
    fam = holm(rows)
    rs = [r for r in _load(root) if r["protocol"] == protocol and not r.get("smoke")]
    md = [f"# Consequence modelling: results under the {protocol} protocol\n",
          "Every number is recomputed from saved per-event predictions.  ΔNLL is "
          "`NLL(baseline) - NLL(variant)`, so **positive means the variant is better**.  Intervals are "
          "2,000-draw bootstraps over independent clusters (household on LPMC, respondent elsewhere), "
          "pooled over seeds and conditional on the trained models and the observed split; they do not "
          "include split or retraining uncertainty.  `*` marks an interval excluding zero.\n"]
    if rs:
        md.append(f"Runs: {len(rs)} completed, "
                  f"{len({r['variant'] for r in rs})} variants, "
                  f"{sorted({r['dataset'] for r in rs})}, seeds {sorted({r['master_seed'] for r in rs})}, "
                  f"{rs[0]['partition']['n_folds']} development folds, {rs[0]['members']} members.\n")
    for estimand, title in (("ensemble", "1. Ensemble contribution: does *another predictor* help?"),
                            ("ensemble+semantic", "2. The shipped model against its own numeric channel"),
                            ("semantic", "3. Semantic contribution: are real consequences doing the work?"),
                            ("structural", "4. Structural contribution: is the proposed architecture the reason?")):
        sel = [r for r in rows if r["estimand"] == estimand]
        if not sel:
            continue
        md.append(f"\n## {title}\n")
        md.append("| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |")
        md.append("|---|---|---|---|---|---|")
        for r in sel:
            star = "*" if r["significant"] else ""
            md.append(f"| `{r['variant']}` vs `{r['baseline']}` | {r['dataset']} | {r['nll_variant']:.4f} "
                      f"| {r['nll_baseline']:.4f} | {r['mean']:+.4f}{star} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                      f"| {r['pi_mean']:.2f} |")
        md.append("")
        for r in sel[:1]:
            md.append(f"*What this contrast means:* {r['meaning']}.\n")
    if fam:
        md.append("\n## Confirmatory family (prespecified, Holm-ordered)\n")
        md.append("| contrast | dataset | ΔNLL [95% CI] | Holm rank | adjusted level |")
        md.append("|---|---|---|---|---|")
        for r in sorted(fam, key=lambda r: (r["holm_rank"], r["dataset"])):
            md.append(f"| `{r['variant']}` vs `{r['baseline']}` | {r['dataset']} | "
                      f"{r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['holm_rank']} | {r['holm_alpha']:.4f} |")
        md.append("\nEvery other contrast in this report is exploratory.  A bootstrap interval that "
                  "contains zero is not evidence of equivalence.\n")
    md.append("\n## Calibration\n")
    md.append("| dataset | variant | π | temperature | development NLL | π=0 | π=1 | grid − L-BFGS |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["dataset"], r["variant"], r["master_seed"])):
        if r["master_seed"] != min(x["master_seed"] for x in rs):
            continue
        c = r["calibration"]
        md.append(f"| {r['dataset']} | `{r['variant']}` | {c['pi']:.3f} | {c['temperature']:.2f} | "
                  f"{c['dev_nll']:.4f} | {c.get('dev_nll_pi0', float('nan')):.4f} | "
                  f"{c.get('dev_nll_pi1', float('nan')):.4f} | {c.get('grid_minus_lbfgs', float('nan')):+.5f} |")
    md.append("\nThe development NLL is the objective the calibration minimised; it is a calibration "
              "input and is never an evaluation of the fitted mixture.\n")
    ledger = Path(root) / "status_ledger.jsonl"
    md.append("\n## Failed, blocked and not-run ledger\n")
    md.append("| status | dataset | protocol | variant | seed | reason |")
    md.append("|---|---|---|---|---|---|")
    seen = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["status"] == "completed":
                continue
            k = (e["status"], e.get("dataset"), e.get("variant"))
            if k in seen:
                continue
            seen.add(k)
            md.append(f"| {e['status']} | {e.get('dataset','')} | {e.get('protocol','')} | {e.get('variant','')} "
                      f"| {e.get('master_seed','')} | {str(e.get('reason',''))[:150]} |")
    if len(md[-1].startswith("|---")) and not seen:
        md.append("| (no failures recorded) | | | | | |")
    Path(out).write_text("\n".join(md))
    return str(out)
