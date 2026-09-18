"""Tables and report built only from saved per-event predictions."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .metrics import all_metrics, cluster_draws, event_nll, paired_diff

BASELINE = "numeric_only"


def _load(root: Path) -> List[Dict]:
    out = []
    for f in sorted(Path(root).rglob("result.json")):
        if "superseded" in str(f):
            continue          # archived runs from a fixed defect are never read into a table
        r = json.loads(f.read_text())
        p = f.parent / "predictions.json"
        if p.exists():
            r["_preds"] = json.loads(p.read_text())
            r["_dir"] = str(f.parent)
            out.append(r)
    return out


def _arrays(r: Dict):
    pr = r["_preds"]
    lp = np.array([p["final_logprobs"] for p in pr], dtype=float)
    y = np.array([p["chosen_index"] for p in pr])
    cl = [p["cluster_id"] for p in pr]
    ev = [p["event_id"] for p in pr]
    return lp, y, cl, ev


def evaluate_all(root: Path) -> str:
    rs = _load(root)
    lines = [f"{len(rs)} completed runs under {root}"]
    for r in rs:
        lp, y, cl, _ = _arrays(r)
        m = all_metrics(lp, y)
        lines.append(f"  {r['dataset']:11s} {r['protocol']:22s} {r['variant']:24s} seed{r['master_seed']} "
                     f"nll={m['nll']:.4f} top1={m['top1']*100:.1f}% pi={r['calibration']['pi']:.3f}"
                     + ("  [smoke]" if r.get("smoke") else ""))
    return "\n".join(lines)


def contrast_table(root: Path, protocol: str, baseline: str = BASELINE, smoke: bool = False) -> str:
    rs = [r for r in _load(root) if r["protocol"] == protocol and bool(r.get("smoke")) == smoke]
    by = defaultdict(dict)
    for r in rs:
        by[(r["dataset"], r["master_seed"])][r["variant"]] = r
    datasets = sorted({d for d, _ in by})
    variants = sorted({v for k in by for v in by[k]})
    head = "| variant | " + " | ".join(f"{d}: NLL | ΔNLL vs {baseline} [95% CI]" for d in datasets) + " |"
    rows = [head, "|---|" + "|".join("---|---" for _ in datasets) + "|"]
    for v in variants:
        cells = []
        for d in datasets:
            seeds = sorted(s for (dd, s) in by if dd == d and v in by[(dd, s)] and baseline in by[(dd, s)])
            if not seeds:
                cells += ["—", "—"]; continue
            nll, base_e, var_e, cl = [], [], [], []
            for s in seeds:
                rv, rb = by[(d, s)][v], by[(d, s)][baseline]
                lpv, yv, clv, evv = _arrays(rv); lpb, yb, clb, evb = _arrays(rb)
                assert evv == evb, "variant and baseline evaluated different events"
                nll.append(event_nll(lpv, yv).mean())
                var_e.append(event_nll(lpv, yv)); base_e.append(event_nll(lpb, yb)); cl += clv
            base_e = np.concatenate(base_e); var_e = np.concatenate(var_e)
            d_ = paired_diff(base_e, var_e, cl, cluster_draws(cl))
            star = "*" if d_["significant"] else ""
            cells += [f"{np.mean(nll):.4f}", f"{d_['mean']:+.4f}{star} [{d_['lo']:+.4f}, {d_['hi']:+.4f}]"]
        rows.append(f"| `{v}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(root: Path, out: Optional[Path] = None) -> str:
    out = out or Path(root) / "report.md"
    rs = _load(root)
    protocols = sorted({r["protocol"] for r in rs})
    md = ["# Consequence-modelling experiments: results from saved predictions\n",
          "Positive ΔNLL means the variant improves on the baseline. Intervals are 2,000-draw "
          "bootstraps over independent clusters (household where the identifier encodes one, "
          "otherwise respondent), conditional on the trained models and the observed split. "
          "`*` marks an interval excluding zero.\n"]
    for p in protocols:
        for smoke in (False, True):
            t = contrast_table(root, p, smoke=smoke)
            if t.count("\n") > 1:
                md.append(f"\n## {p}{' (smoke runs, reduced folds/members)' if smoke else ''}\n")
                md.append(t + "\n")
    ledger = Path(root) / "status_ledger.jsonl"
    if ledger.exists():
        rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
        bad = [r for r in rows if r["status"] != "completed"]
        md.append("\n## Failed, blocked and not-run ledger\n")
        md.append("| status | dataset | protocol | variant | seed | reason |\n|---|---|---|---|---|---|")
        for r in bad:
            md.append(f"| {r['status']} | {r.get('dataset','')} | {r.get('protocol','')} | {r.get('variant','')} "
                      f"| {r.get('master_seed','')} | {str(r.get('reason',''))[:160]} |")
        if not bad:
            md.append("| (none) | | | | | |")
    Path(out).write_text("\n".join(md))
    return str(out)
