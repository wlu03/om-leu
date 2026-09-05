"""Structural smoke for the real-slate adapters (expedia_rectour, swissmetro).

Mirrors scripts/run_dataset.py up through model.fit with StubLLMClient +
StubEncoder so the whole chain (YAML schema -> clean -> survey-join ->
state-features -> split -> popularity -> brand -> REAL displayed slates ->
assemble_batch -> POLEU.fit -> compute_all) runs without LLM calls.

    python scripts/smoke_real_slates.py --adapter swissmetro --data-dir swissmetro/data
    python scripts/smoke_real_slates.py --adapter expedia_rectour --data-dir expedia_rectour/data/fixture
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

logger = logging.getLogger("smoke_real_slates")

ADAPTERS: dict[str, dict] = {
    "expedia_rectour": {
        "prompt_version": "v5_hotel_anchored", "min_events": 3,
        "slates_module": "src.data.expedia_slates", "id_kw": "search_id_column",
        "id_default": "search_id", "slate_default": "slate_prop_ids",
    },
    "swissmetro": {
        "prompt_version": "v6_travel_anchored", "min_events": 5,
        "slates_module": "src.data.swissmetro_slates", "id_kw": "scenario_id_column",
        "id_default": "scenario_id", "slate_default": "slate_alt_ids",
    },
    # Generic mode-choice adapters: YAML block ``modechoice:``.
    "optima": {
        "prompt_version": "v7_modechoice_anchored", "min_events": 1, "split_mode": "cold_start",
        "slates_module": "src.data.modechoice_slates", "id_kw": "event_id_column",
        "id_default": "event_id", "slate_default": "slate_alt_ids", "block": "modechoice",
    },
    "lpmc": {
        "prompt_version": "v7_modechoice_anchored", "min_events": 5,
        "slates_module": "src.data.modechoice_slates", "id_kw": "event_id_column",
        "id_default": "event_id", "slate_default": "slate_alt_ids", "block": "modechoice",
    },
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory holding events/persons/impressions/properties CSVs "
                        "(default: <adapter>/data).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-customers", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-events-per-customer", type=int, default=None)
    p.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "default.yaml",
                   help="Config YAML whose regularizers: block is used (e.g. configs/head_aligned.yaml).")
    return p


def _yaml_with_data_dir(yaml_path: Path, block: str, data_dir: Path | None, out_dir: Path) -> Path:
    if data_dir is None:
        return yaml_path
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    d = Path(data_dir)
    doc["dataset"]["events"]["path"] = str(d / "events.csv")
    doc["dataset"]["persons"]["path"] = str(d / "persons.csv")
    ex = doc.setdefault(block, {})
    ex["impressions_path"] = str(d / "impressions.csv")
    ex["properties_path"] = str(d / "properties.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_yaml = out_dir / f"{block}.resolved.yaml"
    tmp_yaml.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return tmp_yaml


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    spec = ADAPTERS[args.adapter]
    out_dir = args.output_dir or (REPO_ROOT / args.adapter / "results" / "stub")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    min_events = (args.min_events_per_customer if args.min_events_per_customer is not None
                  else spec["min_events"])

    import importlib

    from src.data.adapter import YamlAdapter
    from src.data.alt_rendering import register_on_adapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import compute_hotel_aggregates, extract_extra_fields_from_row
    from src.data.expedia_slates import load_alt_catalog
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_brand_map, attach_train_popularity, compute_state_features,
    )
    from src.data.survey_join import join_survey
    from src.train.loop import try_import_subsample_weights

    slates = importlib.import_module(spec["slates_module"])
    block = spec.get("block", args.adapter)
    yaml_path = _yaml_with_data_dir(
        REPO_ROOT / "configs" / "datasets" / f"{args.adapter}.yaml", block,
        args.data_dir, out_dir,
    )
    adapter = YamlAdapter(yaml_path)
    yaml_doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    ex_cfg = yaml_doc.get(block) or {}
    data_root = Path(adapter.schema.events_path).parent
    impressions_path = Path(ex_cfg.get("impressions_path", data_root / "impressions.csv"))
    properties_path = Path(ex_cfg.get("properties_path", data_root / "properties.csv"))

    events = clean_events(adapter.load_events(), adapter.schema)
    persons_raw = adapter.load_persons()
    events = join_survey(events, persons_raw, adapter.schema)
    events = compute_state_features(events)
    counts = events.groupby("customer_id").size()
    keep = set(counts[counts >= int(min_events)].index)
    events = events[events["customer_id"].isin(keep)].reset_index(drop=True)
    if spec.get("split_mode") == "cold_start":
        from src.data.split import cold_start_split
        events = cold_start_split(events, schema=adapter.schema, seed=int(args.seed))
    else:
        events = temporal_split(events, adapter.schema)
    events = attach_train_popularity(events)
    events = attach_train_brand_map(events)

    train_events = events[events["split"] == "train"].copy()
    selected_ids, _w = try_import_subsample_weights(
        train_events, n_customers=int(args.n_customers), seed=int(args.seed),
    )
    if selected_ids is None:
        logger.error("subsample failed")
        return 1
    selected = set(selected_ids.tolist() if hasattr(selected_ids, "tolist") else list(selected_ids))
    # Mirror run_dataset.py: keep selected TRAIN customers plus every
    # customer with no train rows (cold-start val/test customers).
    train_customers = set(events.loc[events["split"] == "train", "customer_id"].unique())
    no_train = set(events["customer_id"].unique()) - train_customers
    events = events[events["customer_id"].isin(selected | no_train)].reset_index(drop=True)
    train_events_subset = events[events["split"] == "train"].copy()
    register_on_adapter(adapter, train_events_subset)

    persons_id_col = adapter.schema.persons_id_column
    surveyed = set(persons_raw[persons_id_col].dropna().astype(str).unique().tolist())
    joint = set(events["customer_id"].astype(str).unique().tolist()) & surveyed
    events = events[events["customer_id"].astype(str).isin(joint)].reset_index(drop=True)
    train_events_subset = events[events["split"] == "train"].copy()
    persons_raw_keep = persons_raw[persons_raw[persons_id_col].astype(str).isin(joint)].copy()
    persons_canonical = adapter.translate_z_d(persons_raw_keep, training_events=train_events_subset)
    if len(train_events_subset) > 0 and "purchase_frequency" in persons_canonical.columns:
        d = pd.to_datetime(train_events_subset["order_date"])
        weeks = max((d.max() - d.min()).days / 7.0, 1.0)
        persons_canonical = persons_canonical.copy()
        persons_canonical["purchase_frequency"] = (
            persons_canonical["purchase_frequency"].astype(float) / weeks
        )
    surviving = set(persons_canonical["customer_id"])
    events = events[events["customer_id"].isin(surviving)].reset_index(drop=True)

    extras_block = (yaml_doc.get("dataset", {}).get("persons", {}).get("c_d_extra_fields", {})) or {}
    customer_to_extras: dict = {}
    for _, raw_row in persons_raw.iterrows():
        cid = str(raw_row.get(persons_id_col, ""))
        if cid in surviving and extras_block:
            customer_to_extras[cid] = extract_extra_fields_from_row(raw_row.to_dict(), extras_block)
    if args.adapter == "expedia_rectour":
        for cid, agg in compute_hotel_aggregates(events, train_only=True).items():
            customer_to_extras.setdefault(str(cid), {}).update(agg)
    # Mirror configs/default.yaml data.enrich_customer_context (true by
    # default in run_dataset.py) so the previewed prompt is the real one.
    from src.data.context_string import compute_customer_aggregates
    default_cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text()) or {}
    data_cfg = dict(default_cfg.get("data") or {})
    data_cfg.update(yaml_doc.get("data") or {})  # dataset YAML override (run_dataset does the same)
    if bool(data_cfg.get("enrich_customer_context", False)):
        for cid, agg in compute_customer_aggregates(events, train_only=True).items():
            customer_to_extras.setdefault(str(cid), {}).update(agg)

    extra_kw = {}
    if block == "modechoice":
        extra_kw["currency"] = ex_cfg.get("currency")
    overrides_fn = slates.make_per_event_alt_overrides_fn(
        events, impressions_path, properties_path,
        **{spec["id_kw"]: str(ex_cfg.get(spec["id_kw"], spec["id_default"]))}, **extra_kw,
    )
    alt_catalog = load_alt_catalog(properties_path) if properties_path.exists() else None
    if block == "modechoice":
        per_event_context_fn = slates.make_per_event_context_fn(
            events, phrase_column=str(ex_cfg.get("phrase_column", "trip_phrase")))
    else:
        per_event_context_fn = slates.make_per_event_context_fn(events)

    records = build_choice_sets(
        events, persons_canonical, adapter,
        seed=int(args.seed), n_resamples=int(adapter.schema.n_resamples),
        n_negatives=int(adapter.schema.choice_set_size) - 1,
        customer_to_extras=customer_to_extras or None,
        per_event_alt_overrides_fn=overrides_fn,
        real_slate_column=str(ex_cfg.get("slate_column", spec["slate_default"])),
        alt_catalog=alt_catalog, per_event_context_fn=per_event_context_fn,
    )
    rec_train = [r for r in records if r.get("split") == "train"]
    rec_val = [r for r in records if r.get("split") == "val"]
    rec_test = [r for r in records if r.get("split") == "test"]
    logger.info("records: train=%d val=%d test=%d", len(rec_train), len(rec_val), len(rec_test))

    slate_col = str(ex_cfg.get("slate_column", spec["slate_default"]))
    slate_sets = [set(str(s).split("|")) for s in events[slate_col].astype(str)]
    mismatched = sum(1 for r, s in zip(records, slate_sets) if set(r["choice_asins"]) != s)
    if mismatched:
        raise AssertionError(f"{mismatched} records do not match their displayed slate")

    sample_cd = rec_train[0]["c_d"] if rec_train else "(no train records)"
    sample_alts = rec_train[0]["alt_texts"] if rec_train else []

    from src.data.batching import assemble_batch, iter_to_torch_batches
    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter
    from src.outcomes.encode import StubEncoder
    from src.outcomes.generate import StubLLMClient, generate_outcomes

    out_cache = OutcomesCache(out_dir / "outcomes.sqlite")
    emb_cache = EmbeddingsCache(out_dir / "embeddings.sqlite")

    def _assemble(recs):
        return assemble_batch(
            recs, adapter=adapter, llm_client=StubLLMClient(),
            encoder=StubEncoder(encoder_id="stub-real-slates", d_e=64),
            outcomes_cache=out_cache, embeddings_cache=emb_cache,
            K=int(args.K), seed=int(args.seed), prompt_version=spec["prompt_version"],
            diversity_filter=diversity_filter, omega=None,
        )

    batch_train = _assemble(rec_train)
    batch_val = _assemble(rec_val) if rec_val else batch_train
    batch_test = _assemble(rec_test) if rec_test else batch_val

    # Render one real prompt so the smoke output shows what the LLM sees.
    class _Capture:
        model_id = "capture"
        def generate(self, messages, **kw):
            self.messages = messages
            from src.outcomes.generate import GenerationResult
            return GenerationResult(text="\n".join(["x"] * int(args.K)), finish_reason="stop", model_id="capture")
    cap = _Capture()
    if rec_train:
        generate_outcomes("c", "a", sample_cd, sample_alts[rec_train[0]["chosen_idx"]],
                          K=int(args.K), seed=0, prompt_version=spec["prompt_version"], client=cap)

    from src.eval.metrics import compute_all
    from src.model.po_leu import POLEU
    from src.train.loop import TrainConfig, fit
    from src.train.regularizers import RegularizerConfig

    p_eff = int(batch_train.z_d.shape[1]); J = int(batch_train.E.shape[1]); d_e = int(batch_train.E.shape[3])
    n_categories = max(1, len(getattr(batch_train, "category_vocab", ()) or ()))
    model = POLEU(K=int(args.K), J=J, d_e=d_e, p=p_eff, n_categories=n_categories)
    tcfg = TrainConfig.from_default(); tcfg.batch_size = int(args.batch_size); tcfg.max_epochs = int(args.n_epochs)
    rcfg = RegularizerConfig.from_default(args.train_config)
    g_train = torch.Generator().manual_seed(int(args.seed))
    n_per_epoch = max(1, math.ceil(len(batch_train) / tcfg.batch_size))
    state = fit(model,
                lambda: iter_to_torch_batches(batch_train, batch_size=tcfg.batch_size, shuffle=True, generator=g_train),
                lambda: iter_to_torch_batches(batch_val, batch_size=tcfg.batch_size, shuffle=False),
                train_cfg=tcfg, reg_cfg=rcfg, total_steps=n_per_epoch * tcfg.max_epochs, seed=int(args.seed))
    model.eval()
    with torch.no_grad():
        logits_test, inter_test = model(batch_test.z_d, batch_test.E)
    metrics = compute_all(logits_test, batch_test.c_star, n_params=model.num_params(), n_train=len(batch_train))
    from src.train.regularizers import head_slot_alignment_diagnostics
    align = head_slot_alignment_diagnostics(inter_test.A)

    summary = {
        "adapter": args.adapter, "structural": True, "stub_llm": True,
        "n_customers_selected": len(selected & surviving),
        "n_train_records": len(batch_train), "n_val_records": len(batch_val), "n_test_records": len(batch_test),
        "p": p_eff, "J": J, "K": int(args.K), "d_e": d_e, "n_categories": n_categories,
        "param_count": model.num_params(), "epochs": tcfg.max_epochs,
        "train_loss": float(state.train_loss),
        "val_nll": float(state.val_nll) if state.val_nll is not None else None,
        "metrics_test": metrics.to_dict(), "sample_c_d": sample_cd, "sample_alt_texts": sample_alts,
        "sample_prompt": getattr(cap, "messages", None),
        "head_alignment": align, "head_alignment_lambda": float(rcfg.head_alignment),
    }
    print(f"head alignment: lambda={rcfg.head_alignment} slot_argmax_agreement="
          f"{align['slot_argmax_agreement']} per_head_top_slot={align['per_head_top_slot']}")
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"smoke_real_slates[{args.adapter}] OK  n_train={len(batch_train)} J={J} p={p_eff} "
          f"top1={metrics.top1:.4f}  out={out_dir / 'smoke_summary.json'}")
    print("--- sample prompt (system) ---"); print(cap.messages[0]["content"] if getattr(cap, "messages", None) else "")
    print("--- sample prompt (user) ---"); print(cap.messages[1]["content"] if getattr(cap, "messages", None) else "")
    out_cache.close(); emb_cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(_build_arg_parser().parse_args()))
