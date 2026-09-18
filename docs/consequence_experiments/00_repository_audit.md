# 00 Repository audit

Audited checkout: `/Users/wesleylu/Projects/Research/structure_descent` at `6a1774e` (branch
`goleu-redesign`, working tree clean at the time of the audit).  Worktree for this programme:
`/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/00-foundation` on branch `exp/consequence-20260910T094713Z-31092/00-foundation`.  Artifact root:
`/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts`.  Python: `/Users/wesleylu/Projects/Research/structure_descent/venv/bin/python`.

Every command below was executed during the audit.

## 1. Where the existing system lives

| stage | file | entry point |
|---|---|---|
| Stage 0 generation | `src/outcomes/generate.py`, prompts in `src/outcomes/prompts.py` | `generate_outcomes()`; mobility prompt `v7_modechoice_anchored`, K=5 |
| Stage 0 caches | `<dataset>/results/<run_tag>/seed_<s>/cache/{outcomes,embeddings}.sqlite` | `src/outcomes/cache.py` |
| Stage 0 encoder | `src/outcomes/encode.py` | `sentence-transformers/all-mpnet-base-v2`, mean pooling, max_length 64, L2-normalised, d=768 |
| Stage 1 structural | `experiments/models/hetero.py` | `build_hetero_v1`; used through `experiments/models/omleu2.py::structural_stage` |
| Stage 2 residual | `experiments/models/boost.py`, child process `experiments/models/_lgb_child.py` | `omleu2.boost_stage` |
| Stage 3 semantic | `experiments/models/pref.py::PrefBranch` (`V1`) | `omleu2.semantic_members` |
| Mixture | `experiments/models/omleu2.py::Omleu2.forward`, `stack_on_val`, `_fit_scalars_oof` | variants in `experiments/models/__init__.py` |
| Exported tensors | `experiments/data/<dataset>_seed<s>.npz` + `.json` | `experiments/export_tensors.py` |
| Earlier ablations | `ablation/` (this session's earlier work) | `ablation/run.py`, `ablation/make_tables.py` |

Reproduce the exported tensors (no LLM calls; refuses on a cache miss):

```bash
cd /Users/wesleylu/Projects/Research/structure_descent
venv/bin/python experiments/export_tensors.py --datasets optima lpmc swissmetro --seeds 7 11 13
```

## 2. Verification of the described architecture

Confirmed against the code: frozen LLM generations on five axes; frozen MPNet embeddings, d=768,
L2-normalised; Stage 1 with `-softplus` time and cost coefficients, functional intercepts
`g_j(z_i)`, L2-penalised person effects and five fits averaged in probability; Stage 2 with one
booster per alternative, monotone `-1` on time and cost, cross-fitted Stage-1 offsets and a
choice-set softmax loss; Stage 3 with a shared 768→32 projection, attention over the K sentences,
five linear heads, person-conditioned head weights, slot dropout 0.15, an auxiliary probe loss and
five members averaged in probability; final `P = (1-pi) softmax(a u) + pi q` with `a = 1/T`
calibrated after the base models are trained.

Discrepancies against the description supplied with the programme:

1. **Attention is person-agnostic in the shipped model.**  `experiments/models/pref.py::V1` sets
   `attn="salience"`: the attention scores are `v^T H` with no person query.  The person enters
   only through the head weights.  Person-conditioned attention exists (`V2`) but is not the
   preserved configuration.  Figures in `tmp/figures/` that draw a person arrow into the attention
   block are wrong for the shipped model.
2. **The auxiliary probe is a within-choice-set conditional logit on the mean projected sentence**
   (`PREF_LAM_PROBE = 0.5`), not a generic contrastive loss over a corpus.
3. **`pi` and `T` are fitted on one validation split in the shipped variants**
   (`Omleu2.stack_on_val`); an out-of-fold estimator was added later (`pi_fit="oof"`).

## 3. Dataset semantics

```bash
PYTHONPATH=/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/00-foundation venv/bin/python -m omleu_experiments.cli audit --datasets optima lpmc swissmetro
```

| dataset | event | respondent | household | alternatives | ordering field | ordering verdict |
|---|---|---|---|---|---|---|
| Swissmetro | one stated-preference task | `customer_id` = respondent (1,004 for 9,036 events) | not identified | train, sm, car | `order_date`, **9 distinct values**, 1998-01-01 + one week per task index | **task index, not calendar time**; a temporal protocol is unsupported |
| Optima | one revealed trip loop | `customer_id` = respondent (1,486 for 1,906 loops) | not identified | pt, car, soft | `order_date`, **4 distinct values** over three weeks | **manufactured by the preparation script**; unsupported |
| LPMC | one recorded trip | `h<household>_p<person>` (600 respondents for 3,556 trips in the exported subset) | **encoded in the identifier**: `h10044` | walk, cycle, pt, drive | `order_date`, 3,371 distinct timestamps 2012-04 to 2015-03 | genuine calendar time; temporal protocol supported |

Consequences: the bootstrap clusters on household for LPMC and on respondent elsewhere
(`omleu_experiments/runner.py::cluster_ids`); the time-respecting protocol runs on LPMC only, and
is recorded as `blocked_data` with the reason above for the other two.

**Availability.** No dataset carries an availability mask; every alternative is treated as available
in every event (`methods/common/data.py` has no `avail` field).  This is a modelling assumption
inherited from the source case studies, not a property of the data, and it is recorded in the
contracts (`Event.avail` all true) so that a future dataset with real masking is handled correctly.
Every event has exactly one chosen, available alternative; asserted in `Event.__post_init__`.

## 4. Admissible information

* **History features** (`is_repeat`, `log1p(purchase_count)`) are strictly-prior counts: for an
  event at date `d` the count is the number of that respondent's *earlier reference-frame*
  purchases of that alternative (`src/data/choice_sets.py`, "strictly-prior count" block, which
  replaced an earlier within-train future leak).  Admissible.
* **Recent-choice text in the prompt** uses a strict prefix window `[d - window, d)`
  (`src/data/choice_sets.py` "Per-event recent_purchases slice"), so the current and later events
  are excluded.  Admissible.
* **FINDING 1 — label-derived person aggregates reach the generation prompt.**  The person profile
  `c_d` is rendered from a per-respondent row that contains `purchase_frequency` (a count of that
  respondent's events) and `novelty_rate` (the mean of a per-event `novelty` indicator).  Both are
  `derived_from_events` aggregates over *all* of that respondent's events, including their
  validation and test events, and `novelty` is a function of the chosen alternative.  The rendered
  phrases are coarse ("nearly all of loops are first-time mode choices"), so the channel is weak,
  but it is label-derived and future-derived and must be reported as such.  Any grounded-generation
  variant in E1 removes these two fields from the allowlist; `contracts.FORBIDDEN_GENERATION_KEYS`
  rejects them programmatically.
* **FINDING 2 — covariate standardisation was fitted on the historical training split.**
  `methods/common/data.py` standardises `Z` with `splits["train"]` statistics, and the exported
  tensors inherit it.  Under a person-disjoint re-split those rows are no longer the training
  partition.  Standardisation is affine, so the foundation re-standardises the exported values
  inside each fitting partition (`omleu_experiments/views.py::restandardise_covariates`), which is
  exactly standardising the raw covariates with partition statistics.  Attribute standardisation
  (`Bundle.xnum_std`) already reads the view's own training mask and needed no change.

## 5. Cache identity

```bash
sed -n '180,215p' src/outcomes/cache.py     # outcomes key
sed -n '603,626p' src/outcomes/generate.py  # composite prompt version
```

The four-field key is `customer_id || asin || seed || prompt_version`, which alone would be keyed
only by person and alternative.  It is saved from that failure by
`build_cache_prompt_version(prompt_version, K, model_id, c_d, alt)`, which folds a hash of the
person text and of the rendered alternative (including its history fields) into the version string.
Cache identity therefore does include event content, alternative, generator revision and K.
Verified empirically: no two events of the same respondent share an embedding for the same
alternative (0 identical pairs out of 9,600 on Swissmetro, 1,260 on Optima, 7,856 on LPMC).

Not covered by the existing identity, and added in `contracts.Consequence.cache_identity`: the
decoding settings and an explicit history-policy hash.  Embedding identity is
`(text, encoder_id)`, which is sufficient.

## 6. Two report checks requested with the programme

1. **"only Stage 1 is correct" was wrong.**  For the worked Optima example (test row 1746, seed 7)
   the semantic channel's probabilities are soft 0.205, pt 0.232, car 0.563; its arg-max is the car,
   which is the chosen alternative.  The correct statement is that Stage 1 and the semantic channel
   both rank the car first, while Stage 2 and the mixture rank walking first.
2. **The temperature was applied exactly once.**  In `Omleu2.forward` the numeric part is
   `log_softmax(a * U)` and the mixture combines that with the semantic log-probabilities, so the
   0.275 quoted for "stages 1 and 2" in that example already includes the temperature and the
   mixture did not re-apply it.  The foundation enforces this structurally: callers pass raw
   logits and `calibrate.mixture_logprob` is the only place `a` is applied
   (`tests/test_foundation.py::test_temperature_applied_exactly_once`).

## 7. Claims in existing documents that this programme retracts or softens

* `docs/math/02_stage1_structural_utility.md` and the Figure-1 caption describe the negative
  `-softplus` coefficients as what "a valid random-utility model requires".  They are a modelling
  assumption about tastes; random utility itself imposes no sign.
* `docs/math/04_stage3_semantic_mixture.md` states that a utility-level sum "can only ever make the
  combined probabilities sharper".  That is false in general: adding a scaled semantic utility can
  reduce the spread of the combined utilities, and with a negative or opposing component it softens
  the probabilities.  The defensible statement is the one about the zero-initialised multiplicative
  gate receiving no gradient, plus the empirical result that the mixture form worked and the gate
  form did not.
* The five heads and the axis slots are named, not identified.  Head-slot agreement measured
  earlier was about 0.2, so neither the heads nor `pi` may be described as measured psychological
  quantities.
* Monotonicity: the booster constraints are per-alternative on that alternative's own time and cost
  columns.  Distance, access time, waiting time and any derived numeric path are unconstrained,
  Stage 1 contributes its own sign assumption separately, and the calibrated mixture with a second
  channel is not monotone in an attribute merely because one component is.  E7 states the extra
  `(q_j - p_j) d pi/d x` term explicitly.

## 8. Environment

macOS 26.4, Apple silicon, 18 cores, 48 GB.  LightGBM and PyTorch cannot share a process here, so
boosting runs in `experiments/models/_lgb_child.py`; this is a property of this machine's OpenMP
build, not a general restriction.  Local generation is possible without paid requests
(`ollama` with `llama3.3` and `qwen3-vl` installed) but the original consequences were produced by
`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` on Modal; no generation budget is configured, so new
paid generation is `blocked_generation_budget` and any local generation is recorded as a different
generator revision.

## 9. Do the reported numbers match the saved predictions?

Checked by recomputing each stored metric from the per-event predictions in the same artifact:

```bash
venv/bin/python - <<'PY'
import json, numpy as np
d = json.load(open("experiments/results/combo_full/optima_seed7.json"))
print(d["nll"], np.mean(d["per_event_nll"]), d["top1"], np.mean(d["per_event_top1"]))
PY
```

The stored negative log-likelihood and Top-1 agree with the mean of the saved per-event values to
about 6e-9, which is float32 rounding.  Six headline numbers quoted in the project's write-ups
(`pi_oof`, `combo_full`, `abl_cold_start_struct`, `pi_cold_oof` across the three datasets) were
checked against the artifacts for seeds 7, 11 and 13 and match to three decimal places, with the
protocol and calibration variant named in each case.  Two presentation rules follow and are applied
in this programme's report: a number is always printed with the protocol and the calibration variant
that produced it, and prose never carries a rounded value that is not also in a table generated from
the artifacts.
