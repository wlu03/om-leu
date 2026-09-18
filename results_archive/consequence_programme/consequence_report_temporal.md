# Consequence modelling: results under the temporal protocol

Every number is recomputed from saved per-event predictions.  ΔNLL is `NLL(baseline) - NLL(variant)`, so **positive means the variant is better**.  Intervals are 2,000-draw bootstraps over independent clusters (household on LPMC, respondent elsewhere), pooled over seeds and conditional on the trained models and the observed split; they do not include split or retraining uncertainty.  `*` marks an interval excluding zero.

Runs: 48 completed, 16 variants, ['lpmc'], seeds [7, 11, 13], 3 development folds, 5 members.


## 1. Ensemble contribution: does *another predictor* help?

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `numeric_auxiliary` vs `numeric_only` | lpmc | 0.4896 | 0.5356 | +0.0454* [+0.0262, +0.0636] | 0.85 |

*What this contrast means:* a second, non-semantic predictor mixed through the same calibration.


## 2. The shipped model against its own numeric channel

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `preserved_llm` vs `numeric_only` | lpmc | 0.5115 | 0.5356 | +0.0240* [+0.0019, +0.0453] | 0.70 |

*What this contrast means:* the shipped full model against its own numeric channel.


## 3. Semantic contribution: are real consequences doing the work?

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `preserved_llm` vs `numeric_auxiliary` | lpmc | 0.5115 | 0.4896 | -0.0214 [-0.0442, +0.0003] | 0.70 |
| `preserved_llm` vs `preserved_shuffled` | lpmc | 0.5115 | 0.5010 | -0.0103 [-0.0290, +0.0073] | 0.70 |
| `preserved_llm` vs `preserved_identity` | lpmc | 0.5115 | 0.5066 | -0.0046 [-0.0239, +0.0153] | 0.70 |
| `preserved_llm` vs `preserved_random` | lpmc | 0.5115 | 0.5166 | +0.0052 [-0.0158, +0.0254] | 0.70 |
| `preserved_llm` vs `preserved_template` | lpmc | 0.5115 | 0.5008 | -0.0103 [-0.0324, +0.0094] | 0.70 |
| `axis_template` vs `axis_llm` | lpmc | 0.4777 | 0.5168 | +0.0389* [+0.0194, +0.0591] | 0.87 |

*What this contrast means:* consequences against a matched non-semantic channel with the same mixture.


## 4. Structural contribution: is the proposed architecture the reason?

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `preserved_llm` vs `plain_slots_llm` | lpmc | 0.5115 | 0.5436 | +0.0322* [+0.0121, +0.0533] | 0.70 |
| `axis_llm` vs `plain_slots_llm` | lpmc | 0.5168 | 0.5436 | +0.0266* [+0.0126, +0.0410] | 0.77 |
| `axis_llm` vs `preserved_llm` | lpmc | 0.5168 | 0.5115 | -0.0056 [-0.0225, +0.0119] | 0.77 |

*What this contrast means:* the shipped reader against an ordinary network with identical information.


## Confirmatory family (prespecified, Holm-ordered)

| contrast | dataset | ΔNLL [95% CI] | Holm rank | adjusted level |
|---|---|---|---|---|
| `preserved_llm` vs `plain_slots_llm` | lpmc | +0.0322 [+0.0121, +0.0533] | 1 | 0.0100 |
| `axis_llm` vs `plain_slots_llm` | lpmc | +0.0266 [+0.0126, +0.0410] | 2 | 0.0125 |
| `preserved_llm` vs `numeric_auxiliary` | lpmc | -0.0214 [-0.0442, +0.0003] | 3 | 0.0167 |
| `preserved_llm` vs `preserved_template` | lpmc | -0.0103 [-0.0324, +0.0094] | 4 | 0.0250 |
| `preserved_llm` vs `preserved_shuffled` | lpmc | -0.0103 [-0.0290, +0.0073] | 5 | 0.0500 |

Every other contrast in this report is exploratory.  A bootstrap interval that contains zero is not evidence of equivalence.


## Calibration

| dataset | variant | π | temperature | development NLL | π=0 | π=1 | grid − L-BFGS |
|---|---|---|---|---|---|---|---|
| lpmc | `axis_llm` | 1.000 | 4.88 | 0.6112 | 1.0110 | 0.6112 | -0.03621 |
| lpmc | `axis_llm_anchored` | 1.000 | 4.94 | 0.6123 | 1.0147 | 0.6123 | -0.03609 |
| lpmc | `axis_template` | 1.000 | 14.90 | 0.5528 | 1.2372 | 0.5528 | -0.00112 |
| lpmc | `numeric_auxiliary` | 0.904 | 1.91 | 0.5608 | 0.7700 | 0.5722 | +0.00001 |
| lpmc | `numeric_only` | 0.002 | 1.35 | 0.7375 | nan | nan | +nan |
| lpmc | `numeric_stage1_only` | 0.002 | 2.05 | 0.8299 | nan | nan | +nan |
| lpmc | `plain_all_inputs_llm` | 0.888 | 2.27 | 0.5405 | 0.8052 | 0.5632 | -0.00115 |
| lpmc | `plain_mean_llm` | 0.731 | 2.00 | 0.6348 | 0.7787 | 0.6679 | -0.03036 |
| lpmc | `plain_slots_llm` | 1.000 | 6.40 | 0.6311 | 1.0801 | 0.6311 | -0.05982 |
| lpmc | `plain_slots_template` | 0.680 | 0.67 | 0.5982 | 0.9118 | 0.6331 | +0.00010 |
| lpmc | `plain_slots_z_llm` | 1.000 | 5.95 | 0.6000 | 1.0623 | 0.6000 | -0.03803 |
| lpmc | `preserved_identity` | 0.463 | 0.88 | 0.6668 | 0.7998 | 0.7503 | +0.00006 |
| lpmc | `preserved_llm` | 0.641 | 0.91 | 0.5992 | 0.7903 | 0.6448 | +0.00002 |
| lpmc | `preserved_random` | 0.025 | 1.24 | 0.7337 | 0.7396 | 1.3883 | +0.00133 |
| lpmc | `preserved_shuffled` | 0.244 | 1.02 | 0.7039 | 0.7638 | 0.9025 | +0.00002 |
| lpmc | `preserved_template` | 1.000 | 4.15 | 0.5571 | 0.9666 | 0.5571 | -0.00199 |

The development NLL is the objective the calibration minimised; it is a calibration input and is never an evaluation of the fitted mixture.


## Failed, blocked and not-run ledger

| status | dataset | protocol | variant | seed | reason |
|---|---|---|---|---|---|
| blocked_data | lpmc | person_disjoint | transfer_from_lpmc | 7 | transfer variant is unsupported here: source and target are both lpmc |
| blocked_data | lpmc | person_disjoint | transfer_from_lpmc_finetune | 7 | transfer variant is unsupported here: source and target are both lpmc |
| blocked_data | optima | person_disjoint | transfer_from_optima | 7 | transfer variant is unsupported here: source and target are both optima |
| blocked_data | optima | person_disjoint | transfer_from_optima_finetune | 7 | transfer variant is unsupported here: source and target are both optima |
| blocked_data | swissmetro | person_disjoint | transfer_from_swissmetro | 7 | transfer variant is unsupported here: source and target are both swissmetro |
| blocked_data | swissmetro | person_disjoint | transfer_from_swissmetro_finetune | 7 | transfer variant is unsupported here: source and target are both swissmetro |