# Consequence modelling: results under the person_disjoint protocol

Every number is recomputed from saved per-event predictions.  ΔNLL is `NLL(baseline) - NLL(variant)`, so **positive means the variant is better**.  Intervals are 2,000-draw bootstraps over independent clusters (household on LPMC, respondent elsewhere), pooled over seeds and conditional on the trained models and the observed split; they do not include split or retraining uncertainty.  `*` marks an interval excluding zero.

Runs: 193 completed, 55 variants, ['lpmc', 'optima', 'swissmetro'], seeds [7, 11, 13], 3 development folds, 5 members.


## 1. Ensemble contribution: does *another predictor* help?

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `numeric_auxiliary` vs `numeric_only` | lpmc | 0.5116 | 0.5225 | +0.0110* [+0.0038, +0.0185] | 0.37 |
| `numeric_auxiliary` vs `numeric_only` | optima | 0.4301 | 0.4490 | +0.0189* [+0.0059, +0.0320] | 0.52 |
| `numeric_auxiliary` vs `numeric_only` | swissmetro | 0.5975 | 0.5955 | -0.0021 [-0.0058, +0.0013] | 0.38 |

*What this contrast means:* a second, non-semantic predictor mixed through the same calibration.


## 2. The shipped model against its own numeric channel

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `preserved_llm` vs `numeric_only` | lpmc | 0.4902 | 0.5225 | +0.0323* [+0.0206, +0.0450] | 0.25 |
| `preserved_llm` vs `numeric_only` | optima | 0.4422 | 0.4490 | +0.0066 [-0.0065, +0.0207] | 0.40 |
| `preserved_llm` vs `numeric_only` | swissmetro | 0.5868 | 0.5955 | +0.0087* [+0.0014, +0.0178] | 0.21 |

*What this contrast means:* the shipped full model against its own numeric channel.


## 3. Semantic contribution: are real consequences doing the work?

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `preserved_llm` vs `numeric_auxiliary` | lpmc | 0.4902 | 0.5116 | +0.0213* [+0.0104, +0.0328] | 0.25 |
| `preserved_llm` vs `numeric_auxiliary` | optima | 0.4422 | 0.4301 | -0.0123 [-0.0244, +0.0007] | 0.40 |
| `preserved_llm` vs `numeric_auxiliary` | swissmetro | 0.5868 | 0.5975 | +0.0107* [+0.0019, +0.0226] | 0.21 |
| `preserved_llm` vs `preserved_shuffled` | lpmc | 0.4902 | 0.5224 | +0.0322* [+0.0205, +0.0450] | 0.25 |
| `preserved_llm` vs `preserved_shuffled` | optima | 0.4422 | 0.4445 | +0.0022 [-0.0079, +0.0122] | 0.40 |
| `preserved_llm` vs `preserved_shuffled` | swissmetro | 0.5868 | 0.5931 | +0.0063* [+0.0009, +0.0125] | 0.21 |
| `preserved_llm` vs `preserved_identity` | lpmc | 0.4902 | 0.5204 | +0.0302* [+0.0192, +0.0422] | 0.25 |
| `preserved_llm` vs `preserved_identity` | optima | 0.4422 | 0.4426 | +0.0002 [-0.0101, +0.0119] | 0.40 |
| `preserved_llm` vs `preserved_identity` | swissmetro | 0.5868 | 0.5932 | +0.0064* [+0.0010, +0.0127] | 0.21 |
| `preserved_llm` vs `preserved_random` | lpmc | 0.4902 | 0.5224 | +0.0321* [+0.0205, +0.0450] | 0.25 |
| `preserved_llm` vs `preserved_random` | optima | 0.4422 | 0.4450 | +0.0026 [-0.0093, +0.0151] | 0.40 |
| `preserved_llm` vs `preserved_random` | swissmetro | 0.5868 | 0.5937 | +0.0069* [+0.0009, +0.0144] | 0.21 |
| `preserved_llm` vs `preserved_template` | lpmc | 0.4902 | 0.5090 | +0.0188* [+0.0093, +0.0290] | 0.25 |
| `preserved_llm` vs `preserved_template` | optima | 0.4422 | 0.4388 | -0.0035 [-0.0143, +0.0087] | 0.40 |
| `preserved_llm` vs `preserved_template` | swissmetro | 0.5868 | 0.5897 | +0.0029 [-0.0007, +0.0065] | 0.21 |
| `axis_template` vs `axis_llm` | lpmc | 0.5064 | 0.4783 | -0.0279* [-0.0417, -0.0156] | 0.32 |
| `axis_template` vs `axis_llm` | optima | 0.4359 | 0.4497 | +0.0138 [-0.0003, +0.0262] | 0.40 |
| `axis_template` vs `axis_llm` | swissmetro | 0.5812 | 0.5804 | -0.0008 [-0.0058, +0.0043] | 0.37 |
| `gate_conditional_llm` vs `gate_conditional_numeric_aux` | lpmc | 0.4408 | 0.4498 | +0.0090 [-0.0068, +0.0242] | 0.29 |

*What this contrast means:* consequences against a matched non-semantic channel with the same mixture.


## 4. Structural contribution: is the proposed architecture the reason?

| contrast | dataset | NLL variant | NLL baseline | ΔNLL [95% CI] | π |
|---|---|---|---|---|---|
| `preserved_llm` vs `plain_slots_llm` | lpmc | 0.4902 | 0.4784 | -0.0119* [-0.0208, -0.0040] | 0.25 |
| `preserved_llm` vs `plain_slots_llm` | optima | 0.4422 | 0.4520 | +0.0098 [-0.0011, +0.0215] | 0.40 |
| `preserved_llm` vs `plain_slots_llm` | swissmetro | 0.5868 | 0.5812 | -0.0056* [-0.0090, -0.0024] | 0.21 |
| `axis_llm` vs `plain_slots_llm` | lpmc | 0.4783 | 0.4784 | -0.0000 [-0.0051, +0.0048] | 0.28 |
| `axis_llm` vs `plain_slots_llm` | optima | 0.4497 | 0.4520 | +0.0024 [-0.0044, +0.0091] | 0.33 |
| `axis_llm` vs `plain_slots_llm` | swissmetro | 0.5804 | 0.5812 | +0.0008 [-0.0018, +0.0031] | 0.26 |
| `axis_llm` vs `preserved_llm` | lpmc | 0.4783 | 0.4902 | +0.0119* [+0.0050, +0.0194] | 0.28 |
| `axis_llm` vs `preserved_llm` | optima | 0.4497 | 0.4422 | -0.0074 [-0.0162, +0.0013] | 0.33 |
| `axis_llm` vs `preserved_llm` | swissmetro | 0.5804 | 0.5868 | +0.0064* [+0.0038, +0.0090] | 0.26 |
| `mixture_aware_preserved` vs `preserved_llm` | lpmc | 0.4416 | 0.4404 | -0.0013 [-0.0056, +0.0027] | 0.29 |
| `mixture_aware_plain` vs `plain_slots_llm` | lpmc | 0.4214 | 0.4208 | -0.0007 [-0.0047, +0.0032] | 0.30 |
| `gate_conditional_llm` vs `gate_global_llm` | lpmc | 0.4408 | 0.4404 | -0.0004 [-0.0083, +0.0066] | 0.29 |

*What this contrast means:* the shipped reader against an ordinary network with identical information.


## Confirmatory family (prespecified, Holm-ordered)

| contrast | dataset | ΔNLL [95% CI] | Holm rank | adjusted level |
|---|---|---|---|---|
| `preserved_llm` vs `preserved_shuffled` | lpmc | +0.0322 [+0.0205, +0.0450] | 1 | 0.0033 |
| `preserved_llm` vs `numeric_auxiliary` | lpmc | +0.0213 [+0.0104, +0.0328] | 2 | 0.0036 |
| `preserved_llm` vs `preserved_template` | lpmc | +0.0188 [+0.0093, +0.0290] | 3 | 0.0038 |
| `preserved_llm` vs `numeric_auxiliary` | optima | -0.0123 [-0.0244, +0.0007] | 4 | 0.0042 |
| `preserved_llm` vs `plain_slots_llm` | lpmc | -0.0119 [-0.0208, -0.0040] | 5 | 0.0045 |
| `preserved_llm` vs `numeric_auxiliary` | swissmetro | +0.0107 [+0.0019, +0.0226] | 6 | 0.0050 |
| `preserved_llm` vs `plain_slots_llm` | optima | +0.0098 [-0.0011, +0.0215] | 7 | 0.0056 |
| `preserved_llm` vs `preserved_shuffled` | swissmetro | +0.0063 [+0.0009, +0.0125] | 8 | 0.0063 |
| `preserved_llm` vs `plain_slots_llm` | swissmetro | -0.0056 [-0.0090, -0.0024] | 9 | 0.0071 |
| `preserved_llm` vs `preserved_template` | optima | -0.0035 [-0.0143, +0.0087] | 10 | 0.0083 |
| `preserved_llm` vs `preserved_template` | swissmetro | +0.0029 [-0.0007, +0.0065] | 11 | 0.0100 |
| `axis_llm` vs `plain_slots_llm` | optima | +0.0024 [-0.0044, +0.0091] | 12 | 0.0125 |
| `preserved_llm` vs `preserved_shuffled` | optima | +0.0022 [-0.0079, +0.0122] | 13 | 0.0167 |
| `axis_llm` vs `plain_slots_llm` | swissmetro | +0.0008 [-0.0018, +0.0031] | 14 | 0.0250 |
| `axis_llm` vs `plain_slots_llm` | lpmc | -0.0000 [-0.0051, +0.0048] | 15 | 0.0500 |

Every other contrast in this report is exploratory.  A bootstrap interval that contains zero is not evidence of equivalence.


## Calibration

| dataset | variant | π | temperature | development NLL | π=0 | π=1 | grid − L-BFGS |
|---|---|---|---|---|---|---|---|
| lpmc | `axis_identity` | 0.017 | 0.96 | 0.5347 | 0.5349 | 0.7497 | +0.00024 |
| lpmc | `axis_llm` | 0.304 | 0.78 | 0.5079 | 0.5497 | 0.6147 | +0.00016 |
| lpmc | `axis_llm_anchored` | 0.305 | 0.78 | 0.5072 | 0.5500 | 0.6149 | +0.00017 |
| lpmc | `axis_llm_fixed_s` | 0.305 | 0.78 | 0.5077 | 0.5505 | 0.6141 | +0.00020 |
| lpmc | `axis_llm_no_history` | 0.305 | 0.79 | 0.5075 | 0.5491 | 0.6139 | +0.00012 |
| lpmc | `axis_llm_person_s` | 0.320 | 0.77 | 0.5047 | 0.5516 | 0.6141 | +0.00023 |
| lpmc | `axis_llm_shared_head` | 0.272 | 0.80 | 0.5106 | 0.5473 | 0.6397 | +0.00012 |
| lpmc | `axis_llm_uniform_w` | 0.305 | 0.77 | 0.5079 | 0.5525 | 0.6163 | +0.00016 |
| lpmc | `axis_shuffled` | 0.021 | 0.94 | 0.5343 | 0.5352 | 0.9230 | +0.00068 |
| lpmc | `axis_template` | 0.374 | 0.88 | 0.5221 | 0.5386 | 0.5533 | +0.00012 |
| lpmc | `combined_axis_llm_mixture_aware` | 0.309 | 0.81 | 0.5061 | 0.5454 | 0.6178 | +0.00002 |
| lpmc | `combined_axis_template_mixture_aware` | 0.364 | 0.91 | 0.5224 | 0.5365 | 0.5550 | +0.00002 |
| lpmc | `e3_consistency_both` | 0.327 | 0.89 | 0.5248 | 0.5379 | 0.5607 | +0.00009 |
| lpmc | `e3_consistency_edit` | 0.361 | 0.87 | 0.5227 | 0.5389 | 0.5552 | +0.00014 |
| lpmc | `e3_consistency_none` | 0.374 | 0.88 | 0.5221 | 0.5386 | 0.5533 | +0.00012 |
| lpmc | `e3_consistency_para` | 0.349 | 0.88 | 0.5238 | 0.5379 | 0.5577 | +0.00005 |
| lpmc | `gate_conditional_llm` | 0.294 | 0.76 | 0.5073 | 0.5546 | 0.6217 | +0.00004 |
| lpmc | `gate_conditional_llm_weak_shrinkage` | 0.294 | 0.76 | 0.5073 | 0.5546 | 0.6217 | +0.00004 |
| lpmc | `gate_conditional_numeric_aux` | 0.307 | 0.95 | 0.5280 | 0.5350 | 0.5624 | +0.00033 |
| lpmc | `gate_conditional_template` | 0.276 | 0.84 | 0.5261 | 0.5418 | 0.5764 | +0.00007 |
| lpmc | `gate_global_llm` | 0.294 | 0.76 | 0.5073 | 0.5546 | 0.6217 | +0.00004 |
| lpmc | `lc_axis_llm_010` | 0.788 | 2.63 | 0.7510 | 1.5463 | 0.7857 | +0.00009 |
| lpmc | `lc_axis_llm_025` | 0.855 | 2.11 | 0.6904 | 1.8287 | 0.7071 | +0.00002 |
| lpmc | `lc_axis_llm_050` | 1.000 | 469.44 | 0.6880 | 1.3709 | 0.6880 | -0.01276 |
| lpmc | `lc_axis_llm_100` | 0.933 | 1.73 | 0.6118 | 2.1581 | 0.6179 | +0.00027 |
| lpmc | `lc_axis_template_010` | 0.818 | 2.12 | 0.7375 | 1.8213 | 0.7622 | +0.00015 |
| lpmc | `lc_axis_template_025` | 0.916 | 1.26 | 0.6371 | 2.8539 | 0.6445 | +0.00017 |
| lpmc | `lc_axis_template_050` | 1.000 | 0.01 | 0.6077 | 11.3317 | 0.6077 | -0.00151 |
| lpmc | `lc_axis_template_100` | 1.000 | 0.01 | 0.5557 | 11.4063 | 0.5557 | -0.00000 |
| lpmc | `lc_preserved_llm_010` | 0.698 | 2.41 | 0.8092 | 1.6473 | 0.8629 | +0.00001 |
| lpmc | `lc_preserved_llm_025` | 0.816 | 2.19 | 0.7210 | 1.7737 | 0.7451 | +0.00013 |
| lpmc | `lc_preserved_llm_050` | 0.883 | 2.12 | 0.6790 | 1.8244 | 0.6902 | +0.00017 |
| lpmc | `lc_preserved_llm_100` | 0.933 | 1.63 | 0.6165 | 2.2758 | 0.6216 | +0.00025 |
| lpmc | `mixture_aware_plain` | 0.303 | 0.83 | 0.5050 | 0.5427 | 0.6382 | +0.00004 |
| lpmc | `mixture_aware_preserved` | 0.295 | 0.80 | 0.5070 | 0.5472 | 0.6267 | +0.00007 |
| lpmc | `mixture_aware_template` | 0.300 | 0.86 | 0.5252 | 0.5401 | 0.5685 | +0.00023 |
| lpmc | `numeric_auxiliary` | 0.307 | 0.95 | 0.5280 | 0.5350 | 0.5624 | +0.00033 |
| lpmc | `numeric_only` | 0.002 | 0.98 | 0.5348 | nan | nan | +nan |
| lpmc | `numeric_stage1_only` | 0.002 | 1.19 | 0.5630 | nan | nan | +nan |
| lpmc | `plain_all_inputs_llm` | 0.511 | 0.85 | 0.4922 | 0.5413 | 0.5322 | +0.00009 |
| lpmc | `plain_mean_llm` | 0.242 | 0.81 | 0.5129 | 0.5456 | 0.6589 | +0.00004 |
| lpmc | `plain_slots_llm` | 0.308 | 0.79 | 0.5052 | 0.5484 | 0.6234 | +0.00010 |
| lpmc | `plain_slots_template` | 0.217 | 0.85 | 0.5252 | 0.5412 | 0.6231 | +0.00020 |
| lpmc | `plain_slots_z_llm` | 0.388 | 0.79 | 0.4975 | 0.5490 | 0.5757 | +0.00016 |
| lpmc | `preserved_identity` | 0.002 | 0.98 | 0.5348 | 0.5348 | 0.7570 | +0.00017 |
| lpmc | `preserved_llm` | 0.294 | 0.76 | 0.5073 | 0.5546 | 0.6217 | +0.00004 |
| lpmc | `preserved_random` | 0.006 | 0.95 | 0.5346 | 0.5350 | 1.3877 | +0.00042 |
| lpmc | `preserved_shuffled` | 0.014 | 0.96 | 0.5346 | 0.5350 | 0.8940 | +0.00034 |
| lpmc | `preserved_template` | 0.276 | 0.84 | 0.5261 | 0.5418 | 0.5764 | +0.00007 |
| lpmc | `transfer_from_optima` | 0.057 | 0.90 | 0.5318 | 0.5369 | 0.8723 | +0.00004 |
| lpmc | `transfer_from_optima_finetune` | 0.313 | 0.80 | 0.5056 | 0.5477 | 0.6182 | +0.00008 |
| lpmc | `transfer_from_swissmetro` | 0.008 | 0.96 | 0.5345 | 0.5350 | 1.1751 | +0.00048 |
| lpmc | `transfer_from_swissmetro_finetune` | 0.294 | 0.79 | 0.5078 | 0.5494 | 0.6282 | +0.00019 |
| optima | `axis_llm` | 0.275 | 0.77 | 0.4600 | 0.5080 | 0.5655 | +0.00013 |
| optima | `axis_llm_anchored` | 0.283 | 0.77 | 0.4589 | 0.5084 | 0.5640 | +0.00008 |
| optima | `axis_template` | 0.338 | 0.85 | 0.4620 | 0.4950 | 0.5169 | +0.00008 |
| optima | `e3_consistency_both` | 0.338 | 0.86 | 0.4625 | 0.4938 | 0.5158 | +0.00012 |
| optima | `e3_consistency_edit` | 0.344 | 0.85 | 0.4614 | 0.4952 | 0.5134 | +0.00007 |
| optima | `e3_consistency_none` | 0.338 | 0.85 | 0.4620 | 0.4950 | 0.5169 | +0.00008 |
| optima | `e3_consistency_para` | 0.335 | 0.86 | 0.4625 | 0.4941 | 0.5183 | +0.00012 |
| optima | `numeric_auxiliary` | 0.474 | 0.95 | 0.4547 | 0.4859 | 0.4889 | +0.00016 |
| optima | `numeric_only` | 0.002 | 1.08 | 0.4820 | nan | nan | +nan |
| optima | `numeric_stage1_only` | 0.002 | 1.56 | 0.5162 | nan | nan | +nan |
| optima | `plain_all_inputs_llm` | 0.405 | 0.90 | 0.4555 | 0.4898 | 0.5027 | +0.00001 |
| optima | `plain_mean_llm` | 0.225 | 0.78 | 0.4647 | 0.5073 | 0.6118 | +0.00016 |
| optima | `plain_slots_llm` | 0.230 | 0.81 | 0.4639 | 0.5016 | 0.5974 | +0.00016 |
| optima | `plain_slots_template` | 0.206 | 0.79 | 0.4645 | 0.5041 | 0.6031 | +0.00006 |
| optima | `plain_slots_z_llm` | 0.297 | 0.88 | 0.4644 | 0.4911 | 0.5411 | +0.00004 |
| optima | `preserved_identity` | 0.244 | 0.91 | 0.4696 | 0.4889 | 0.5605 | +0.00001 |
| optima | `preserved_llm` | 0.334 | 0.81 | 0.4566 | 0.5009 | 0.5264 | +0.00006 |
| optima | `preserved_random` | 0.040 | 0.88 | 0.4746 | 0.4919 | 1.1020 | +0.00039 |
| optima | `preserved_shuffled` | 0.190 | 0.87 | 0.4696 | 0.4921 | 0.6023 | +0.00018 |
| optima | `preserved_template` | 0.295 | 0.88 | 0.4650 | 0.4910 | 0.5381 | +0.00005 |
| optima | `transfer_from_lpmc` | 0.201 | 0.77 | 0.4589 | 0.5085 | 0.6474 | +0.00011 |
| optima | `transfer_from_lpmc_finetune` | 0.279 | 0.76 | 0.4577 | 0.5110 | 0.5750 | +0.00009 |
| optima | `transfer_from_swissmetro` | 0.143 | 0.83 | 0.4653 | 0.4982 | 0.6781 | +0.00002 |
| optima | `transfer_from_swissmetro_finetune` | 0.300 | 0.77 | 0.4555 | 0.5092 | 0.5590 | +0.00006 |
| swissmetro | `axis_llm` | 0.287 | 0.83 | 0.5838 | 0.6189 | 0.6745 | +0.00004 |
| swissmetro | `axis_llm_anchored` | 0.284 | 0.84 | 0.5837 | 0.6183 | 0.6763 | +0.00006 |
| swissmetro | `axis_template` | 0.413 | 0.87 | 0.5849 | 0.6141 | 0.6051 | +0.00007 |
| swissmetro | `numeric_auxiliary` | 0.394 | 0.99 | 0.5957 | 0.6065 | 0.6100 | +0.00000 |
| swissmetro | `numeric_only` | 0.002 | 1.06 | 0.6055 | nan | nan | +nan |
| swissmetro | `numeric_stage1_only` | 0.002 | 1.12 | 0.6482 | nan | nan | +nan |
| swissmetro | `plain_all_inputs_llm` | 0.494 | 0.95 | 0.5868 | 0.6083 | 0.6028 | +0.00009 |
| swissmetro | `plain_mean_llm` | 0.230 | 0.83 | 0.5874 | 0.6197 | 0.6978 | +0.00008 |
| swissmetro | `plain_slots_llm` | 0.280 | 0.83 | 0.5828 | 0.6193 | 0.6784 | +0.00007 |
| swissmetro | `plain_slots_template` | 0.271 | 0.86 | 0.5885 | 0.6156 | 0.6392 | +0.00017 |
| swissmetro | `plain_slots_z_llm` | 0.268 | 0.86 | 0.5871 | 0.6152 | 0.6747 | +0.00017 |
| swissmetro | `preserved_identity` | 0.054 | 0.96 | 0.6014 | 0.6077 | 0.8380 | +0.00021 |
| swissmetro | `preserved_llm` | 0.244 | 0.85 | 0.5889 | 0.6173 | 0.6847 | +0.00008 |
| swissmetro | `preserved_random` | 0.042 | 0.90 | 0.5995 | 0.6113 | 1.0993 | +0.00016 |
| swissmetro | `preserved_shuffled` | 0.058 | 0.95 | 0.6010 | 0.6085 | 0.8677 | +0.00036 |
| swissmetro | `preserved_template` | 0.241 | 0.90 | 0.5946 | 0.6119 | 0.6410 | +0.00002 |
| swissmetro | `transfer_from_lpmc` | 0.061 | 0.90 | 0.5990 | 0.6114 | 0.9662 | +0.00016 |
| swissmetro | `transfer_from_lpmc_finetune` | 0.286 | 0.84 | 0.5845 | 0.6183 | 0.6757 | +0.00005 |
| swissmetro | `transfer_from_optima` | 0.074 | 0.89 | 0.5980 | 0.6122 | 0.8946 | +0.00051 |
| swissmetro | `transfer_from_optima_finetune` | 0.286 | 0.84 | 0.5840 | 0.6186 | 0.6749 | +0.00004 |

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
| blocked_generation_budget | swissmetro | person_disjoint | e1_grounded | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_generation_budget | swissmetro | person_disjoint | e1_grounded_no_person | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_generation_budget | swissmetro | person_disjoint | e1_existing_prompt_replication | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_annotations | swissmetro | n/a | e1_human_audit | 7 | label-blind export written to e1_audit/; no annotator has scored it |
| blocked_data | swissmetro | person_disjoint | e5_risk_sensitive | 7 | stated-preference attributes are fixed scenario values with no distribution; would be enabled by a travel-time or delay distribution per alternative ( |
| blocked_generation_budget | optima | person_disjoint | e1_grounded | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_generation_budget | optima | person_disjoint | e1_grounded_no_person | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_generation_budget | optima | person_disjoint | e1_existing_prompt_replication | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_annotations | optima | n/a | e1_human_audit | 7 | label-blind export written to e1_audit/; no annotator has scored it |
| blocked_data | optima | person_disjoint | e5_risk_sensitive | 7 | reported trip attributes are single values; the survey has attitudes, not beliefs about delay; would be enabled by a travel-time or delay distribution |
| blocked_generation_budget | lpmc | person_disjoint | e1_grounded | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_generation_budget | lpmc | person_disjoint | e1_grounded_no_person | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_generation_budget | lpmc | person_disjoint | e1_existing_prompt_replication | 7 | no bounded generation budget configured; request counts in e1_generation/generation_manifest.json |
| blocked_annotations | lpmc | n/a | e1_human_audit | 7 | label-blind export written to e1_audit/; no annotator has scored it |
| blocked_data | lpmc | person_disjoint | e5_risk_sensitive | 7 | route-planner attributes are point estimates; no travel-time distribution is distributed with the data; would be enabled by a travel-time or delay dis |
| blocked_data | swissmetro | temporal | core_matrix | 7 | ordering field is a stated-preference task index, not calendar time |
| blocked_data | optima | temporal | core_matrix | 7 | order dates were manufactured by the preparation script (4 distinct values) |
| not_run | optima+swissmetro | person_disjoint | core_matrix_extended | 7 | extended matrix, E3, E4 and E7 suites were run on LPMC only; one training process at a time |
| not_run | all | person_disjoint | e6_learning_curve_seeds_11_13 | 11 | learning curves and transfers were run at seed 7 only |