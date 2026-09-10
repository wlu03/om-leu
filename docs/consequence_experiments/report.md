# Consequence-modelling experiments: results from saved predictions

Positive ΔNLL means the variant improves on the baseline. Intervals are 2,000-draw bootstraps over independent clusters (household where the identifier encodes one, otherwise respondent), conditional on the trained models and the observed split. `*` marks an interval excluding zero.


## person_disjoint

| variant | lpmc: NLL | ΔNLL vs numeric_only [95% CI] | optima: NLL | ΔNLL vs numeric_only [95% CI] | swissmetro: NLL | ΔNLL vs numeric_only [95% CI] |
|---|---|---|---|---|---|---|
| `axis_identity` | 0.4624 | +0.0002 [-0.0009, +0.0014] | — | — | — | — |
| `axis_llm` | 0.4783 | +0.0441* [+0.0291, +0.0601] | 0.4497 | -0.0008 [-0.0162, +0.0146] | 0.5804 | +0.0151* [+0.0057, +0.0280] |
| `axis_llm_anchored` | 0.4782 | +0.0442* [+0.0289, +0.0611] | 0.4490 | -0.0001 [-0.0154, +0.0155] | 0.5812 | +0.0142* [+0.0053, +0.0268] |
| `axis_llm_fixed_s` | 0.4239 | +0.0388* [+0.0206, +0.0606] | — | — | — | — |
| `axis_llm_no_history` | 0.4297 | +0.0330* [+0.0157, +0.0531] | — | — | — | — |
| `axis_llm_person_s` | 0.4226 | +0.0400* [+0.0211, +0.0613] | — | — | — | — |
| `axis_llm_shared_head` | 0.4392 | +0.0235* [+0.0074, +0.0413] | — | — | — | — |
| `axis_llm_uniform_w` | 0.4220 | +0.0406* [+0.0212, +0.0631] | — | — | — | — |
| `axis_shuffled` | 0.4596 | +0.0031 [-0.0006, +0.0086] | — | — | — | — |
| `axis_template` | 0.5064 | +0.0162* [+0.0079, +0.0243] | 0.4359 | +0.0130 [-0.0002, +0.0272] | 0.5812 | +0.0142* [+0.0031, +0.0308] |
| `combined_axis_llm_mixture_aware` | 0.4251 | +0.0376* [+0.0188, +0.0595] | — | — | — | — |
| `combined_axis_template_mixture_aware` | 0.4481 | +0.0146* [+0.0022, +0.0271] | — | — | — | — |
| `e3_consistency_both` | 0.4479 | +0.0148* [+0.0037, +0.0256] | 0.4162 | +0.0173* [+0.0014, +0.0321] | — | — |
| `e3_consistency_edit` | 0.4444 | +0.0182* [+0.0062, +0.0305] | 0.4170 | +0.0165 [-0.0006, +0.0328] | — | — |
| `e3_consistency_none` | 0.4435 | +0.0192* [+0.0064, +0.0320] | 0.4135 | +0.0200* [+0.0048, +0.0347] | — | — |
| `e3_consistency_para` | 0.4462 | +0.0165* [+0.0046, +0.0286] | 0.4124 | +0.0210* [+0.0060, +0.0349] | — | — |
| `gate_conditional_llm` | 0.4408 | +0.0219* [+0.0054, +0.0404] | — | — | — | — |
| `gate_conditional_llm_weak_shrinkage` | 0.4410 | +0.0217* [+0.0053, +0.0400] | — | — | — | — |
| `gate_conditional_numeric_aux` | 0.4498 | +0.0129* [+0.0033, +0.0234] | — | — | — | — |
| `gate_conditional_template` | 0.4490 | +0.0137* [+0.0021, +0.0256] | — | — | — | — |
| `gate_global_llm` | 0.4404 | +0.0223* [+0.0050, +0.0421] | — | — | — | — |
| `lc_axis_llm_010` | 0.6275 | -0.1649* [-0.2262, -0.1019] | — | — | — | — |
| `lc_axis_llm_025` | 0.5767 | -0.1140* [-0.1697, -0.0569] | — | — | — | — |
| `lc_axis_llm_050` | 0.5067 | -0.0440 [-0.0938, +0.0071] | — | — | — | — |
| `lc_axis_llm_100` | 0.4749 | -0.0123 [-0.0570, +0.0339] | — | — | — | — |
| `lc_axis_template_010` | 0.5485 | -0.0859* [-0.1367, -0.0329] | — | — | — | — |
| `lc_axis_template_025` | 0.5213 | -0.0587* [-0.1037, -0.0120] | — | — | — | — |
| `lc_axis_template_050` | 0.4979 | -0.0352 [-0.0759, +0.0067] | — | — | — | — |
| `lc_axis_template_100` | 0.4407 | +0.0219 [-0.0076, +0.0502] | — | — | — | — |
| `lc_preserved_llm_010` | 0.6619 | -0.1992* [-0.2683, -0.1314] | — | — | — | — |
| `lc_preserved_llm_025` | 0.5837 | -0.1211* [-0.1799, -0.0608] | — | — | — | — |
| `lc_preserved_llm_050` | 0.5530 | -0.0903* [-0.1414, -0.0379] | — | — | — | — |
| `lc_preserved_llm_100` | 0.5097 | -0.0471* [-0.0946, -0.0035] | — | — | — | — |
| `mixture_aware_plain` | 0.4214 | +0.0412* [+0.0188, +0.0667] | — | — | — | — |
| `mixture_aware_preserved` | 0.4416 | +0.0210* [+0.0033, +0.0411] | — | — | — | — |
| `mixture_aware_template` | 0.4517 | +0.0110 [-0.0008, +0.0225] | — | — | — | — |
| `numeric_auxiliary` | 0.5116 | +0.0110* [+0.0042, +0.0183] | 0.4301 | +0.0189* [+0.0035, +0.0346] | 0.5975 | -0.0021 [-0.0068, +0.0019] |
| `numeric_only` | 0.5225 | +0.0000 [+0.0000, +0.0000] | 0.4490 | +0.0000 [+0.0000, +0.0000] | 0.5955 | +0.0000 [+0.0000, +0.0000] |
| `numeric_stage1_only` | 0.5434 | -0.0210* [-0.0335, -0.0098] | 0.4886 | -0.0395* [-0.0531, -0.0264] | 0.6405 | -0.0450* [-0.0607, -0.0331] |
| `plain_all_inputs_llm` | 0.4859 | +0.0366* [+0.0218, +0.0526] | 0.4446 | +0.0043 [-0.0130, +0.0215] | 0.5887 | +0.0068* [+0.0013, +0.0123] |
| `plain_mean_llm` | 0.4854 | +0.0370* [+0.0240, +0.0515] | 0.4509 | -0.0021 [-0.0173, +0.0137] | 0.5840 | +0.0114* [+0.0027, +0.0238] |
| `plain_slots_llm` | 0.4784 | +0.0441* [+0.0279, +0.0615] | 0.4520 | -0.0032 [-0.0197, +0.0140] | 0.5812 | +0.0143* [+0.0049, +0.0273] |
| `plain_slots_template` | 0.5101 | +0.0123* [+0.0035, +0.0217] | 0.4399 | +0.0090 [-0.0045, +0.0229] | 0.5868 | +0.0087 [-0.0013, +0.0232] |
| `plain_slots_z_llm` | 0.4754 | +0.0469* [+0.0308, +0.0641] | 0.4497 | -0.0008 [-0.0168, +0.0144] | 0.5817 | +0.0138* [+0.0045, +0.0272] |
| `preserved_identity` | 0.5204 | +0.0021* [+0.0003, +0.0044] | 0.4426 | +0.0064 [-0.0047, +0.0183] | 0.5932 | +0.0023 [-0.0009, +0.0074] |
| `preserved_llm` | 0.4902 | +0.0323* [+0.0202, +0.0459] | 0.4422 | +0.0066 [-0.0091, +0.0219] | 0.5868 | +0.0087* [+0.0004, +0.0205] |
| `preserved_random` | 0.5224 | +0.0001 [-0.0007, +0.0011] | 0.4450 | +0.0040 [-0.0028, +0.0123] | 0.5937 | +0.0017 [-0.0015, +0.0071] |
| `preserved_shuffled` | 0.5224 | +0.0001 [-0.0006, +0.0007] | 0.4445 | +0.0044 [-0.0070, +0.0155] | 0.5931 | +0.0024 [-0.0010, +0.0077] |
| `preserved_template` | 0.5090 | +0.0135* [+0.0052, +0.0226] | 0.4388 | +0.0101 [-0.0020, +0.0228] | 0.5897 | +0.0058 [-0.0019, +0.0177] |
| `transfer_from_lpmc` | — | — | 0.4396 | -0.0062 [-0.0218, +0.0085] | 0.5920 | -0.0039* [-0.0074, -0.0001] |
| `transfer_from_lpmc_finetune` | — | — | 0.4191 | +0.0143 [-0.0043, +0.0338] | 0.5854 | +0.0027 [-0.0057, +0.0111] |
| `transfer_from_optima` | 0.4567 | +0.0060 [-0.0011, +0.0157] | — | — | 0.5928 | -0.0047* [-0.0082, -0.0013] |
| `transfer_from_optima_finetune` | 0.4315 | +0.0312* [+0.0134, +0.0520] | — | — | 0.5844 | +0.0038 [-0.0045, +0.0125] |
| `transfer_from_swissmetro` | 0.4627 | -0.0000 [-0.0017, +0.0022] | 0.4265 | +0.0069 [-0.0060, +0.0216] | — | — |
| `transfer_from_swissmetro_finetune` | 0.4367 | +0.0260* [+0.0098, +0.0432] | 0.4128 | +0.0206* [+0.0005, +0.0438] | — | — |


## temporal

| variant | lpmc: NLL | ΔNLL vs numeric_only [95% CI] |
|---|---|---|
| `axis_llm` | 0.5168 | +0.0184 [-0.0064, +0.0432] |
| `axis_llm_anchored` | 0.5128 | +0.0224 [-0.0025, +0.0469] |
| `axis_template` | 0.4777 | +0.0573* [+0.0376, +0.0764] |
| `numeric_auxiliary` | 0.4896 | +0.0454* [+0.0236, +0.0661] |
| `numeric_only` | 0.5356 | +0.0000 [+0.0000, +0.0000] |
| `numeric_stage1_only` | 0.6116 | -0.0757* [-0.0845, -0.0668] |
| `plain_all_inputs_llm` | 0.4740 | +0.0611* [+0.0417, +0.0805] |
| `plain_mean_llm` | 0.5483 | -0.0128 [-0.0357, +0.0108] |
| `plain_slots_llm` | 0.5436 | -0.0082 [-0.0381, +0.0221] |
| `plain_slots_template` | 0.5242 | +0.0107 [-0.0106, +0.0306] |
| `plain_slots_z_llm` | 0.4938 | +0.0413* [+0.0180, +0.0652] |
| `preserved_identity` | 0.5066 | +0.0286* [+0.0155, +0.0413] |
| `preserved_llm` | 0.5115 | +0.0240* [+0.0008, +0.0459] |
| `preserved_random` | 0.5166 | +0.0188* [+0.0155, +0.0221] |
| `preserved_shuffled` | 0.5010 | +0.0342* [+0.0230, +0.0453] |
| `preserved_template` | 0.5008 | +0.0343* [+0.0131, +0.0552] |


## Failed, blocked and not-run ledger

| status | dataset | protocol | variant | seed | reason |
|---|---|---|---|---|---|
| blocked_data | lpmc | person_disjoint | transfer_from_lpmc | 7 | transfer variant is unsupported here: source and target are both lpmc |
| blocked_data | lpmc | person_disjoint | transfer_from_lpmc_finetune | 7 | transfer variant is unsupported here: source and target are both lpmc |
| blocked_data | optima | person_disjoint | transfer_from_optima | 7 | transfer variant is unsupported here: source and target are both optima |
| blocked_data | optima | person_disjoint | transfer_from_optima_finetune | 7 | transfer variant is unsupported here: source and target are both optima |
| blocked_data | swissmetro | person_disjoint | transfer_from_swissmetro | 7 | transfer variant is unsupported here: source and target are both swissmetro |
| blocked_data | swissmetro | person_disjoint | transfer_from_swissmetro_finetune | 7 | transfer variant is unsupported here: source and target are both swissmetro |