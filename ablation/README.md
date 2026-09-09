# Ablations: LLM sentences vs the designed model

Every row is OM-LEU 2 with one change, three seeds per dataset. **Sentence-only NLL** is the sentence channel evaluated alone (five members averaged in probability, no structural model). **Mixture NLL** is the full system p = (1−π)·q + π·p̄ with that sentence channel. Δ is the paired per-event ΔNLL against `full_model` pooled over seeds with a 2000-draw bootstrap 95 % CI; `*` = CI excludes 0; negative = better than the reference. Plain-MLP baselines use exactly the same consequence sentences and embeddings as the full model.

**Interpretation: `ablation/FINDINGS.md`.** Code: `ablation/<name>/model.py`; per-run JSON: `ablation/<name>/results/`; maths: `ablation/<name>/breakdown.md`; runner: `ablation/run.py`; this file: `ablation/make_tables.py`.


## chronological within-person split — π fitted on the validation split

### Where the improvement comes from (mean test NLL over seeds)

| model | swissmetro | optima | lpmc |
|---|---|---|---|
| no LLM, no designed sentence model: structural utility + residual | 0.5583 | 0.4108 | 0.5434 |
| no LLM, no structure: plain MLP on numeric inputs | 0.5820 | 0.4425 | 0.5892 |
| LLM sentences, no structure: plain MLP on the sentence embedding | 0.7379 | 0.5498 | 0.6455 |
| LLM sentences + z_i, no structure: plain MLP | 0.7253 | 0.4608 | 0.5903 |
| all inputs, no structure: plain MLP on [sentences, z_i, x_ij, h_ij] | 0.5759 | 0.4365 | 0.5620 |
| LLM sentences + designed sentence model (alone) | 0.7330 | 0.4647 | 0.6205 |
| plain sentence MLP mixed with the structural model | 0.5574 | 0.4032 | 0.5414 |
| full OM-LEU 2 (designed sentence model mixed with the structural model) | 0.5581 | 0.3982 | 0.5412 |

### A. Same sentences, unstructured baselines — sentence channel alone

| variant | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | — | — | — |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.5820 | -0.1510* [-0.1695, -0.1331] | 0.4425 | -0.0228 [-0.0589, +0.0139] | 0.5892 | -0.0312 [-0.0662, +0.0024] |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.7379 | +0.0049 [-0.0073, +0.0168] | 0.5498 | +0.0858* [+0.0575, +0.1176] | 0.6455 | +0.0251* [+0.0051, +0.0439] |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.7240 | -0.0089 [-0.0216, +0.0038] | 0.5333 | +0.0692* [+0.0402, +0.0994] | 0.6411 | +0.0206 [-0.0013, +0.0417] |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.7253 | -0.0077 [-0.0186, +0.0031] | 0.4608 | -0.0039 [-0.0241, +0.0177] | 0.5903 | -0.0302* [-0.0515, -0.0097] |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.5759 | -0.1571* [-0.1744, -0.1398] | 0.4365 | -0.0285* [-0.0566, -0.0003] | 0.5620 | -0.0585* [-0.0876, -0.0302] |
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |

### A'. Same baselines mixed with the structural model

| variant | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | 0.5583 | +0.0002 [-0.0013, +0.0018] | 0.00 | 0.4108 | +0.0126* [+0.0034, +0.0237] | 0.00 | 0.5434 | +0.0022* [+0.0007, +0.0038] | 0.00 |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.5583 | +0.0001 [-0.0013, +0.0018] | 0.00 | 0.4116 | +0.0132* [+0.0018, +0.0249] | 0.27 | 0.5433 | +0.0021* [+0.0007, +0.0037] | 0.00 |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.5574 | -0.0007 [-0.0019, +0.0007] | 0.01 | 0.4032 | +0.0052 [-0.0022, +0.0131] | 0.14 | 0.5414 | +0.0002 [-0.0009, +0.0012] | 0.02 |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.5577 | -0.0004 [-0.0017, +0.0008] | 0.03 | 0.4032 | +0.0051 [-0.0024, +0.0135] | 0.15 | 0.5430 | +0.0018* [+0.0004, +0.0034] | 0.00 |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.5572 | -0.0010 [-0.0022, +0.0004] | 0.01 | 0.3994 | +0.0012 [-0.0048, +0.0081] | 0.25 | 0.5404 | -0.0008 [-0.0024, +0.0007] | 0.04 |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.5582 | +0.0001 [-0.0013, +0.0017] | 0.00 | 0.4082 | +0.0098* [+0.0011, +0.0191] | 0.31 | 0.5433 | +0.0021* [+0.0006, +0.0037] | 0.00 |
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |

### B. Knock-outs inside the designed sentence model — sentence channel alone

| variant | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.7479 | +0.0150* [+0.0065, +0.0240] | 0.5538 | +0.0897* [+0.0632, +0.1174] | 0.6497 | +0.0292* [+0.0162, +0.0421] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7505 | +0.0175* [+0.0088, +0.0269] | 0.5464 | +0.0823* [+0.0551, +0.1107] | 0.6486 | +0.0281* [+0.0147, +0.0404] |
| `no_salience` no salience attention (mean over the K sentences) | 0.7242 | -0.0088* [-0.0162, -0.0018] | 0.4692 | +0.0044 [-0.0121, +0.0217] | 0.6041 | -0.0164* [-0.0329, -0.0005] |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.7262 | -0.0067 [-0.0154, +0.0018] | 0.4708 | +0.0058 [-0.0123, +0.0240] | 0.6123 | -0.0082 [-0.0260, +0.0094] |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.7319 | -0.0011 [-0.0116, +0.0085] | 0.4657 | +0.0007 [-0.0160, +0.0162] | 0.6232 | +0.0028 [-0.0156, +0.0205] |
| `no_probe` no InfoNCE probe loss during pretraining | 0.7369 | +0.0039 [-0.0014, +0.0090] | 0.4669 | +0.0023 [-0.0080, +0.0127] | 0.6183 | -0.0022 [-0.0129, +0.0081] |
| `no_slot_dropout` no sentence-slot dropout | 0.7202 | -0.0128* [-0.0191, -0.0062] | 0.4898 | +0.0252* [+0.0142, +0.0354] | 0.6251 | +0.0047 [-0.0070, +0.0157] |
| `single_member` one member instead of 5 | 0.7438 | +0.0108* [+0.0061, +0.0159] | 0.5076 | +0.0423* [+0.0227, +0.0625] | 0.6500 | +0.0295* [+0.0170, +0.0426] |

### B'. Same knock-outs mixed with the structural model

| variant | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.5580 | -0.0002 [-0.0010, +0.0006] | 0.02 | 0.4056 | +0.0075* [+0.0001, +0.0153] | 0.16 | 0.5433 | +0.0021* [+0.0006, +0.0037] | 0.00 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.5580 | -0.0001 [-0.0011, +0.0007] | 0.02 | 0.4054 | +0.0073 [-0.0007, +0.0158] | 0.17 | 0.5432 | +0.0021* [+0.0006, +0.0036] | 0.00 |
| `no_salience` no salience attention (mean over the K sentences) | 0.5578 | -0.0003 [-0.0013, +0.0010] | 0.01 | 0.4017 | +0.0036 [-0.0015, +0.0092] | 0.20 | 0.5432 | +0.0020* [+0.0006, +0.0036] | 0.00 |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.5575 | -0.0006 [-0.0016, +0.0007] | 0.01 | 0.4009 | +0.0026 [-0.0030, +0.0089] | 0.26 | 0.5432 | +0.0020* [+0.0005, +0.0035] | 0.00 |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.5574 | -0.0007 [-0.0020, +0.0006] | 0.01 | 0.3975 | -0.0007 [-0.0061, +0.0041] | 0.27 | 0.5434 | +0.0022* [+0.0007, +0.0039] | 0.01 |
| `no_probe` no InfoNCE probe loss during pretraining | 0.5579 | -0.0002 [-0.0012, +0.0010] | 0.01 | 0.3983 | +0.0001 [-0.0030, +0.0032] | 0.23 | 0.5432 | +0.0020* [+0.0005, +0.0035] | 0.00 |
| `no_slot_dropout` no sentence-slot dropout | 0.5580 | -0.0002 [-0.0012, +0.0010] | 0.01 | 0.4014 | +0.0032 [-0.0004, +0.0066] | 0.24 | 0.5410 | -0.0002 [-0.0016, +0.0011] | 0.05 |
| `single_member` one member instead of 5 | 0.5582 | +0.0001 [-0.0009, +0.0011] | 0.01 | 0.4058 | +0.0075* [+0.0027, +0.0130] | 0.20 | 0.5431 | +0.0019* [+0.0005, +0.0035] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

### Where the improvement comes from (mean test NLL over seeds)

| model | swissmetro | optima | lpmc |
|---|---|---|---|
| no LLM, no designed sentence model: structural utility + residual | 0.6166 | 0.4017 | 0.4993 |
| no LLM, no structure: plain MLP on numeric inputs | 0.6345 | 0.4690 | 0.4990 |
| LLM sentences, no structure: plain MLP on the sentence embedding | 0.7017 | 0.5978 | 0.5645 |
| LLM sentences + z_i, no structure: plain MLP | 0.7141 | 0.4999 | 0.5521 |
| all inputs, no structure: plain MLP on [sentences, z_i, x_ij, h_ij] | 0.6256 | 0.4598 | 0.4626 |
| LLM sentences + designed sentence model (alone) | 0.6986 | 0.5037 | 0.5498 |
| plain sentence MLP mixed with the structural model | 0.6107 | 0.4089 | 0.4720 |
| full OM-LEU 2 (designed sentence model mixed with the structural model) | 0.6127 | 0.4106 | 0.4736 |

### A. Same sentences, unstructured baselines — sentence channel alone

| variant | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | — | — | — |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.6345 | -0.0641* [-0.0804, -0.0465] | 0.4690 | -0.0347* [-0.0629, -0.0036] | 0.4990 | -0.0509* [-0.0846, -0.0176] |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.7017 | +0.0031 [-0.0067, +0.0122] | 0.5978 | +0.0937* [+0.0678, +0.1207] | 0.5645 | +0.0147 [-0.0058, +0.0353] |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.6887 | -0.0099 [-0.0215, +0.0011] | 0.5880 | +0.0841* [+0.0562, +0.1126] | 0.5436 | -0.0062 [-0.0280, +0.0164] |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.7141 | +0.0155* [+0.0064, +0.0244] | 0.4999 | -0.0038 [-0.0247, +0.0163] | 0.5521 | +0.0024 [-0.0183, +0.0234] |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.6256 | -0.0730* [-0.0884, -0.0564] | 0.4598 | -0.0440* [-0.0669, -0.0208] | 0.4626 | -0.0873* [-0.1157, -0.0589] |
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |

### A'. Same baselines mixed with the structural model

| variant | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | 0.6166 | +0.0039* [+0.0008, +0.0072] | 0.00 | 0.4017 | -0.0088* [-0.0166, -0.0003] | 0.00 | 0.4993 | +0.0257* [+0.0148, +0.0369] | 0.00 |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.6165 | +0.0038* [+0.0007, +0.0072] | 0.00 | 0.4088 | -0.0018 [-0.0107, +0.0079] | 0.28 | 0.4992 | +0.0256* [+0.0147, +0.0368] | 0.00 |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.6107 | -0.0021* [-0.0041, -0.0002] | 0.11 | 0.4089 | -0.0017 [-0.0073, +0.0042] | 0.18 | 0.4720 | -0.0016 [-0.0086, +0.0055] | 0.25 |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.6080 | -0.0047* [-0.0074, -0.0020] | 0.16 | 0.4045 | -0.0060 [-0.0132, +0.0012] | 0.12 | 0.4644 | -0.0092* [-0.0168, -0.0016] | 0.31 |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.6120 | -0.0008 [-0.0024, +0.0009] | 0.09 | 0.4020 | -0.0086* [-0.0159, -0.0008] | 0.08 | 0.4872 | +0.0135* [+0.0060, +0.0217] | 0.20 |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.6101 | -0.0026 [-0.0069, +0.0019] | 0.16 | 0.4076 | -0.0031 [-0.0101, +0.0040] | 0.37 | 0.4821 | +0.0084 [-0.0018, +0.0186] | 0.22 |
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |

### B. Knock-outs inside the designed sentence model — sentence channel alone

| variant | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.7054 | +0.0068 [-0.0002, +0.0134] | 0.6042 | +0.1002* [+0.0754, +0.1258] | 0.5749 | +0.0250* [+0.0098, +0.0395] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7052 | +0.0066 [-0.0006, +0.0134] | 0.5931 | +0.0892* [+0.0644, +0.1150] | 0.5707 | +0.0208* [+0.0055, +0.0361] |
| `no_salience` no salience attention (mean over the K sentences) | 0.6943 | -0.0043 [-0.0105, +0.0013] | 0.5187 | +0.0150 [-0.0009, +0.0309] | 0.5479 | -0.0018 [-0.0183, +0.0141] |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.7034 | +0.0048 [-0.0021, +0.0120] | 0.5016 | -0.0021 [-0.0184, +0.0136] | 0.5499 | +0.0001 [-0.0157, +0.0165] |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.7242 | +0.0256* [+0.0173, +0.0340] | 0.4987 | -0.0049 [-0.0216, +0.0115] | 0.5891 | +0.0394* [+0.0213, +0.0578] |
| `no_probe` no InfoNCE probe loss during pretraining | 0.7011 | +0.0024 [-0.0013, +0.0062] | 0.4992 | -0.0045 [-0.0153, +0.0066] | 0.5579 | +0.0082 [-0.0013, +0.0179] |
| `no_slot_dropout` no sentence-slot dropout | 0.6969 | -0.0017 [-0.0063, +0.0035] | 0.5192 | +0.0154* [+0.0062, +0.0245] | 0.5556 | +0.0057 [-0.0035, +0.0153] |
| `single_member` one member instead of 5 | 0.7057 | +0.0071* [+0.0025, +0.0114] | 0.5053 | +0.0015 [-0.0122, +0.0156] | 0.5656 | +0.0159* [+0.0054, +0.0274] |

### B'. Same knock-outs mixed with the structural model

| variant | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.6107 | -0.0020 [-0.0050, +0.0007] | 0.12 | 0.4140 | +0.0033 [-0.0031, +0.0100] | 0.22 | 0.4728 | -0.0008 [-0.0055, +0.0040] | 0.26 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.6090 | -0.0038* [-0.0065, -0.0012] | 0.16 | 0.4133 | +0.0026 [-0.0036, +0.0089] | 0.22 | 0.4721 | -0.0015 [-0.0067, +0.0034] | 0.26 |
| `no_salience` no salience attention (mean over the K sentences) | 0.6117 | -0.0011* [-0.0022, -0.0000] | 0.12 | 0.4099 | -0.0007 [-0.0051, +0.0039] | 0.22 | 0.4738 | +0.0001 [-0.0058, +0.0061] | 0.29 |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.6146 | +0.0018 [-0.0000, +0.0039] | 0.09 | 0.4101 | -0.0005 [-0.0054, +0.0045] | 0.30 | 0.4768 | +0.0032 [-0.0025, +0.0088] | 0.24 |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.6138 | +0.0011 [-0.0003, +0.0025] | 0.09 | 0.4100 | -0.0006 [-0.0049, +0.0039] | 0.30 | 0.4868 | +0.0132* [+0.0070, +0.0198] | 0.20 |
| `no_probe` no InfoNCE probe loss during pretraining | 0.6163 | +0.0036* [+0.0006, +0.0068] | 0.00 | 0.4087 | -0.0019 [-0.0050, +0.0014] | 0.30 | 0.4771 | +0.0035* [+0.0002, +0.0071] | 0.27 |
| `no_slot_dropout` no sentence-slot dropout | 0.6128 | +0.0001 [-0.0009, +0.0011] | 0.13 | 0.4115 | +0.0008 [-0.0021, +0.0036] | 0.28 | 0.4817 | +0.0080* [+0.0019, +0.0145] | 0.21 |
| `single_member` one member instead of 5 | 0.6124 | -0.0003 [-0.0011, +0.0004] | 0.11 | 0.4067 | -0.0039* [-0.0075, -0.0005] | 0.26 | 0.4747 | +0.0011 [-0.0026, +0.0049] | 0.28 |


## person-level split (no test person in training) — π stacked out of fold

### Where the improvement comes from (mean test NLL over seeds)

| model | swissmetro | optima | lpmc |
|---|---|---|---|
| no LLM, no designed sentence model: structural utility + residual | — | 0.3688 | 0.4769 |
| no LLM, no structure: plain MLP on numeric inputs | — | 0.4301 | — |
| LLM sentences, no structure: plain MLP on the sentence embedding | — | 0.6220 | — |
| LLM sentences + z_i, no structure: plain MLP | — | 0.4817 | — |
| all inputs, no structure: plain MLP on [sentences, z_i, x_ij, h_ij] | — | 0.4321 | 0.4284 |
| LLM sentences + designed sentence model (alone) | — | 0.5293 | 0.5262 |
| plain sentence MLP mixed with the structural model | — | 0.3602 | — |
| full OM-LEU 2 (designed sentence model mixed with the structural model) | — | 0.4121 | 0.4459 |

### A. Same sentences, unstructured baselines — sentence channel alone

| variant | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | — | — | — |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | — | — | 0.4301 | — | — | — |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | — | — | 0.6220 | — | — | — |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | — | — | 0.6000 | — | — | — |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | — | — | 0.4817 | — | — | — |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | — | — | 0.4321 | — | 0.4284 | -0.0978* [-0.1467, -0.0540] |
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | 0.5293 | +0.0000 [+0.0000, +0.0000] | 0.5262 | +0.0000 [+0.0000, +0.0000] |

### A'. Same baselines mixed with the structural model

| variant | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | 0.3688 | — | 0.00 | 0.4769 | +0.0310* [+0.0112, +0.0529] | 0.00 |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | — | — | — | 0.3558 | — | 0.29 | — | — | — |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | — | — | — | 0.3602 | — | 0.12 | — | — | — |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | — | — | — | 0.3594 | — | 0.11 | — | — | — |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | — | — | — | 0.3568 | — | 0.13 | — | — | — |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | — | — | — | 0.3590 | — | 0.32 | 0.4345 | -0.0114 [-0.0289, +0.0059] | 0.62 |
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | — | 0.4121 | +0.0000 [+0.0000, +0.0000] | 0.21 | 0.4459 | +0.0000 [+0.0000, +0.0000] | 0.32 |

### B. Knock-outs inside the designed sentence model — sentence channel alone

| variant | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | 0.5293 | +0.0000 [+0.0000, +0.0000] | 0.5262 | +0.0000 [+0.0000, +0.0000] |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | — | — | 0.6346 | +0.1051* [+0.0763, +0.1338] | 0.5628 | +0.0366* [+0.0162, +0.0562] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | — | — | 0.6082 | — | — | — |
| `no_salience` no salience attention (mean over the K sentences) | — | — | 0.4891 | — | 0.5110 | -0.0152 [-0.0438, +0.0115] |
| `person_attention` person-conditioned attention over the K sentences instead of salience | — | — | 0.4753 | — | 0.5254 | -0.0008 [-0.0262, +0.0231] |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | — | — | 0.4751 | — | 0.5473 | +0.0211 [-0.0131, +0.0541] |
| `no_probe` no InfoNCE probe loss during pretraining | — | — | 0.4731 | — | 0.5266 | +0.0004 [-0.0159, +0.0157] |
| `no_slot_dropout` no sentence-slot dropout | — | — | 0.5191 | — | 0.5383 | +0.0121 [-0.0055, +0.0288] |
| `single_member` one member instead of 5 | — | — | 0.4889 | — | — | — |

### B'. Same knock-outs mixed with the structural model

| variant | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | — | 0.4121 | +0.0000 [+0.0000, +0.0000] | 0.21 | 0.4459 | +0.0000 [+0.0000, +0.0000] | 0.32 |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | — | — | — | 0.4135 | +0.0014 [-0.0042, +0.0070] | 0.13 | 0.4476 | +0.0017 [-0.0044, +0.0075] | 0.30 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | — | — | — | 0.3609 | — | 0.14 | — | — | — |
| `no_salience` no salience attention (mean over the K sentences) | — | — | — | 0.3590 | — | 0.19 | 0.4531 | +0.0072 [-0.0035, +0.0178] | 0.30 |
| `person_attention` person-conditioned attention over the K sentences instead of salience | — | — | — | 0.3572 | — | 0.21 | 0.4506 | +0.0047 [-0.0039, +0.0138] | 0.30 |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | — | — | — | 0.3565 | — | 0.19 | 0.4595 | +0.0136* [+0.0018, +0.0256] | 0.19 |
| `no_probe` no InfoNCE probe loss during pretraining | — | — | — | 0.3574 | — | 0.20 | 0.4526 | +0.0067* [+0.0009, +0.0132] | 0.29 |
| `no_slot_dropout` no sentence-slot dropout | — | — | — | 0.3609 | — | 0.17 | 0.4472 | +0.0013 [-0.0054, +0.0075] | 0.30 |
| `single_member` one member instead of 5 | — | — | — | 0.3553 | — | 0.00 | — | — | — |
