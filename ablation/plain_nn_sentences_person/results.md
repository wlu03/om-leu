# `plain_nn_sentences_person`: plain MLP on [mean sentence embedding, person covariates z_i]

Config overrides: `{'member_kind': 'plain_nn', 'member_kw': (('inputs', ('sent', 'z')),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.7253 | -0.0077 [-0.0186, +0.0031] | 0.4608 | -0.0039 [-0.0241, +0.0177] | 0.5903 | -0.0302* [-0.0515, -0.0097] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.5572 | -0.0010 [-0.0022, +0.0004] | 0.01 | 0.3994 | +0.0012 [-0.0048, +0.0081] | 0.25 | 0.5404 | -0.0008 [-0.0024, +0.0007] | 0.04 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.7141 | +0.0155* [+0.0064, +0.0244] | 0.4999 | -0.0038 [-0.0247, +0.0163] | 0.5521 | +0.0024 [-0.0183, +0.0234] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.6120 | -0.0008 [-0.0024, +0.0009] | 0.09 | 0.4020 | -0.0086* [-0.0159, -0.0008] | 0.08 | 0.4872 | +0.0135* [+0.0060, +0.0217] | 0.20 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.7141 | +0.0155* [+0.0064, +0.0244] | 0.4999 | -0.0038 [-0.0247, +0.0163] | 0.5521 | +0.0024 [-0.0183, +0.0234] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6087 | +0.0000 [+0.0000, +0.0000] | 0.18 | 0.4004 | +0.0000 [+0.0000, +0.0000] | 0.24 | 0.4708 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `plain_nn_sentences_person` plain MLP on [mean sentence embedding, person covariates z_i] | 0.6097 | +0.0010 [-0.0011, +0.0032] | 0.14 | 0.3976 | -0.0028 [-0.0079, +0.0025] | 0.10 | 0.4767 | +0.0059 [-0.0012, +0.0132] | 0.29 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5597 | 0.002 | 1.33 | 0.7229 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5579 | 0.028 | 1.44 | 0.7293 | 75.4% |
| chrono | val | swissmetro | 13 | 0.5539 | 0.002 | 1.30 | 0.7238 | 75.7% |
| chrono | val | optima | 7 | 0.4053 | 0.173 | 1.08 | 0.4704 | 83.6% |
| chrono | val | optima | 11 | 0.4191 | 0.445 | 1.41 | 0.4746 | 81.8% |
| chrono | val | optima | 13 | 0.3740 | 0.124 | 1.21 | 0.4375 | 86.0% |
| chrono | val | lpmc | 7 | 0.5396 | 0.002 | 0.95 | 0.5582 | 81.3% |
| chrono | val | lpmc | 11 | 0.5616 | 0.112 | 0.91 | 0.6159 | 79.3% |
| chrono | val | lpmc | 13 | 0.5200 | 0.002 | 1.01 | 0.5966 | 81.0% |
| person | oof | swissmetro | 7 | 0.6634 | 0.118 | 1.11 | 0.7446 | 70.6% |
| person | oof | swissmetro | 11 | 0.5488 | 0.177 | 1.18 | 0.6546 | 76.4% |
| person | oof | swissmetro | 13 | 0.6169 | 0.111 | 1.11 | 0.7431 | 72.0% |
| person | oof | optima | 7 | 0.3568 | 0.131 | 1.11 | 0.4817 | 84.6% |
| person | oof | optima | 11 | 0.4621 | 0.002 | 1.02 | 0.5600 | 80.1% |
| person | oof | optima | 13 | 0.3740 | 0.163 | 1.07 | 0.4579 | 84.3% |
| person | oof | lpmc | 7 | 0.4500 | 0.292 | 1.14 | 0.5011 | 83.7% |
| person | oof | lpmc | 11 | 0.5784 | 0.308 | 1.17 | 0.6331 | 77.9% |
| person | oof | lpmc | 13 | 0.4016 | 0.270 | 1.19 | 0.5219 | 84.2% |
| person | val | swissmetro | 7 | 0.6701 | 0.002 | 1.01 | 0.7446 | 70.3% |
| person | val | swissmetro | 11 | 0.5490 | 0.139 | 1.13 | 0.6546 | 76.4% |
| person | val | swissmetro | 13 | 0.6168 | 0.131 | 1.17 | 0.7431 | 72.0% |
| person | val | optima | 7 | 0.3688 | 0.002 | 0.95 | 0.4817 | 85.3% |
| person | val | optima | 11 | 0.4619 | 0.002 | 0.96 | 0.5600 | 80.1% |
| person | val | optima | 13 | 0.3751 | 0.226 | 1.11 | 0.4579 | 84.6% |
| person | val | lpmc | 7 | 0.4764 | 0.002 | 1.02 | 0.5011 | 82.2% |
| person | val | lpmc | 11 | 0.5750 | 0.285 | 1.08 | 0.6331 | 77.9% |
| person | val | lpmc | 13 | 0.4102 | 0.312 | 1.08 | 0.5219 | 84.0% |
