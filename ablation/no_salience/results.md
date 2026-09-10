# `no_salience`: no salience attention (mean over the K sentences)

Config overrides: `{'member_kw': (('attn', 'mean'),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_salience` no salience attention (mean over the K sentences) | 0.7242 | -0.0088* [-0.0162, -0.0018] | 0.4692 | +0.0044 [-0.0121, +0.0217] | 0.6041 | -0.0164* [-0.0329, -0.0005] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_salience` no salience attention (mean over the K sentences) | 0.5578 | -0.0003 [-0.0013, +0.0010] | 0.01 | 0.4017 | +0.0036 [-0.0015, +0.0092] | 0.20 | 0.5432 | +0.0020* [+0.0006, +0.0036] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_salience` no salience attention (mean over the K sentences) | 0.6943 | -0.0043 [-0.0105, +0.0013] | 0.5187 | +0.0150 [-0.0009, +0.0309] | 0.5479 | -0.0018 [-0.0183, +0.0141] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_salience` no salience attention (mean over the K sentences) | 0.6117 | -0.0011* [-0.0022, -0.0000] | 0.12 | 0.4099 | -0.0007 [-0.0051, +0.0039] | 0.22 | 0.4738 | +0.0001 [-0.0058, +0.0061] | 0.29 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_salience` no salience attention (mean over the K sentences) | 0.6943 | -0.0043 [-0.0105, +0.0013] | 0.5187 | +0.0150 [-0.0009, +0.0309] | 0.5479 | -0.0018 [-0.0183, +0.0141] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6087 | +0.0000 [+0.0000, +0.0000] | 0.18 | 0.4004 | +0.0000 [+0.0000, +0.0000] | 0.24 | 0.4708 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_salience` no salience attention (mean over the K sentences) | 0.6077 | -0.0009 [-0.0024, +0.0005] | 0.18 | 0.4034 | +0.0029 [-0.0007, +0.0067] | 0.21 | 0.4719 | +0.0011 [-0.0050, +0.0074] | 0.30 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.32 | 0.7112 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5599 | 0.026 | 1.44 | 0.7373 | 75.5% |
| chrono | val | swissmetro | 13 | 0.5539 | 0.002 | 1.30 | 0.7242 | 75.7% |
| chrono | val | optima | 7 | 0.4075 | 0.133 | 1.07 | 0.4664 | 82.9% |
| chrono | val | optima | 11 | 0.4204 | 0.322 | 1.35 | 0.4860 | 83.2% |
| chrono | val | optima | 13 | 0.3773 | 0.134 | 1.24 | 0.4553 | 86.4% |
| chrono | val | lpmc | 7 | 0.5397 | 0.002 | 0.95 | 0.5932 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6252 | 78.8% |
| chrono | val | lpmc | 13 | 0.5203 | 0.002 | 1.02 | 0.5939 | 81.0% |
| person | oof | swissmetro | 7 | 0.6594 | 0.175 | 1.15 | 0.7314 | 70.9% |
| person | oof | swissmetro | 11 | 0.5477 | 0.200 | 1.20 | 0.6345 | 76.5% |
| person | oof | swissmetro | 13 | 0.6162 | 0.165 | 1.15 | 0.7169 | 71.9% |
| person | oof | optima | 7 | 0.3590 | 0.186 | 1.17 | 0.4891 | 84.3% |
| person | oof | optima | 11 | 0.4740 | 0.211 | 1.12 | 0.6044 | 79.1% |
| person | oof | optima | 13 | 0.3773 | 0.226 | 1.13 | 0.4628 | 84.6% |
| person | oof | lpmc | 7 | 0.4531 | 0.300 | 1.17 | 0.5110 | 84.1% |
| person | oof | lpmc | 11 | 0.5647 | 0.321 | 1.22 | 0.6006 | 78.5% |
| person | oof | lpmc | 13 | 0.3979 | 0.273 | 1.23 | 0.5321 | 84.8% |
| person | val | swissmetro | 7 | 0.6699 | 0.002 | 1.01 | 0.7314 | 70.3% |
| person | val | swissmetro | 11 | 0.5484 | 0.132 | 1.13 | 0.6345 | 76.3% |
| person | val | swissmetro | 13 | 0.6166 | 0.214 | 1.25 | 0.7169 | 72.0% |
| person | val | optima | 7 | 0.3836 | 0.349 | 1.04 | 0.4891 | 82.9% |
| person | val | optima | 11 | 0.4717 | 0.180 | 1.10 | 0.6044 | 79.4% |
| person | val | optima | 13 | 0.3745 | 0.126 | 1.08 | 0.4628 | 84.6% |
| person | val | lpmc | 7 | 0.4564 | 0.218 | 1.10 | 0.5110 | 83.3% |
| person | val | lpmc | 11 | 0.5594 | 0.355 | 1.16 | 0.6006 | 78.7% |
| person | val | lpmc | 13 | 0.4055 | 0.287 | 1.11 | 0.5321 | 84.6% |
