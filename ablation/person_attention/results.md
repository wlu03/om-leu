# `person_attention`: person-conditioned attention over the K sentences instead of salience

Config overrides: `{'member_kw': (('attn', 'person'),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.7262 | -0.0067 [-0.0154, +0.0018] | 0.4708 | +0.0058 [-0.0123, +0.0240] | 0.6123 | -0.0082 [-0.0260, +0.0094] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.5575 | -0.0006 [-0.0016, +0.0007] | 0.01 | 0.4009 | +0.0026 [-0.0030, +0.0089] | 0.26 | 0.5432 | +0.0020* [+0.0005, +0.0035] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.7034 | +0.0048 [-0.0021, +0.0120] | 0.5016 | -0.0021 [-0.0184, +0.0136] | 0.5499 | +0.0001 [-0.0157, +0.0165] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `person_attention` person-conditioned attention over the K sentences instead of salience | 0.6146 | +0.0018 [-0.0000, +0.0039] | 0.09 | 0.4101 | -0.0005 [-0.0054, +0.0045] | 0.30 | 0.4768 | +0.0032 [-0.0025, +0.0088] | 0.24 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | 0.5293 | +0.0000 [+0.0000, +0.0000] | 0.5262 | +0.0000 [+0.0000, +0.0000] |
| `person_attention` person-conditioned attention over the K sentences instead of salience | — | — | 0.4753 | — | 0.5254 | -0.0008 [-0.0262, +0.0231] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | — | 0.4121 | +0.0000 [+0.0000, +0.0000] | 0.21 | 0.4459 | +0.0000 [+0.0000, +0.0000] | 0.32 |
| `person_attention` person-conditioned attention over the K sentences instead of salience | — | — | — | 0.3572 | — | 0.21 | 0.4506 | +0.0047 [-0.0039, +0.0138] | 0.30 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.33 | 0.7200 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5590 | 0.032 | 1.45 | 0.7315 | 75.5% |
| chrono | val | swissmetro | 13 | 0.5538 | 0.002 | 1.30 | 0.7272 | 75.7% |
| chrono | val | optima | 7 | 0.4014 | 0.212 | 1.20 | 0.4677 | 84.0% |
| chrono | val | optima | 11 | 0.4266 | 0.380 | 1.43 | 0.5031 | 82.9% |
| chrono | val | optima | 13 | 0.3748 | 0.179 | 1.33 | 0.4415 | 86.0% |
| chrono | val | lpmc | 7 | 0.5397 | 0.002 | 0.95 | 0.6028 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6396 | 78.8% |
| chrono | val | lpmc | 13 | 0.5200 | 0.002 | 1.01 | 0.5944 | 81.0% |
| person | oof | optima | 7 | 0.3572 | 0.207 | 1.19 | 0.4753 | 85.0% |
| person | oof | lpmc | 7 | 0.4506 | 0.303 | 1.18 | 0.5254 | 83.9% |
| person | val | swissmetro | 7 | 0.6700 | 0.002 | 1.01 | 0.7308 | 70.3% |
| person | val | swissmetro | 11 | 0.5545 | 0.002 | 1.01 | 0.6484 | 75.9% |
| person | val | swissmetro | 13 | 0.6192 | 0.257 | 1.30 | 0.7311 | 72.2% |
| person | val | optima | 7 | 0.3842 | 0.433 | 1.08 | 0.4753 | 84.0% |
| person | val | optima | 11 | 0.4715 | 0.271 | 1.18 | 0.5792 | 79.7% |
| person | val | optima | 13 | 0.3746 | 0.186 | 1.10 | 0.4502 | 84.6% |
| person | val | lpmc | 7 | 0.4582 | 0.157 | 1.07 | 0.5254 | 82.7% |
| person | val | lpmc | 11 | 0.5680 | 0.261 | 1.11 | 0.6031 | 78.1% |
| person | val | lpmc | 13 | 0.4043 | 0.305 | 1.09 | 0.5210 | 84.8% |
