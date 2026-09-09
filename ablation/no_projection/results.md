# `no_projection`: no 768 -> 32 projection (heads on the full embedding)

Config overrides: `{'member_kw': (('proj', False),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.7330 | -0.0026 [-0.0147, +0.0092] | 0.4657 | +0.0007 [-0.0160, +0.0162] | 0.6232 | +0.0028 [-0.0156, +0.0205] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.5592 | -0.0009 [-0.0024, +0.0003] | 0.02 | 0.3975 | -0.0007 [-0.0061, +0.0041] | 0.27 | 0.5434 | +0.0022* [+0.0007, +0.0039] | 0.01 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.7149 | +0.0287* [+0.0185, +0.0389] | 0.4987 | -0.0049 [-0.0216, +0.0115] | 0.5891 | +0.0394* [+0.0213, +0.0578] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_projection` no 768 -> 32 projection (heads on the full embedding) | 0.6112 | +0.0012 [-0.0000, +0.0025] | 0.06 | 0.4100 | -0.0006 [-0.0049, +0.0039] | 0.30 | 0.4868 | +0.0132* [+0.0070, +0.0198] | 0.20 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5595 | 0.011 | 1.35 | 0.7320 | 75.5% |
| chrono | val | swissmetro | 11 | 0.5589 | 0.020 | 1.44 | 0.7341 | 75.4% |
| chrono | val | optima | 7 | 0.3983 | 0.282 | 1.24 | 0.4388 | 84.3% |
| chrono | val | optima | 11 | 0.4165 | 0.351 | 1.34 | 0.4830 | 82.9% |
| chrono | val | optima | 13 | 0.3778 | 0.165 | 1.32 | 0.4754 | 85.7% |
| chrono | val | lpmc | 7 | 0.5397 | 0.002 | 0.95 | 0.6009 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6317 | 78.8% |
| chrono | val | lpmc | 13 | 0.5207 | 0.031 | 1.03 | 0.6371 | 81.2% |
| person | val | swissmetro | 7 | 0.6700 | 0.002 | 1.01 | 0.7533 | 70.3% |
| person | val | swissmetro | 11 | 0.5524 | 0.111 | 1.13 | 0.6764 | 76.2% |
| person | val | optima | 7 | 0.3841 | 0.387 | 1.03 | 0.4751 | 83.3% |
| person | val | optima | 11 | 0.4688 | 0.304 | 1.12 | 0.5597 | 78.7% |
| person | val | optima | 13 | 0.3771 | 0.216 | 1.09 | 0.4613 | 83.9% |
| person | val | lpmc | 7 | 0.4675 | 0.076 | 1.04 | 0.5473 | 82.7% |
| person | val | lpmc | 11 | 0.5817 | 0.295 | 1.11 | 0.6522 | 78.9% |
| person | val | lpmc | 13 | 0.4113 | 0.220 | 1.08 | 0.5678 | 84.8% |
