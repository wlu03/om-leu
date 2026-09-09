# `plain_nn_sentences_flat`: plain MLP on the K concatenated sentence embeddings (slot order kept)

Config overrides: `{'member_kind': 'plain_nn', 'member_kw': (('inputs', ('sent_flat',)),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.7255 | -0.0101 [-0.0267, +0.0062] | 0.5333 | +0.0692* [+0.0402, +0.0994] | 0.6411 | +0.0206 [-0.0013, +0.0417] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.5593 | -0.0008 [-0.0019, +0.0001] | 0.02 | 0.4032 | +0.0051 [-0.0024, +0.0135] | 0.15 | 0.5430 | +0.0018* [+0.0004, +0.0034] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.7380 | — | 0.5880 | +0.0841* [+0.0562, +0.1126] | 0.5455 | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `plain_nn_sentences_flat` plain MLP on the K concatenated sentence embeddings (slot order kept) | 0.6700 | — | 0.00 | 0.4045 | -0.0060 [-0.0132, +0.0012] | 0.12 | 0.4957 | — | 0.28 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5596 | 0.002 | 1.32 | 0.7146 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5590 | 0.034 | 1.45 | 0.7363 | 75.5% |
| chrono | val | optima | 7 | 0.4159 | 0.142 | 1.19 | 0.5686 | 82.3% |
| chrono | val | optima | 11 | 0.4139 | 0.154 | 1.31 | 0.4959 | 83.9% |
| chrono | val | optima | 13 | 0.3798 | 0.165 | 1.40 | 0.5355 | 86.4% |
| chrono | val | lpmc | 7 | 0.5394 | 0.002 | 0.95 | 0.6112 | 81.3% |
| chrono | val | lpmc | 11 | 0.5696 | 0.002 | 0.88 | 0.6830 | 78.8% |
| chrono | val | lpmc | 13 | 0.5200 | 0.002 | 1.01 | 0.6291 | 81.2% |
| person | val | swissmetro | 7 | 0.6700 | 0.002 | 1.01 | 0.7380 | 70.3% |
| person | val | optima | 7 | 0.3688 | 0.002 | 0.95 | 0.6000 | 85.3% |
| person | val | optima | 11 | 0.4722 | 0.259 | 1.30 | 0.6356 | 78.7% |
| person | val | optima | 13 | 0.3726 | 0.105 | 1.11 | 0.5285 | 84.9% |
| person | val | lpmc | 7 | 0.4446 | 0.264 | 1.11 | 0.5195 | 83.7% |
| person | val | lpmc | 11 | 0.5467 | 0.301 | 1.14 | 0.5715 | 79.1% |
