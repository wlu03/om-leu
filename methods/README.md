# methods/ — literature choice models on the same splits as OM-LEU

Each sub-folder is one method from the mode-choice literature that uses the
Swissmetro, Optima or LPMC datasets (surveys in `methods/literature/`). Every
method is implemented against the shared loader in `methods/common/data.py`,
which reads the numeric level-of-service and person columns from the raw
`.dat` files and aligns them to the exact per-seed train / val / test split of
the OM-LEU runs (`<dataset>/results/<run_tag>/seed_<s>/omleu/records.pkl`),
so all rows in the comparison table are paired at the event level.

| Folder | Method | Family | Paper |
|---|---|---|---|
| `mnl/` | Multinomial logit, alt-specific LOS coefficients + socio-demographic shifters | RUM | Bierlaire 2020 (Biogeme); Hillel 2019 |
| `nested_logit/` | Nested logit | RUM | Ben-Akiva & Lerman 1985; Biogeme examples |
| `mixed_logit/` | Panel mixed logit, normal random time / cost coefficients | RUM | Train 2009; Biogeme examples |
| `iclv_hybrid_choice/` | Integrated choice and latent variable (hybrid choice) | Hybrid RUM | Ben-Akiva et al. 2002; Atasoy et al. 2013; Bierlaire 2018 / 2026 |
| `l_mnl/` | Learning-MNL: linear utility + NN representation term | RUM + NN | Sifringer, Lurkin, Alahi 2020 |
| `asu_dnn/` | Alternative-specific utility DNN | NN (RUM-shaped) | Wang, Wang, Zhao 2020 |
| `tastenet_mnl/` | Taste parameters from a network over person covariates | RUM + NN | Han, Pereira, Ben-Akiva, Zegras 2022 |
| `rumboost/` | Gradient-boosted random utility model | RUM + GBDT | Salvadé & Hillel 2025 |
| `gbdt_full_features/` | Multiclass LightGBM on all features | ML classifier | Hillel et al. 2018 / 2021 |

Not implemented (documented in `literature/` only): LLM-based predictors
(Mo et al. 2023, Liu et al. 2026, Alsaleh & Farooq 2025, ATHENA), tabular
foundation-model adapters (Wang et al. 2026), RUMnet, Alt-GNN, RUM-NN,
DCM-ARD, MO-VNS assisted specification, E-MNL embeddings. The repo's own
zero-shot / few-shot Llama rankers already cover the "LLM as direct
predictor" family on these datasets.

## Running

```bash
venv/bin/python methods/run_methods.py --datasets swissmetro optima lpmc --seeds 7 11 13
venv/bin/python methods/make_table.py          # -> methods/results/comparison_table.{md,csv}
venv/bin/python methods/run_methods.py --datasets optima --methods mnl iclv_hybrid_choice --seeds 7 --force
```

Results: `methods/results/<dataset>/<method>/seed_<s>.json` (metrics from
`src/eval/metrics.py`, per-event NLL and top-1 flags for paired tests, fitted
parameters / VOT / nest scales / latent loadings). The OM-LEU run each
dataset is aligned to is set in `methods/common/data.py::DEFAULT_RUN_TAGS`
(override with `METHODS_RUN_TAG_<DATASET>`).

## Protocol

- Same events, same seeds (7, 11, 13) as the OM-LEU runs; metrics computed by
  the same functions; early stopping on the validation split; test reported.
- Features: numeric level of service per alternative (time, cost, waiting /
  access, transfers, traffic, distance) plus person and trip covariates
  (`ALT_FEATURES`, and `z_names` in the dataset object). Model units: hours,
  CHF (Swissmetro cost / 100 as in Biogeme), GBP.
- Nests: Swissmetro {train, Swissmetro} vs {car}; Optima {car} vs {PT, soft};
  LPMC {walk, cycle} vs {PT, drive}.
- ICLV runs on Optima only (the only dataset with attitudinal indicators).
- LightGBM methods run in a torch-free child process (both libraries bundle
  `libomp`; loading both in one process segfaults on macOS).

## Caveats

- Hyperparameters are fixed, not tuned per dataset (hidden 64, dropout 0.1,
  weight decay 1e-4 for the networks; 64 draws for simulated likelihoods).
- The repo's ML baselines (`src/baselines/`) see only price / popularity
  columns, so the literature models here are the fair "tabular" comparison;
  see `docs/optima_lpmc.md` for why.
- This repo's splits (per-person chronological, or cold-start for Optima)
  differ from the canonical splits in the papers, so absolute numbers are not
  comparable with the literature values quoted in the READMEs.

## Results (3 seeds, mean Top-1 / NLL; full table in `results/comparison_table.md`)

| Method | Swissmetro | Optima | LPMC |
|---|---|---|---|
| OM-LEU (repo, β = 1) | 68.4 % / 0.730 | 75.2 % / 0.585 | 74.3 % / 0.684 |
| MNL | 63.1 % / 0.825 | 82.7 % / 0.461 | 76.7 % / 0.645 |
| Nested logit | 63.2 % / 0.825 | 82.5 % / 0.459 | 76.8 % / 0.652 |
| Mixed logit | 64.8 % / 0.805 | 82.2 % / 0.462 | 76.3 % / 0.664 |
| ICLV | n/a | 82.5 % / 0.460 | n/a |
| L-MNL | 69.9 % / 0.699 | 82.5 % / 0.483 | 74.9 % / 0.715 |
| ASU-DNN | 72.9 % / 0.664 | 82.7 % / 0.449 | 76.8 % / 0.684 |
| TasteNet-MNL | 71.9 % / 0.663 | 81.0 % / 0.545 | 77.3 % / 0.677 |
| RUMBoost | 64.4 % / 0.788 | 80.8 % / 0.462 | 74.8 % / 0.669 |
| GBDT, full features | 72.2 % / 0.652 | 83.2 % / 0.411 | 78.3 % / 0.615 |

Discussion: `docs/methods_comparison.md`.

### OM-LEU improved (numeric residual, two-stage; see `docs/omleu_improvements.md`)

| Row | Swissmetro | Optima | LPMC |
|---|---|---|---|
| OM-LEU + numeric residual (LOS only) | 73.6 % / 0.601 | 77.9 % / 0.534 | 78.4 % / 0.589 |
| OM-LEU + numeric residual (LOS + person shifters) | 73.6 % / 0.608 | 82.7 % / 0.455 | 80.2 % / 0.597 |
| residual only (stage-1 MNL inside OM-LEU) | 73.6 % / 0.617 | 83.5 % / 0.453 | 80.3 % / 0.596 |
