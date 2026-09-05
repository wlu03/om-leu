# Swissmetro: literature survey (choice models, hybrid, ML / DL / LLM)

Compiled 2026-09-05 (web survey). "Unverified" marks details behind paywalls.

## The dataset

Bierlaire, Axhausen, Abay (2001), "The acceptance of modal innovation: the case of
Swissmetro", STRC; description
https://transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf.
SP survey, March 1998, rail (441) and car (750) respondents, 9 tasks each → 10,728
rows. Alternatives train / Swissmetro / car (car only for car owners). Attributes
TT, CO, HE per alternative, SM_SEATS; socio-demographics PURPOSE, FIRST, TICKET, WHO,
LUGGAGE, AGE, MALE, INCOME, GA, ORIGIN, DEST. Three preprocessing conventions make
cross-paper numbers non-comparable: Biogeme (PURPOSE ∈ {1,3}, CHOICE ≠ 0 → 6,768),
Sifringer (drop unknown choice and rows with an unavailable alternative → 9,036,
the convention this repo uses), Han/TasteNet (10,692) / RUMnet (10,719).

## Methods

### `mnl_biogeme`, `nested_logit`, `mixed_logit` — Bierlaire (2020), *A short introduction to PandasBiogeme*, TRANSP-OR 200605
V_train = ASC + B_TIME·TT/100 + B_COST·CO/100 (cost 0 for GA holders on rail);
generic time and cost. In-sample (6,768 obs): MNL LL −5,331.3 (ρ² 0.235; B_COST
−1.08, B_TIME −1.28, VOT ≈ 71 CHF/h); nested logit ({train, car} vs {SM}, µ = 2.05)
LL −5,236.9; mixed logit (normal B_TIME) LL −5,214.8. Cross-nested and
latent-class examples in the same gallery.

### `l_mnl` — Sifringer, Lurkin, Alahi (2020), *TR-B* 140:236–261; arXiv:1812.09747; code github.com/BSifringer/EnhancedDCM
U = βᵀx + r(q; w) + ε: interpretable linear part on X plus a dense-NN
representation on the disjoint Q (100 tanh neurons), jointly estimated by Adam.
9,036 obs, 7,234 / 1,802. Test LL: Logit(X1) −1,433 (ρ² 0.28); DNN −1,257;
L-MNL(X1, Q1) −1,181; L-MNL(X2, Q2) −1,108 (ρ² 0.44). VOT 0.52 → 0.94 CHF/min.
Nesting the L-MNL brings the nest parameter to 1.0. β biased if X and Q correlate.

### `tastenet_mnl` — Han, Pereira, Ben-Akiva, Zegras (2022), *TR-B* 163:166–186; arXiv:2002.00922; code github.com/YafeiHan-MIT/TasteNet-MNL
β(z) = TasteNet(z) feeds a linear-in-attributes MNL; cost coefficient fixed to −1
(WTP space); sign constraints on outputs. 10,692 obs (7,484 / 1,604 / 1,604).
Test NLL / accuracy: MNL-A 0.755 / 0.660; MNL-C (all first-order interactions)
0.698 / 0.678; mixed logit 0.703 / 0.686; TasteNet-MNL 0.645 / 0.703. Mean VOT
train 2.33, SM 1.76, car 1.69 CHF/min.

### `mo_vns_assisted_spec` — Ortelli, Hillel, Pereira, de Lapparent, Bierlaire (2021), *JOCM* 39, 100285
Bi-objective VNS over variable inclusion, power transforms and segmentation;
10,395 obs, 80/20. OOS LL benchmark (9 params) −1,633.5 → OOS-optimal (19)
−1,515.6; BIC-optimal (27) −1,524.0.

### `dcm_ard` — Rodrigues, Ortelli, Bierlaire, Pereira (2022), *IEEE T-ITS* 23(4); arXiv:1906.03855; code github.com/fmpr/DCM-ARD
Tied automatic-relevance-determination priors over candidate representations
(log, Box-Cox, interactions) with variational inference; 10,692 obs, 70/30.
Test LL / accuracy: benchmark −2,535 / 0.644; ARD spec −2,421 / 0.677.

### `e_mnl_embeddings` — Arkoudi, Krueger, Azevedo, Pereira (2023), *TR-B* 175, 102783; arXiv:2109.12042
Categorical variables mapped to J-dimensional embeddings (one coordinate per
alternative, directly interpretable); EL-MNL adds a non-linear residual. Same
9,036 split as Sifringer. Test LL: MNL −1,433; E-MNL −1,231; EL-MNL (D = 5)
−1,097.7 (AIC 9,876) with ≈ half of L-MNL2's parameters.

### `rumnet` — Aouad, Désir (2026), *Management Science*; arXiv:2207.12877; code github.com/antoinedesir/RUMnet
Sample-average approximation of a RUM with K latent customer types × K product
perturbations, each a feed-forward net; universal approximation over the RUM
class. 10,719 obs, 10-fold. Test NLL / accuracy: MNL 0.842 / 0.623; latent-class
0.764 / 0.652; TasteNet 0.568 / 0.782; DeepMNL 0.592 / 0.771; RUMnet 0.561 /
0.793; random forest 0.523 / 0.776 (non-monotone in SM cost).

### `rumboost` / `functional_effects` — Salvadé, Hillel (2025), *TR-C* 170, 104897; arXiv:2509.18047
The TR-C RUMBoost paper benchmarks only LPMC; Swissmetro numbers come from the
functional-effects paper (10,692 obs, 70/15/15). Test cross-entropy: MNL (no
socio-demographics) 0.854; RUMBoost 0.786; DNN 0.746; GBDT 0.622; FI-RUMBoost
0.630; FIS-GBDT 0.614.

### `rum_nn` — Bagheri, Ghasri, Barlow (2025), *JOCM* 57, 100583; arXiv:2501.05221
Simulated error terms of arbitrary family inside a network; linear RUM-NN with
Gumbel reproduces MNL. 9,036 obs, 5-fold. Test LL / accuracy: MNL −1,155.8 /
66.2 %; DNN −1,148.4 / 70.7 %; nonlinear RUM-NN Pareto −993.7 / 72.2 %.

### `c_asu_dnn` — Haj-Yahia, Mansour, Toledo (2025), *TR-C* 171, 105014; arXiv:2306.00016
DNN and ASU-DNN with a domain-knowledge loss on pseudo-samples enforcing
monotonicity in own time / cost. 7,778 obs, 60/20/20. Test NLL / accuracy: DNN
0.68 / 70.1 %; C-DNN 0.70 / 69.1 % (0 % negative VOT vs 25–31 %); ASU-DNN 0.72 /
69.4 %; MNL 0.77 / 66.1 %.

### `ass_nn` — Hernández, Mouter, van Cranenburgh (2024), arXiv:2404.13198; code github.com/ighdez/ass_nn_paper
Alternative-specific non-cost sub-networks plus a cost sub-network with weights
shared across alternatives (fungibility of money). 9,036 obs, 80/20. Test LL:
linear MNL −1,447.6; log-linear MNL −1,403.2; ASS-NN −1,392.4; ASU-DNN −1,359.2.
Mean VTT (CHF/min) train / SM / car: ASS-NN 1.52 / 2.11 / 0.81.

### `diff_dcm` — Makinoshima, Mitomi, Makihara, Segawa (2025), *IEEE Access* 13; arXiv:2412.19403
Log/exp activations make each hidden node a monomial, giving a closed-form
symbolic utility. 9,036 obs. Test LL −1,326.8 / accuracy 67.6 % vs expert MNL
−1,434.7 / 64.3 %.

### `reslogit_plus` — Hasanzadeh, Wang, Rönnqvist, Badji, Verma (2025), *Transportation*, DOI 10.1007/s11116-025-10682-x
ResLogit (Wong & Farooq 2021 residual layers) with a genetic algorithm over
initialisation and hyperparameters. 6,768 obs. ρ² / test accuracy: MNL 0.23 / 0.62;
ResLogit 0.23 / 0.71; ResLogit Plus 0.26 / 0.76 (partly unverified).

### LLM-based (details in `llm_choice_survey.md`)
`llm_zero_shot` (Mo et al. 2023/2026); `llm_persona_embedding` (Liu, Li, Yin
2026, *Transportation Science*, DOI 10.1287/trsc.2025.0330); `llm_finetuned_litransmc`
(Alsaleh & Farooq 2025); `llm_athena` (Zhao et al. NeurIPS 2025);
`llm_generalizable_altsets` (Nishida et al. 2025); `fm_dcm_adapter` (Wang et al.
2026: TabPFN / Mitra probabilities inside a sign-constrained MNL, accuracy 76.5 %
with VOT 84.4 CHF/h preserved vs raw TabPFN 78.0 %, MNL 63.6 %).

### Papers on the initial list that do not estimate on Swissmetro
ASU-DNN (Wang, Wang, Zhao 2020: Singapore SP + R `TRAIN`); Wang, Mo, Zhao 2020
(Singapore + LPMC); ResLogit (Wong & Farooq 2021: Montréal); HAMABS (LPMC +
MTMC); van Cranenburgh et al. 2022 (discussion paper); Kim & Bansal 2024
(datasets unverified); Sfeir et al. 2025 (Apollo synthetic); Alt-GNN (LPMC +
Chicago).

## Summary

| Method | Paper | Best Swissmetro metric (test unless noted) |
|---|---|---|
| MNL / NL / MXL | Bierlaire 2020 | in-sample LL −5,331 / −5,237 / −5,215 (6,768 obs) |
| L-MNL | Sifringer et al. 2020 | LL −1,108 vs MNL −1,433 (1,802 test) |
| TasteNet-MNL | Han et al. 2022 | NLL 0.645, acc 0.703 vs MNL-C 0.698 / 0.678 |
| MO-VNS assisted spec | Ortelli et al. 2021 | OOS LL −1,515.6 vs −1,633.5 |
| DCM-ARD | Rodrigues et al. 2022 | LL −2,421, acc 0.677 vs −2,535 / 0.644 |
| E-MNL / EL-MNL | Arkoudi et al. 2023 | LL −1,097.7 |
| RUMnet | Aouad & Désir 2026 | NLL 0.561, acc 0.793 |
| RUMBoost / FIS-GBDT | Salvadé & Hillel 2025 | CEL 0.786 / 0.614 vs MNL 0.854 |
| RUM-NN | Bagheri et al. 2025 | LL −993.7, acc 72.2 % |
| C-ASU-DNN | Haj-Yahia et al. 2025 | DNN NLL 0.68 / 70.1 % |
| ASS-NN | Hernández et al. 2024 | LL −1,392.4 |
| Diff-DCM | Makinoshima et al. 2025 | LL −1,326.8, acc 67.6 % |
| LLM persona embedding | Liu et al. 2026 | wF1 0.683, JSD 0.021 |
| LiTransMC | Alsaleh & Farooq 2025 | wF1 0.6845 |
| ATHENA | Zhao et al. 2025 | acc 0.813 |
| FM-DCM adapter | Wang et al. 2026 | acc 76.5 %, VOT 84.4 CHF/h |

Comparability caveat: test LLs are on different splits and sizes (≈1,800-row
tests from the 9,036 screening; 1,604–1,607 from 10,692–10,719; LLM papers use
200–400-row balanced samples). This repo's Swissmetro protocol (9,036 rows,
per-respondent chronological 7/1/1 split over scenarios) matches none of them
exactly, so only the ordering of methods transfers.
