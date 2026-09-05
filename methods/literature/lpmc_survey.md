# LPMC (London Passenger Mode Choice): literature survey

Compiled 2026-09-05 (web survey). "Unverified" marks details behind paywalls.

## Dataset

Hillel, Elshafie, Jin (2018), "Recreating passenger mode choice-sets for transport
simulation: a case study of London, UK", *Proc. ICE Smart Infrastructure and
Construction* 171(1):29–42, DOI 10.1680/jsmic.17.00018; Biogeme note
https://transp-or.epfl.ch/documents/technicalReports/CS_LPMC.pdf. 81,086 trips,
31,954 individuals, 17,616 households, LTDS April 2012–March 2015; walk / cycle /
PT / drive shares 17.6 / 3.0 / 35.3 / 44.2 %. Level of service from Google
Directions (walk, cycle, PT, drive under optimistic / pessimistic / best-guess
traffic), TfL fare model, WebTAG vehicle operating cost + congestion charge.
`driving_traffic_percent = (d_pes − d_opt) / d_typ`. Canonical split used by the
follow-ups: train 2012/13–2013/14 (54,766 trips), test 2014/15 (26,320), with
household-grouped CV folds.

## Papers and methods

### `gbdt_choice_set_lpmc` — Hillel, Elshafie, Jin (2018)
Two XGBoost probabilistic classifiers: raw LTDS variables only vs. plus the imputed
level of service. NLL objective, Hyperopt search, household-grouped 10-fold CV.
Holdout 2014/15: raw NLL 0.717 / accuracy 71.5 %; choice-set model NLL **0.651**,
accuracy ≈ 74.8 %. Argues for NLL and expected simulation error over accuracy.
Limitation: no behavioural indicators from trees.

### `hillel_thesis_ml_suite` — Hillel (2019), PhD thesis, Cambridge, DOI 10.17863/CAM.40710
Suite of ML classifiers vs MNL / nested logit; "assisted specification" where GBDT
partial dependence informs a parametric RUM. Linear-RUM VOT 8.73 £/h (PT) and
40 £/h (driving). Defines the LPMC_DC (13 params), LPMC_RR (54), LPMC_Full (100)
MNL specifications reused by later papers.

### `grouped_sampling_ml_review` — Hillel (2020 report; STRC 2021)
Shows trip-wise CV leaks the chosen mode via matching trips of the same person or
household. LPMC: GBDT grouped CV CEL 0.634 → external 0.651 / DCA 0.748; trip-wise
CV 0.467 (leaked) but external 0.730; logistic regression external 0.693 / 0.736.
Non-linear advantage is ≈ 0.04 CEL and ≈ 1 pp accuracy.

### `ml_systematic_review` — Hillel, Bierlaire, Elshafie, Jin (2021), *JOCM* 38, 100221
Review of 70 papers: trip-wise sampling, accuracy-only metrics, no external
validation and no hyperparameter search are the recurrent flaws.

### `hamabs` — Lederrey, Lurkin, Hillel, Bierlaire (2021), *JOCM* 38, 100226; code github.com/glederrey/HAMABS
Hybrid stochastic adaptive batch-size MLE: stochastic Hessian on a batch, batch grows
geometrically when the windowed LL stops improving, switch to BFGS above 30 % of the
data. LPMC_Full_L 486 s vs 4,758 s for Biogeme BFGS, identical LL.

### `mo_vns_assisted_spec` — Ortelli, Hillel, Pereira, de Lapparent, Bierlaire (2021), *JOCM* 39, 100285
Bi-objective variable-neighbourhood search over utility specifications (variable
inclusion, power transforms, segmentation); Pareto front of LL vs parameters.
Case study verified on Swissmetro (OOS LL −1,515.6 vs −1,633.5); LPMC use unverified.

### `lsh_dr_resampling` — Ortelli, de Lapparent, Bierlaire (2024), *JOCM*, DOI 10.1016/j.jocm.2023.100467
Locality-sensitive hashing to reduce the estimation sample; weighted MLE. LPMC
MNL-L (53 params) OSLL ≈ −0.704 / obs; 40 % samples keep parameters and drive VOT
(≈ 40–50 £/h) close to full-sample values.

### `gen_mnr_robit` — Krueger, Bierlaire, Gasos, Bansal (2023), *Statistics and Computing* 33(1):2
Multinomial robit (t-distributed kernel errors) and generalised MNR by Gibbs
sampling. LPMC subsample (10,820 train / 1,250 test): test LL MNP −960.9, MNR
−955.3, Gen-MNR −953.9; WTP in-vehicle time 19–22 £/h, out-of-vehicle 33–37 £/h.

### `ml_vs_rum_behavioural` — Martín-Baos, López-Gómez, Rodriguez-Benitez, Hillel, García-Ródenas (2023), *TR-C* 156, 104318; code github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
MNL, SVM, RF, XGBoost, NN, DNN with household-grouped CV; VOT from numerical
probability derivatives. LPMC test accuracy / GMPCA: MNL 72.54 / 48.85; XGBoost
**74.72 / 51.85** (CEL ≈ 0.657); NN 74.25 / 51.03. Tree-ensemble derivatives give
unusable VOT (IQR contains 0 for 41–43 % of observations).

### `rumboost` — Salvadé, Hillel (2025), *TR-C* 170, 104897; arXiv:2401.11954; code github.com/big-ucl/rumboost
Every linear parameter of a 62-parameter LPMC MNL replaced by an ensemble of
depth-1 trees, boosted on cross-entropy with: alternative-specific attributes enter
only their own utility, one attribute per tree, monotone constraints (negative on
time / cost / distance / congestion, positive on car ownership and licence for
driving). Leaf value γ = −Σ g / Σ h with g = p_i − y_i, h = p_i(1 − p_i). ASCs
recovered post hoc; PCUF fits monotone piecewise-cubic splines for VOT and
elasticities. Test CEL: MNL 0.7085; NN 0.6667; DNN 0.6735; LightGBM 0.6537;
RUMBoost-GBUV 0.6737; RUMBoost-PCUF 0.6730; nested logit 0.7091; nested RUMBoost
0.6731; FE-RUMBoost 0.6626. Train time per fold MNL 242 s, RUMBoost 6.5 s.
VOT: rail 2–5 £/h for short trips, driving peaks at 17.5 £/h.

### `rumboost_nested_cnl` — Salvadé, Hillel (2024), hEART 2024
NL and cross-nested probability functions in the RUMBoost loss; RUMBoost-CNL test
CEL 0.6716; CNL 0.7070.

### `functional_effects_rumboost` — Salvadé, Hillel (2025), arXiv:2509.18047
Intercepts and/or slopes as GBDT / DNN functions of socio-demographics
(V = g_i0(s) + Σ g_im(s) x_im). FI-RUMBoost test CEL 0.673; baselines without
socio-demographics: RUMBoost 0.824, MNL 0.841, GBDT 0.805.

### `paramboost` — Salvadé, Hillel (2026), arXiv:2604.18864
Boosted piecewise-cubic leaves (GAM shape functions with C²); LPMC CEL 0.6734
unconstrained, 0.6757 with all constraints.

### `nystrom_klr` — Martín-Baos, García-Ródenas, Rodriguez-Benitez, Bierlaire (2025), *Neurocomputing* 617, 128975
Kernel logistic regression with Nyström approximation (500 landmarks). LPMC test
accuracy 73.6 %, GMPCA 50.4 (XGBoost 74.7 / 51.9).

### `rum_nn` — Bagheri, Ghasri, Barlow (2025), arXiv:2501.05221
Neural network with simulated parametric error terms (Gumbel / Normal / …) and a
learned Cholesky correlation; reduces exactly to MNL or MNP. LPMC 5-fold: MNL LL
−13,449 / 64.2 %; nonlinear RUM-NN Normal −12,587 / 67.5 % (feature set differs
from other papers).

### `alt_gnn` — Zhou, Cheng, Zhuang, Hu, Bu, Wang (2025), arXiv:2509.07123
Alternatives as graph nodes; message passing generalises MNL (0 layers), ASU-DNN
and nested logit. LPMC 10k-trip subsample, 80/20: MNL LL −1,469 / 0.699; ASU-DNN
−1,413 / 0.726; Attention Alt-GNN −1,373 / 0.733.

### `tfm_choice_adapter` — Wang, Sun, Li, Fan, Zhuang (2026), arXiv:2605.26559 / 2606.26432
Tabular foundation models (TabPFN v2, Mitra) audited for monotonicity / VOT
violations; two-stage adapter embeds their probabilities in a sign-constrained MNL.
Full LPMC 70/15/15: MNL 69.9 % / NLL 0.725; raw Mitra 74.2 % / 0.668 (29 % monotone,
driving VOT negative); adapter 72.8 % / 0.697, 100 % monotone, VOT 1.8 / 16.6 £/h.

### `litransmc` — Alsaleh, Farooq (2025), arXiv:2507.21432
QLoRA-fine-tuned open LLMs on textualised choice situations; LPMC among three
datasets (100 respondents train, 200 test observations); pooled weighted-F1 0.6845.

### `llm_text_mode_choice` — Nishida, Ishigaki, Onishi (2025), TRR 2679(12)
Sentence-rendered alternatives → LLM outputs the mode word; one model across
alternative sets (LPMC among four datasets, unverified).

### `bmtm_dlp_recsys` — Lai et al. (2023), *IET ITS* 17(4):667–677
Recommender-style deep model with traveller-ID embeddings; reports 92.2 % accuracy
on LPMC, almost certainly trip-wise leakage (grouped studies top out ≈ 75 %).

### `biogeme_cs_lpmc` — Hillel (2019) Biogeme note; Bierlaire MOOC notebook
V_walk = ASC + β_t,walk dur_walking; V_cycle = ASC + β_t,cycle dur_cycling;
V_pt = ASC + β_cost cost_transit + β_access dur_pt_access + β_rail dur_pt_rail +
β_bus dur_pt_bus + β_int dur_pt_int; V_drive = ASC + β_t,drive dur_driving +
β_cost (fuel + ccharge) + β_traffic driving_traffic_percent. Grouped CV per-obs LL
≈ −0.82 to −0.86 for this small MNL vs −0.69 to −0.71 for logistic regression on
all features.

### Adjacent
- `datgan` / `cidatgan` — Lederrey, Hillel, Bierlaire (2022), arXiv:2203.03489 /
  2210.02404: DAG-structured tabular GAN for population synthesis on LPMC.
- `llm_prompt_learning` — Zhai et al. (2024), arXiv:2406.13558, withdrawn.
- Checked and not using LPMC: DeepLogit, Villarraga & Daziano, DeepHalo, RUMnet,
  the original ResLogit / TasteNet / L-MNL / ASU-DNN papers.

## Reference points (canonical split, household-grouped CV, test 2014/15)

| Model | test CEL | accuracy |
|---|---|---|
| MNL (62 params) | 0.7085 | ≈ 72.5 % |
| Nested logit | 0.7091 | |
| Cross-nested logit | 0.7070 | |
| Logistic regression, all features | 0.693 | 73.6 % |
| NN / DNN | 0.667 / 0.674 | 74.3 / 74.1 % |
| RUMBoost / FE-RUMBoost / RUMBoost-CNL | 0.673 / 0.663 / 0.672 | |
| XGBoost / LightGBM | 0.651–0.654 | 74.7–74.8 % |

Note: this repo's LPMC runs use a different protocol (600 people with ≥ 5 trips,
per-person chronological split), so numbers are not directly comparable to these
reference points; the ordering of methods is what transfers.
