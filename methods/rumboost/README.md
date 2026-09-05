# RUMBoost (gradient-boosted random utility model)

**Papers.** Salvadé, Hillel (2025), "RUMBoost: gradient boosted random utility
models", *TR-C* 170:104897, arXiv:2401.11954, code github.com/big-ucl/rumboost;
Salvadé, Hillel (2024, hEART) nested / cross-nested RUMBoost; Salvadé, Hillel
(2025, arXiv:2509.18047) functional-effects RUMBoost; Salvadé, Hillel (2026,
arXiv:2604.18864) ParamBoost.

**Model.** Every linear parameter of a RUM utility is replaced by an ensemble
of regression trees, boosted directly on the multinomial-logit cross-entropy:

    V_j = Σ_k f_jk(x_jk),      f_jk = Σ_t tree_t
    g_j = P_j − 1[y = j],      h_j = P_j (1 − P_j)          (RUM gradient / hessian)
    leaf value γ = −Σ g / Σ h

with three structural constraints: alternative-specific attributes enter only
their own utility, one attribute per tree (additive utility, no unspecified
interactions) and monotonicity (negative in time and cost). Utilities stay
interpretable as RUM utilities; the paper smooths the piecewise-constant
utilities with monotone cubic splines (PCUF) to read off VOT and elasticities.

**Literature results (LPMC, canonical split, test CEL).** MNL 0.7085; NN
0.6667; DNN 0.6735; LightGBM 0.6537; RUMBoost 0.6737 (PCUF 0.6730); nested
RUMBoost 0.6731; RUMBoost-CNL 0.6716; FE-RUMBoost 0.6626. Swissmetro (functional
effects paper, without socio-demographics): MNL 0.854, RUMBoost 0.786, GBDT
0.622; FIS-GBDT 0.614.

**Implementation.** `model.py::run` implements the algorithm directly on
LightGBM (one `Booster` per alternative updated with the RUM gradient each
round, `interaction_constraints` = one feature per tree, `monotone_constraints`
= −1 on time and cost, person covariates included in every utility as in the
paper's socio-demographic terms; lr 0.1, 31 leaves, early stopping on
validation NLL). The `rumboost` PyPI package is not used because its pinned
Biogeme build does not import under NumPy 2 in this environment. No PCUF
smoothing.
