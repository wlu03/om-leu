# Combining an LLM-derived signal with a structural choice model — survey (2026-09-05)

Notation: p = (1 − π) p_struct + π p_sem. "cached" = uses the existing sentences /
embeddings only. [unverified] as marked.

## 1. Input-dependent mixing (gates)
- Bayesian hierarchical stacking (Yao, Pirš, Vehtari, Gelman 2021/22, arXiv:2101.08954):
  w_k(x) = softmax(μ_k + Σ_m α_mk f_m(x)), α ~ N(0, σ_k), σ_k → 0 recovers scalar
  stacking. Theorem 1: the scalar weight ≈ Pr(model k is locally best); Theorem 4: the
  pointwise gain is bounded by −log ρ_X where ρ_X = max_k Pr(k locally best) — a gate
  pays only if no model dominates everywhere. Empirically ≈ 0.01 nats over scalar
  stacking on average, 0.1–0.2 in the tails; **unregularised per-cell stacking is worse
  than scalar at n ≤ 400**. Guidance: 2–5 standardised features, shrink to the scalar.
- Feature-weighted linear stacking (Sill et al. 2009, arXiv:0911.0460); stacking ≻ BMA
  in M-open (Yao et al. 2018, arXiv:1704.02030); learning to defer (Mozannar & Sontag
  2020), RouteLLM (arXiv:2406.18665), Hybrid LLM (arXiv:2404.14618); MoE for choice
  (Vallarino 2025, arXiv:2503.05800 [numbers unverified]).
Recipe (cached): gate features = cold-respondent flag, log n_obs(person), structural
margin / entropy, choice-set size; L2 toward the scalar π; nested CV for the penalty.
Expected +0.003–0.008 nats pooled, more on cold subsets.

## 2. Complementarity by construction
- L-MNL data separation (Sifringer et al. 2020); Sifringer & Alahi 2023
  (arXiv:2312.14724): the neural part re-learns tabular variables from the other
  modality, biasing β; architectural fixes inconclusive, **masking at the data source
  worked** — analogue: sentences written from prompts with numbers and alternative
  identity let p_sem re-learn both (our identity-only / shuffled controls).
- Residual hybrids: TB-ResNet (Wang, Mo, Zhao 2020, arXiv:2010.11644), RUMBoost;
  cross-fitted offsets per DML. Concept erasure: LEACE (Belrose et al. 2023,
  arXiv:2306.03819) — closed-form projection that provably removes linearly recoverable
  variables from embeddings with minimal change.
Recipes: (a) LEACE the embeddings on [attributes, alt one-hots, covariates] before the
sentence heads (cached); (b) train the sentence head against a cross-fitted structural
offset with shrinkage (cached); (c) number-free prompts (new generations). Measured p_sem
gain will fall — what survives minus the shuffled control is the identifiable LLM
contribution (guess: warm 0–0.005, cold 0.01–0.02).

## 3. LLM signal as prior rather than feature
- Prior elicitation: AutoElicit (arXiv:2411.17284), Selby et al. (arXiv:2402.07770), LLM
  Processes (arXiv:2405.12856); LLMs get signs right, miscalibrate widths (Riegler et al.
  2025 [abstract only]). Bandit warm-starts help up to ~30 % label corruption, harmful ≥
  50 % or under systematic misalignment (arXiv:2604.02527).
- Debiasing against a small human sample: PPI (Angelopoulos et al. 2023), PPI++
  (arXiv:2311.01453), DSL (arXiv:2306.04746), Ludwig, Mullainathan, Rambachan 2024
  (arXiv:2412.07031); **GAI** (Lu, Wang, Zhang, Zhang 2026, arXiv:2604.14575): LLM output
  as a feature with an orthogonal score, conjoint MAPE 19–32 % → 16–17 %, 50 labels ≈
  200 human-only even with 54 %-accurate LLM choices; **AEM** (Zhang, Li, Hortaçsu, Ye,
  Chernozhukov et al. 2025, arXiv:2510.25743): logistic correction on 10 % human data,
  MAPE −16.5 %, uncorrected LLM choices +0.1 %; Wang, Zhang, Zhang 2024
  (arXiv:2412.19363): two-stage soft-target distillation, naive pooling inconsistent.
Recipes (cached): (i) prior for cold respondents, β_i ~ N(μ + Γ s_i, Σ) with s_i a
low-dimensional projection of the person's sentence embeddings — cleanest
identification (compare Γ against shuffled s_i); plausibly 0.03–0.05 nats on cold
respondents [speculative]; (ii) GAI / AAE-style soft targets on unlabelled situations.

## 4. Semi-supervised / augmentation
Synthetic respondents are biased: Goli & Singh 2023; Twin-2K-500 (arXiv:2505.17479,
heterogeneity compression); GTA Berlin agents (arXiv:2601.16778). Corrections: Wang /
Zhang / Zhang, AEM, GAI; Ye & Yoganarasimhan 2026 (arXiv:2604.17267) allocate human
labels by rectification difficulty. Distilling step-by-step (arXiv:2305.02301). **No
paper distils an LLM ranker into a DCM or uses counterfactual-perturbation consistency in
DCMs** — an open slot. Recipe: student structural model with LL(human) + λ KL(g(x, z) ‖
p_struct) on counterfactual perturbations (cached only if sentences are
attribute-independent).

## 5. Latent-variable formulations
SAPA (arXiv:2509.18181): persona → attitude scores; ablation says the persona-derived
propensity and attitude × attribute interactions carry the gain, not the Likert scores.
Liu, Li, Yin 2025 (arXiv:2505.19003): persona loading P(Z_k | d_i) is a latent-class
membership model on covariates. No paper places LLM-scored indicators inside a formal
ICLV measurement model — a gap. Identification caveat: an indicator generated from the
same covariates the structural model sees is a noisy function of z_i; the only new content
is LLM world knowledge and interactions — which is why the warm gain is small and
control-recoverable. Recipe (cached): s_i as indicators, η_i = Γ z_i + ζ_i, V = βx + λη +
λ₂ (η ⊙ x), test λ against shuffled indicators; persona latent classes from clustered
embeddings.

## 6. Where mixtures pay
Ranjan & Gneiting 2010 JRSS-B: any linear pool of distinct calibrated forecasts is
uncalibrated → recalibrate after pooling; Rahaman & Thiery 2021 (arXiv:2007.08792):
ensembling calibrated members gives underconfidence, temperature-scale after averaging.
Log pooling optimal for log loss given calibrated experts (Neyman & Roughgarden 2022,
arXiv:2202.11219) but lets one confidently wrong expert veto; mixtures win when
components err on different subsets (hierarchical-stacking Theorem 1). Buchanan 2026:
LLM logits violate IIA across prompts → probability-level fusion is the safer choice.
Recipe (cached): temperature-scale each component on validation, fit the gate, final
temperature on the mixture; also fit the log-linear pool as comparator; **re-measure the
sentence gain against temperature-scaled p_struct** (part of a warm gain can be
calibration).

## 7. Transfer settings where the LLM is the only informative component
Xu & Jiao 2025 (zero-shot Seattle → Tacoma / NHTS: LLM 84–90 % vs RF 36–41 %, MNL 10–24 %
— baselines likely a schema mismatch); LiTransMC 100-respondent regime; Mo et al.
few-shot; TransMode-LLM (arXiv:2601.13763); PCMC-Net for unseen alternatives (Lhéritier
2019, arXiv:1909.11553); Nishida 2025 not locatable [unverified]. Recommended: (a)
leave-one-alternative-out (structural utility = attribute part only, sentence-derived
ASC prior); (b) new city with p_sem frozen and the gate refit on ≤ 100 respondents; (c)
few-shot curves at 10 / 100 / 1000 respondents — always with identity-only and shuffled
controls.

## Ranked recipes
1. Hierarchically shrunk gate π(x) on cross-fitted pointwise log-densities (cached).
2. Calibrate-then-mix-then-recalibrate; re-baseline the sentence gain (cached).
3. LEACE-orthogonalised embeddings + cross-fitted offset for the sentence head (cached).
4. LLM-informed prior for cold-start person parameters (cached).
5. ICLV with sentence-derived indicators and attitude × attribute interactions (cached).
6. Persona-anchored latent classes (cached; new generations for persona text).
7. Teacher–student distillation with cross-fitted correction on counterfactuals.
8. Transfer benchmark suite with all controls.
Single most promising change: the covariate-dependent gate shrunk toward the scalar π,
with calibration first; recipe 3 makes the reported contribution identifiable.
