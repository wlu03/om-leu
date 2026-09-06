# Behavioural validity, interpretability evaluation, theory and inference for OM-LEU 2 — survey (2026-09-05)

[unverified] / [from memory] mark items not opened at the primary source.

## 1. Behavioural validity of ML choice models

- Wang, Wang, Zhao 2020 *TR-C* 118 (arXiv:1812.04528): DNNs yield probabilities, shares,
  elasticities, MRS / VOT, welfare, but suffer hyperparameter sensitivity,
  non-identification and local irregularity; report aggregates over trainings and
  population, not single-observation values. Wang, Mo, Zhao 2021 *TR-B* (theory-based
  residual networks, DOI 10.1016/j.trb.2021.03.002) is the closest published analogue of
  our stage structure.
- Feng et al. 2024 *TR-C* 166 (arXiv:2404.14701): **strong regularity** = share of
  (n, j) with ΔP_nj / Δx_nd < 0 for own cost / time; **weak regularity** with a threshold
  ε; six gradient regularisers. Chicago MNL 0.998, plain DNN 0.888, regularised 0.990;
  OOD regularity +65 pp.
- Haj-Yahia, Mansour, Toledo 2025 *TR-C* 171 (arXiv:2306.00016): finite-difference
  monotonicity penalty; report % negative VOT (Swissmetro DNN 25–31 % → 0–0.1 %),
  probability-response curves under ±50 % attribute changes, market shares.
- Martín-Baos et al. 2023 *TR-C* 156 (arXiv:2301.04404): the full recipe — GMPCA +
  accuracy, market shares under policy scenarios (L1 share error), calibration curves
  along an attribute, VOT by numerical derivative (h = 5 % SD), invalid-VOT share (RF
  50 %, XGBoost 57 % on Optima), "inconsistent if 0 ∈ IQR"; **Optima cost coefficient
  flips sign if distance is omitted** — the cause of our unidentified Optima VOT.
- Salvadé & Hillel 2024 RUMBoost (arXiv:2401.11954): monotone piecewise-cubic Hermite
  splines (Fritsch–Carlson) fitted to piecewise-constant utilities so VOT = ratio of
  spline derivatives; PCUF (hEART 2025) piecewise-linear utilities avoid smoothing.
- Kim & Bansal 2024 *TR-B* 183 (lattice networks, partial monotonicity; WTP recovery
  [unverified]); van Cranenburgh et al. 2022 *JOCM* 42; Hillel et al. 2021 *JOCM* 38;
  Wang, Mo, Zheng, Hess, Zhao 2024 *TR-B* benchmark (arXiv:2102.01130).

**Audit table to compute (2–3 days for a harness):** strong / weak regularity, % negative /
invalid VOT, elasticity ranges vs published, market-share counterfactuals, availability
leak E[Σ_{k∉K} P_k], ECE after temperature scaling, seed stability — each for the RUM
part, RUM + residual, and the mixture under fixed-q (LLM probabilities not re-queried)
and re-queried-q protocols.

## 2. Interpretability evaluation

- Ground-truth recovery on synthetic data is the gold standard: TasteNet (Han et al.
  2022) recovers nonlinear tastes; Salvadé & Hillel 2025 (arXiv:2509.18047) recover
  individual intercepts with MAE 0.04 vs 0.29 for normal random intercepts; Kim & Bansal
  for WTP.
- Leakage tests: L-MNL keeps interpretable and learned parts on disjoint inputs
  (Sifringer et al. 2020); Sifringer & Alahi 2023 (arXiv:2312.14724) show the neural part
  replicates tabular information when inputs overlap, biasing structural parameters. Wang
  et al. 2026 Prop. 2 formalises the same failure. Direct threat to our residual (it
  takes time / cost) and to the LLM channel.
- Faithfulness tests transferable from XAI: label / weight randomisation sanity checks
  (Adebayo et al. 2018 [from memory]); ROAR remove-and-retrain (Hooker et al. 2019,
  arXiv:1806.10758); adversarial attention (Jain & Wallace 2019); concept-bottleneck
  intervention (Koh et al. 2020), concept leakage (Mahinpei et al. 2021,
  arXiv:2106.13314), "do CBMs learn as intended?" (Margeloiu et al. 2021,
  arXiv:2105.04289); prototype explanations in choice (Alwosheel, van Cranenburgh, Chorus
  2021 *TR-C*).

**Tests for OM-LEU 2:** tastes — synthetic recovery, seed stability, collapse diagnostic
(effective rank of head outputs; between-person vs between-seed variance), VOT(s)
distribution vs lognormal mixed logit. π — sentence shuffle (done: π → 0 expected),
label-randomised LLM, counterfactual sentence edits (directional agreement of ΔP_LLM and
ΔP), **latent-type predictiveness on panel data** (posterior type from early choices
predicts LLM-vs-RUM advantage on later choices — the real test of "a latent type that
follows the LLM"), person-level π(s) vs constant π by LR test. Residual — regress the
boosted residual on time / cost and report the total-utility VOT shift.

## 3. Theory

Verified anchors: McFadden & Train 2000 Theorem 1 (mixed MNL approximates any RUM);
Aouad & Désir (RUMnet Prop. 1: finite MNL mixtures approximate any RUM; every RUMnet is
a RUM; Prop. 2 generalisation bound); Wang, Sun, Li, Fan, Zhuang 2026 (arXiv:2606.26432)
Prop. 1 (fixed-q two-stage keeps MRS exact for attributes only in the structural part)
and Prop. 2 (joint training non-identified when cost is recoverable from q); Vij & Walker
2016 (ICLV reduced forms — our LLM channel is not an ICLV unless a structural link s_i →
latent → sentences and utility is specified); van der Laan, Polley, Hubbard 2007 Super
Learner oracle inequality (consistency of validation-stacked convex weights); Farrell,
Liang, Misra 2021 *Econometrica* (structural parameters as DNN functions of covariates,
automatic influence functions).

**Propositions we can plausibly prove** (V = A_j(s) + Σ β_k(s) x_k + u, residual r_j,
P^R = softmax(V + r), P = π P^L + (1 − π) P^R):

1. *MRS / VOT sandwich.* With β_t, β_c < 0 and r_j non-increasing in own time and cost
   with slope bounds L_t, L_c: |β_t| / (|β_c| + L_c) ≤ VOT_j ≤ (|β_t| + L_t) / |β_c|;
   equality VOT_j = β_t / β_c iff time and cost are excluded from r_j. Consequence: report
   total-utility VOT or exclude time / cost from the residual.
2. *Law of demand preserved.* Own-attribute monotone residual + own-alternative inputs ⇒
   ∂P^R_j / ∂c_j < 0 and cross ≥ 0; strong regularity = 1 by construction; under fixed q
   the mixture inherits it, under re-queried q no guarantee.
3. *Semantic damping of elasticities.* Under fixed q, E^mix_j = w_j E^R_j with
   w_j = (1 − π) P^R_j / P_j ∈ [0, 1]: π is "the fraction of policy responsiveness
   attributed to the numeric utility".
4. *Identification and inference for π.* ℓ(π) is strictly concave on [0, 1] when
   P^L ≠ P^R on a positive-measure set; π̂ is a one-dimensional M-estimator with
   I(π) = Σ (LR_n − 1)² / (1 + π(LR_n − 1))², LR_n = P^L_n(y_n) / P^R_n(y_n); score test of
   π = 0: Σ(LR_n − 1) / √Σ(LR_n − 1)² (cluster by person).
5. *Bounds on the semantic increment.* π E[log LR] ≤ ΔLL ≤ log(1 + π(E[LR] − 1)); ΔLL can
   be positive even when E[log LR] < 0 (complementarity).
6. *Mixture = two-class latent-class RUM* for a fixed choice set (class L with utility
   log P^L_j + Gumbel, class R with V + r + Gumbel); varying choice sets need P^L to
   satisfy Block–Marschak [unverified for LLM outputs].
7. *L2 person effects = MAP random intercepts* with σ_u² = 1/(2λ); unseen persons get
   u = 0 (Krueger et al. 2021 protocol); shrinkage ≈ T_i I_i / (T_i I_i + 2λ).

## 4. Inference

Person-cluster bootstrap B ≥ 100 (Wang 2026 used 10 + sign test) for VOT, elasticities,
paired ΔLL; π and its SE from Prop. 4 without retraining. DML / cross-fitting: Farrell,
Liang, Misra; Acharya, Hainmueller, Xu 2026 (arXiv:2604.10845: DNN mean preferences +
respondent-level empirical Bayes in a logistic RUM, valid inference for population-average
preferences — nearly our architecture). Bayesian: Villarraga & Daziano 2025
(arXiv:2505.18077: SGLD, 97 % coverage of MRS intervals vs 75–81 %); Bansal et al. 2020
*TR-B*, Krueger et al. 2020 (VB mixed logit). Conformal: Romano, Sesia, Candès 2020
adaptive prediction sets (no DCM-specific paper found).

## 5. Panel heterogeneity and VOT identification

Krueger, Bierlaire, Daziano, Rashidi 2021 *JOCM* protocol: predictive LL for new choices
of seen persons (conditional on individual posteriors) and for unseen persons; Salvadé &
Hillel 2025 follow it. Our functional intercepts + TasteNet tastes *is* a
functional-effects model; person random effects are fixed-effect-like and do not transfer.
VOT with random coefficients: Train & Weeks 2005 WTP-space; Scarpa, Thiene, Train 2008;
Daly, Hess, Train 2012 (normal cost coefficient ⇒ WTP without finite moments). Optima fix:
U = −exp(b(s)) [c + exp(a(s)) t] so VOT = exp(a(s)) > 0, include distance in the car
utility; if cost still lacks independent variation, say VOT is not identified.

## 6. Robustness and fairness

Temporal transfer: Fox, Daly, Hess, Miller 2014 *JTLU* (transfer index), Sikder et al.
2013; LPMC years 1–2 → 3. City-to-city / SP-to-RP: Ben-Akiva & Morikawa 1990 [from
memory]; LLM zero-shot transfer claims (Xu & Jiao 2025). Fairness: Zheng, Wang, Zhao 2021
*TR-C* (arXiv:2109.12422): FNR / FPR gaps by ethnicity, income, disability; add per-group
log-loss, ECE, regularity, and whether π differs by group.

## Ranked additions

1. Behavioural audit table (2–3 d). 2. Uncertainty: cluster bootstrap + closed-form SE /
score test for π (1–2 d). 3. π sanity / intervention suite (2 d). 4. Synthetic
ground-truth recovery + collapse diagnostics (2–3 d). 5. WTP-space reparametrisation and
Optima fix (1–2 d). 6. Seen / unseen-person protocol + mixed-logit and functional-effects
baselines with σ_u (2 d). 7. Temporal transfer on LPMC + fairness (2 d). 8. Theory
appendix with Props 1–7 (1–2 d). Optional: DML inference for average VOT (3–4 d),
conformal choice sets (0.5 d).

Key caveat: because the residual takes time and cost, Wang et al.'s exact-MRS result does
not apply to OM-LEU 2; either exclude them from the residual inputs or report the
total-utility VOT with the sandwich bound.
