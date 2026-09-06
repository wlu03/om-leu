# Why information in a second model does not become realised ensemble gain — survey (2026-09-06)

Setting: p = (1 − π) p_s + π p_m, π fitted by ML on a validation split of 280–1,000 events;
sentence model informative (0.025 nats vs shuffled on LPMC) but the realised mixture gain is
smaller, and on Optima the validation π (0.2–0.4) overshoots the test-optimal π (0–0.18).
[from memory] marks items not opened at the primary source.

## 0. Diagnosis in formulas
- L(π) = Σ log((1 − π) p_s + π p_m). Slope at 0: L′(0) = Σ (p_m / p_s − 1): the mixture helps
  iff **E[p_m / p_s] > 1** on the target population — a likelihood-ratio-mean condition, not a
  "wins on x % of events" condition.
- Fisher information I(π) = E[((p_m − p_s) / p_π)²]; with error correlation 0.7–0.8, p_m ≈ p_s on
  most events so I is small and SD(π̂) ≈ 1/√(n I) ≈ 0.08–0.11 at n = 290.
- Expected regret from sampling noise alone ≈ 1/(2n) (interior) → 0.0017 nats at n = 290.
- Jensen bound: gain over p_s = π(E log p_m − E log p_s) + ½ π(1 − π) E[((p_m − p_s)/p)²]. The
  control comparison measures p_m's marginal information; the mixture only monetises the part
  orthogonal to p_s.
- Simulation calibrated to the setting (290 validation events, 29 respondents, 400 replicates):
  ML π regret 0.0024 nats, P(mixture worse than p_s) ≈ 0; OOF on 2,000 training events regret
  0.0004; selecting the best of 8–16 sentence variants on the same split inflates π̂ by
  +0.09–0.12. **The Optima loss (0.004–0.03) is too large for sampling noise: it needs
  validation → test shift in the relative quality of the sentence model and/or reuse of the
  split for other decisions.**

## 1. Sampling variance of stacking weights
Super-learner oracle inequality (van der Laan, Polley, Hubbard 2007; Dudoit & van der Laan
2005): additive O(log K / n) term ≈ 0.002–0.005 nats at n = 290. Yang 2000 (Ann. Stat.)
Theorem 1: progressive-mixture weights are within ln 2/(n+1) = 0.0024 nats of the better
component in KL (a no-harm bound). Juditsky, Rigollet, Tsybakov 2008: any selector, including
"best on a validation set", is √(log M / n) in the worst case; mirror averaging achieves
log M / n. Bates, Hastie, Tibshirani 2023 (JASA): naive CV intervals under-cover (31 % vs
10 %), poor until n ≈ 400. Yao et al. 2018: LOO stacking unstable when n < 5 × parameters;
use simplex constraint, regularisation, Bayesian-bootstrap weight uncertainty. Forecast
combination puzzle (Claeskens et al. 2016; Smith & Wallis 2009): estimated weights are biased
and variance-inflated; no guarantee of beating equal weights or the components.

## 2. Error correlation / diversity
Krogh & Vedelsby 1995 (E = Ē − Ā), Ueda & Nakano 1996, Brown, Wyatt, Tiňo 2005 (diversity =
covariance term). Wood et al. 2023 (JMLR): exact bias + variance − diversity decomposition for
Bregman losses with the loss's centroid combiner (for log-loss: the normalised geometric mean);
for the arithmetic mean the ensemble CE ≤ the weighted average of member losses, **not** ≤ the
best member. Yao, Pirš, Vehtari, Gelman 2022 Theorem 3: stacking gain over the best model ≥
max(L(1 − ρ)(1 − ε) − log K, 0) with ρ = sup_k Pr(model k locally best); if the numeric model
is locally best almost everywhere the guaranteed gain is zero however informative the second
model is. Diagnostic: E_test[p_m / p_s] must exceed 1.

## 3. Shrinkage and robust estimators
Breiman 1996 simplex constraint [from memory]; LeBlanc & Tibshirani 1996 shrink toward equal
[from memory]; Yao 2018 Dirichlet prior / Bayesian bootstrap; Yao 2022 hierarchical prior
(unshrunk pointwise weights degenerate to pointwise selection = overfitting); exponential
weights (Yang 2000) with the ln K/(n+1) guarantee. Simulation: MAP with 2–5 log-point penalty
toward 0, cluster-bootstrap 25th percentile, or a split-half "keep only if it helps the other
half" rule cost 0.001–0.005 nats in expectation when π* > 0 but bound the downside.

## 4. Probability-level vs log-linear pooling under miscalibration
Ranjan & Gneiting 2010 / Gneiting & Ranjan 2013: any linear pool of calibrated components with
positive weights is over-dispersed (uncalibrated); recalibrate after pooling (spread-adjusted /
beta-transformed pools; gains 0–0.05 nats in their examples, ≈ 0 when components are
near-ideal). Rahaman & Thiery 2021: temperature-scale after averaging. Wu & Gales 2021:
calibrated members do not give a calibrated ensemble. Wood et al.: for log-loss the natural
combiner is the geometric mean.

## 5. Validation–test shift with few clusters; cross-fitting
Wolpert 1992 / super learner: fit level-1 weights on out-of-fold level-0 predictions over
**all** training data (person-grouped folds), not one hold-out slice. Yao 2022 §3.3: under
covariate shift complete-pooling stacking is inconsistent; hierarchical weights sidestep it.
With ~30–150 respondents in validation the effective sample size for respondent-level
quantities is that number, not the event count.

## 6. LLM signal "informative but not usable"
Mo et al. 2023 (LLM helpful mainly in small samples); learning to defer (Mozannar & Sontag
2020), selective prediction (Geifman & El-Yaniv 2017): a deferral gate g(x) trained on OOF
data, bounded by −log ρ_X (Yao 2022 Theorem 4). No paper quantifies "informative but redundant
relative to a strong structured model" [unverified].

## Ranked fixes (290-event validation)
1. Cross-fitted π on person-grouped out-of-fold predictions over train + val; reporting split
   untouched. +0.002 (sampling) to +0.03 (if the loss is shift / double-dipping).
2. Nest every sentence-side selection inside the same folds; never select on the split used
   to fit π (removes +0.05–0.12 upward bias in π̂).
3. Pool-then-recalibrate (π, τ jointly on OOF data); also the log-linear pool. +0.002–0.01.
4. Conservative estimator (MAP toward 0, cluster-bootstrap check, or exponential weights with
   the ln 2/n guarantee): −0.001 to −0.004 expected, downside bounded at ≈ 0.
5. Raise diversity, not weight: condition the sentence model on p_s (residual targeting, show
   the numeric prediction in the prompt); gating alone 0–0.005 at n = 290.
Report alongside: E_test[p_m / p_s], the Jensen term, and a cluster-bootstrap SE on π̂.
