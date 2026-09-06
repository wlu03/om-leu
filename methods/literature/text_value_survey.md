# Where text / LLM information adds value in choice modelling — survey (2026-09-05)

Compiled for the gap analysis; effect sizes are as reported by the sources.
[unverified] marks details that could not be checked against the primary text.

## 1. Positive results (text beats numeric attributes)

- **Compiani, Morozov, Seiler (2026), RAND J. Econ., arXiv:2503.20711.** Book choice
  (10 alternatives, first + second choice, randomised prices). Mixed logit with random
  coefficients on PCA of text/image embeddings vs attribute-based mixed logit.
  Counterfactual second-choice RMSE: attributes −11.7 % vs plain logit; review text −23 %;
  titles 4.3 %, descriptions 17 %, reviews 23 %, images 7 %. Adding attributes to the
  review model gives zero-variance random coefficients: "attributes are subsumed by the
  unstructured data". → The strongest clean evidence is about *substitution patterns*
  with many alternatives and rich text, not first-choice fit on 3–4 numeric alternatives.
- **Nishida, Ishigaki, Onishi (2025), TRR.** LLM-based mode choice across four
  alternative sets; claimed value is cross-alternative-set transfer, no numbers in
  abstract.
- **Mo, Xu, Zhao (2023), arXiv:2312.00819.** Swissmetro: at full sample GPT-3.5
  zero-shot 0.586 < NN 0.635; at 10 examples NN + LLM embedding 0.532 vs NN 0.462. → The
  LLM is a few-shot prior, not incremental information at full sample.
- **SAPA (2025), arXiv:2509.18181.** PSRC ridesourcing (1 % base rate): LLM persona →
  scored latent constructs + propensity → LightGBM; PR-AUC 0.141 → 0.248. No
  shuffled / random-score control reported.
- **LABOR-LLM (Vafa, Athey, Blei et al.), arXiv:2406.17972.** Replacing occupation
  titles by codes degrades prediction: alternative *identity text* matters when the
  alternative space is large; with 3–4 modes ASCs already carry it.
- **TextTabBench (ICML 2025), arXiv:2507.07829.** 13 tables with genuine free-text
  columns: text improves 11/13 for every model (fraud 0.852 → 0.962, Spotify 0.663 →
  0.815); simple encoders beat BERT; only 11 of CARTE's 51 datasets are genuinely
  text-bearing.
- **Koloski et al. (2025), arXiv:2502.11596; Kolomenko et al. (2026), arXiv:2603.17737.**
  LLM row embeddings: +1.5 to +3 pp on categorical-heavy tables, a loss on the purely
  numeric one; "strongly depends on pipeline design"; concatenate, never replace.
- **Recsys:** text-based item encoders match ID embeddings only with end-to-end training
  (Yuan et al. 2023, arXiv:2303.13835); the value of text is cold-start / transfer
  (UniSRec 2022; LM prior for cold-start items, arXiv:2411.09065; LLM2Rec,
  arXiv:2506.21579); text ≈ full multimodal, images add little (arXiv:2508.07399);
  review text helps hotel recommendation (arXiv:2601.02362).
- **Conjoint / market research:** Brand, Israeli, Ngwe (SSRN 4395751): GPT WTP realistic
  at aggregate level, weak heterogeneity; Wang, Zhang et al. (arXiv:2412.19363): LLM
  data as augmentation saves 25–80 % of human data, substitution is biased; Generative
  Augmented Inference (arXiv:2604.14575) and Agentic Economic Modeling
  (arXiv:2510.25743): LLM outputs as auxiliary signal debiased on a small human sample;
  uncorrected LLM choices *increase* error; own-review RAG predicts a person's pairwise
  choices at 87.7 % (arXiv:2604.22756); hotel conjoint audit of 12 LLMs
  (arXiv:2606.16344).
- **Cold-start demand:** VISUELLE (arXiv:2109.09824) and follow-ups: unstructured signals
  matter when there is no sales history at all. Semantic insurance pricing
  (arXiv:2606.29371): gains only in data-scarce regimes, prompt-fragile.

## 2. Null results (when LLM features add nothing)

Koloski 2025 (numeric-only table loses); Grinsztajn et al. 2022 (arXiv:2207.08815, trees
win at ~10k rows, deep nets hurt by uninformative directions); TabLLM (arXiv:2210.10723)
and FeatLLM (arXiv:2404.09491) only in few-shot; Mo et al. at full sample; Kolomenko
2026 pipeline dependence; embedding instability (Frontiers Bioinformatics 2026: 32–65 %
of identical embedding calls differ); "Can LLMs assist choice modelling?"
(arXiv:2507.21790: frontier models do best with the data dictionary only); token noise
in LLM choices (arXiv:2404.01332); LLM priors harmful beyond ~40 % misalignment
(arXiv:2604.02527).

**Boundary:** the null zone is numeric-only attributes × small fixed slate × n in the
thousands × warm alternatives — exactly Swissmetro / Optima / LPMC.

## 3. Designs that make the LLM contribution identifiable

1. Held-out alternative / cross-alternative-set transfer (Nishida; LiTransMC
   arXiv:2507.21432; RAG mode choice arXiv:2508.17527: MNL 0.738 vs GPT-4o+RAG 0.808,
   zero-shot transfer to other cities 0.87–0.90 where MNL collapses to 0.10–0.24).
2. Second-choice / diversion-ratio validation with random coefficients on embedding PCs
   (Compiani).
3. Number-free / text-only prompts, so the sentence model cannot re-encode the numbers.
4. LLM-scored latent constructs as ICLV indicators **with** shuffled / random controls.
5. Person-level text (own reviews) for individual heterogeneity.
6. LLM as a prior debiased on a small human sample; report human-observation equivalents.
7. Counterfactual sentence generation — no paper found (a design opening).
8. Persona-conditioned alignment (arXiv:2505.19003).

## 4. Contamination

Swissmetro and Optima ship inside the `biogeme` pip package
(`src/biogeme/data/data/swissmetro.dat`, `optima.dat`) and appear in hundreds of public
repos (GitHub code search: 219 files with PURPOSE/LUGGAGE/SM_HE, 836 with
SM_CO/TRAIN_TT); LPMC processed copies are public too. None of the LLM mode-choice
papers report a memorisation check. Protocol: Bordt et al. (COLM 2024,
arXiv:2404.06209; tool github.com/interpretml/LLM-Tabular-Memorization-Checker) header /
row / feature-completion tests, perturbation (rescale / rename) test, a post-cutoff control
dataset, and a canary test (ask for Swissmetro shares or MNL coefficients without data).
Survey: arXiv:2502.14425.

## 5. Cost and reproducibility

Rarely reported. RAG paper: $0.0009–0.0028 and 451–623 tokens per prediction. LLM
stability (arXiv:2408.04667): up to 15 accuracy points across nominally deterministic
runs; prompt formatting alone can move accuracy by up to 76 points (arXiv:2310.11324).
Reporting template: tokens and $ per event, model id and date, temperature, number of
generation seeds with between-seed std, spread across ≥ 3 prompt paraphrases, embedding
determinism.

## 6. Public datasets where text should matter

Amazon Reviews 2023 (McAuley Lab; item text + reviews); TextTabBench's 13 tables;
VISUELLE; TripAdvisor / Yelp reviews; Expedia ICDM 2013 (numeric only [unverified]);
Trivago 2019 (tag lists, download offline [unverified]); Twin-2K-500 (arXiv:2505.17479);
MovieLens + TMDB descriptions; RecTour (request-only).

## Ranked settings for OM-LEU

1. Held-out alternative transfer on Swissmetro / LPMC (contamination check mandatory).
2. Amazon Reviews 2023 category choice with second-choice validation (Compiani design).
3. Cold-start items with descriptions and no history.
4. Cross-city / cross-survey zero-shot.
5. Few-shot regime (10–200 events) on the existing datasets; report human-observation
   equivalents of the sentence prior.
6. Person-level latent constructs with proper controls (SAPA re-analysis).
7. Number-free sentence arm on the three mode-choice datasets.
8. Hotel choice with joined descriptions / reviews (position confound).

## Boundary statements the paper can make honestly

- On a fixed 3–4 alternative slate with numeric attributes, outcome sentences add ≤ 0.01
  nats beyond a numeric RUM and the gain is not separable from ensembling; this matches
  independent null results for LLM embeddings on numeric-only tables.
- Positive text effects in the literature come from rich item text with many
  alternatives (substitution patterns), cold-start / unseen alternatives or cities, and
  few-shot samples — regimes we have not tested.
- Any LLM-feature paper on small slates should include shuffled-text, random-embedding and
  identity-only controls (SAPA, Koloski, LiTransMC do not).
- Swissmetro and Optima are almost certainly in every frontier model's pretraining corpus.
- LLM pipelines are not deterministic; report seed and prompt spreads and cost per event.
