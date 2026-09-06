# Levers for making the LLM signal in OM-LEU 2 worth more — survey (2026-09-05)

Framing caveat: almost every paper reports an LLM predictor *standalone* vs MNL / RF / NN
in accuracy or F1; none measures the incremental log-likelihood of an LLM channel mixed
with a strong numeric model on warm respondents. Standalone gains of 5–15 pp shrink to a
few hundredths of a nat in that setting. The levers that survive inject information the
numeric model cannot have: respondent-level latent traits, retrieved behaviour of similar
people, and the LLM's slate-level judgement under sparse data. [unverified] as marked.

## Lever 1 — prompt / generation design
- Comparative reasoning guide (Mo et al., arXiv:2312.00819, Table 5): GPT-3.5 0.586 →
  0.552 without it (−5.8 % rel.); structured JSON input +4.4 %; **for GPT-4 all ablations
  cost < 1 %** — a weak-model effect.
- Chain-of-thought is neutral-to-negative: LiTransMC (arXiv:2507.21432) — prompt style +
  temperature explain < 2 % of F1 variance across 11 models × 3 datasets; CoT → direct
  raises mean F1 in 9/11 models. "Mind Your Step" (arXiv:2410.21333): CoT hurts
  implicit-pattern tasks by up to 36 pp. Forced JSON reasoning degrades reasoning
  (arXiv:2408.02442).
- Persona conditioning: Liu, Li, Yin (arXiv:2505.19003), Swissmetro, GPT-4o with learned
  persona-loading embeddings: wF1 0.683 vs zero-shot 0.543, few-shot 0.594, MNL 0.606;
  personas are inferred from each respondent's context–choice history.
- Retrieval-augmented prompts: Xu & Jiao (arXiv:2508.17527), PSRC: GPT-4o zero-shot
  0.711 → basic RAG 0.776 → class-balanced 0.794 → + cross-encoder re-rank 0.808 (MNL
  0.738); transit F1 0.396 → 0.574; naive RAG hurts o3.
- Likert pseudo-indicators: SAPA (arXiv:2509.18181) PR-AUC 0.141 → 0.248; ablation: Likert
  scores alone marginal (0.135 → 0.146), persona-derived *propensity* carries the gain
  (0.206), interactions add the rest.
- Counterfactual generation for consistency training: no paper; Gui & Toubia
  (arXiv:2312.15524) show attribute perturbations in prompts silently shift unspecified
  variables unless the design is "unblinded"; Ye & Yoganarasimhan (arXiv:2604.17267):
  LLM synthetic responses inside prediction-powered inference cut conjoint MSE 10–11 %.
- Self-consistency / temperature: < 2 % of variance (LiTransMC). Number-free prompts: no
  direct evidence [unverified].

Recipe: structured input, no CoT, slate-relative sentences ("saves 40 % vs the best other
option"), a retrieved block of 3–4 similar training respondents (class-balanced, train
split only), a short persona for warm respondents, and a slate-level judgement logged
after the sentences. Expected: ≤ 0.005 nats warm, 0.01–0.03 unseen with persona +
retrieval. Risks: retrieval leakage, double-counting warm history.

## Lever 2 — the LLM's own judgement (logits, verbalised probabilities, pairwise)
- Buchanan & Foster (arXiv:2607.26288): label-token logits admit an exact random-utility
  representation; transitivity 0.96–1.00 but IIA 0.12–0.92, strong first-position bias
  (use order-swapped differencing), label-reflexivity failures in some models.
- Utility Engineering (arXiv:2502.08640): Thurstonian utilities from order-swapped
  forced-choice log-probs; coherence rises with scale.
- Calibration: verbalised probabilities better calibrated than token log-probs for RLHF
  models (Tian et al., arXiv:2305.14975, ~50 % ECE reduction); all elicitation methods
  overconfident (Xiong et al., arXiv:2306.13063) — treat as a feature whose scale is refit
  on validation.
- Pairwise > direct scoring (PairS, arXiv:2403.16950); panels of small judges beat one
  large judge at 7× lower cost (arXiv:2404.18796).
- No paper compares logit features vs sentence embeddings inside a mixture with a strong
  numeric model — a gap OM-LEU 2 can fill.

Recipe: one forward pass per choice situation ("answer with the letter only"),
log-softmax over the J label tokens, averaged over two label orders; enter u_j = a ·
logit_j (+ b · log p_verbal_j) with a, b fitted on validation next to π. Expected:
0.01–0.03 nats unseen, 0.005–0.01 warm; cheaper than K × J generations; needs logit
access (vLLM / HF). Risks: IIA / position artefacts, contamination.

## Lever 3 — encoder side
- MTEB: Qwen3-Embedding-8B 75.2, 4B 74.6, 0.6B 70.7; NV-Embed-v2 69.8; gte-Qwen2-7B 70.7;
  OpenAI text-embedding-3-large 66.4; all-mpnet-base-v2 ≈ 57–58 [from memory].
- Embeddings do not carry numbers: Davies et al. (arXiv:2510.08009) — numbers are
  linearly recoverable (R² ≥ 0.95) but occupy a tiny share of variance and add noise;
  Deng et al. (arXiv:2509.05691) — 13 embedders cannot distinguish "grew 2 %" from "grew
  20 %". A better embedder will not fix OM-LEU's R² 0.55–0.90 probe; xVal / FoNE apply
  only to models trained with custom number tokens.
- LLM embeddings for tabular prediction: Kim et al. (arXiv:2410.07395) — no consistent
  in-distribution win over XGBoost, +5.4 pp under Y|X shift with 32 target samples; Tang
  et al. (arXiv:2411.14708) — size does not reliably help regression from embeddings.
- Contrastive fine-tuning on interactions: LLM2Rec (arXiv:2506.21579) R@10 +12.6 to
  +23.6 % in-domain vs BGE; LLM2Vec (arXiv:2404.05961) decoder hidden states as
  embeddings; Sun, Shen, Ton (ICLR 2025, arXiv:2411.04991) — a classification head on
  frozen embeddings beats a Bradley–Terry head and is more robust to label noise.
- PCA: Compiani et al. — 6 PCs explain 70–80 % of embedding variance.

Recipe: (i) Qwen3-Embedding-0.6B with an instruction prefix (re-encode only, ≤ 0.005
nats); (ii) LoRA-fine-tune the encoder with a chosen-vs-unchosen classification loss
(0.005–0.015 nats, ~1 GPU-hour, overfits Optima-size data); (iii) pool the generator's
own last-layer hidden states (free during generation, untested).

## Lever 4 — fine-tuning the generator / a small LM on choices
- LiTransMC: Gemma-3-12B QLoRA (r 32, loss masking, ≤ 5 epochs) on 100 respondents,
  single 12 GB GPU: Swissmetro wF1 0.6845 vs best untuned local 0.625, GPT-4o 0.543,
  GPT-4o + persona 0.683, NN 0.676, MNL 0.639; JSD 0.00025 vs 0.021 (best GPT-4o) vs 0.483
  (MNL).
- Centaur (arXiv:2410.20268): Llama-3.1-70B QLoRA on 10M choices: held-out participant NLL
  0.44 vs 0.58 base vs 0.56 best cognitive models; fine-tuned hidden states carry more
  choice-relevant content.
- Distillation of ranker logits into the sentence model / first-token distribution
  losses: no paper [untested].

Recipe: QLoRA 8–12B on the training split (respondent-level splits, early stop on val
NLL); use it as the logit source for Lever 2 and optionally as the generator. Expected:
the largest of any lever — 0.02–0.06 nats unseen, 0.01–0.02 warm; little on Optima. Cost
1–10 GPU-hours per dataset. Risk: contamination argument gets harder; changes the story
from "frozen LLM prior" to "another supervised model".

## Lever 5 — regimes where text is large; scaling with n
- Few-shot: Mo et al. n = 10 → NN 0.462 → NN + embedding 0.532; lift vanishes at
  n = 1000. TabLLM (arXiv:2210.10723): beats XGBoost at 0–32 shots, parity at 128–256,
  trees win at 512+; FeatLLM (arXiv:2404.09491) +10 % in few-shot with no per-sample calls.
- Shift / cold-start: Kim et al. +5.4 pp with 32 target samples; Xu & Jiao zero-shot
  transfer Seattle → Tacoma / NHTS: GPT-4o 0.81 / 0.86 vs RF 0.41 / 0.36, MNL 0.24 / 0.10
  (baseline numbers look like a coding artefact; LLM numbers plausible).
- Rich text: Compiani, Morozov, Seiler (arXiv:2503.20711) — reviews −23 % second-choice
  RMSE vs attributes −11.7 %; titles −4.3 %, descriptions −17 %; unstructured beats
  attributes in AIC in every Amazon category.
- Unseen alternatives (Nishida 2025): not locatable on arXiv [unverified].

Implication: evaluate at n = 25 / 50 / 100 / 250 / 1000 respondents, cold-start persons,
cross-dataset zero-shot, hold-out-a-mode. Expected gap vs numeric-only at n ≤ 100
respondents: 0.05–0.15 nats.

## Lever 6 — generator scale
Mo et al.: GPT-3.5 0.586, GPT-4 0.570, Llama-3.1-8B 0.551, 70B 0.540 — no monotone
scale effect; GPT-4's advantage is prompt robustness. Xu & Jiao: o3 zero-shot 0.783 vs
GPT-4o 0.711, but with the best RAG GPT-4o 0.808 ≥ o3 0.801 — conditioning closes the
gap. LiTransMC: model identity explains > 70 % of F1 variance; reasoning-distilled 7–8B
models are best zero-shot; "training recipe matters as much as parameter count".
Conclusion: Llama-3.3-70B is not the bottleneck; spend on conditioning and logits.

## Ranked recipes (expected nats over the current sentence channel, unseen / warm)
1. Slate-level forced-choice label logits (order-swapped) + verbalised probs as a utility
   term with scale fitted on validation: +0.01–0.03 / +0.005–0.01; cheaper than now.
2. Class-balanced retrieval of similar training respondents + LLM persona in the prompt:
   +0.01–0.03 / ~0; regeneration.
3. QLoRA fine-tune an 8–12B LM as logit source and generator: +0.02–0.06 / +0.01–0.02.
4. Respondent-level Likert pseudo-indicators + persona propensity as person weights / ICLV
   latents: +0.01–0.02 cold-start.
5. Fine-tune the encoder with a chosen-vs-unchosen classification head: +0.005–0.015.
6. Learning-curve and cold-start protocol (where the channel is worth 0.05–0.15 nats).
7. Number-free, slate-relative sentences, no CoT: +0.002–0.005 [unverified].
8. Better frozen embedder + PCA: ≤ 0.005.
Not recommended: CoT before sentences, forced JSON reasoning, self-consistency sampling,
scaling the generator, counterfactual-consistency training without an unblinded design.

## Most promising single change
Add the LLM's slate-level judgement directly: one retrieval-augmented, persona-conditioned
forced-choice prompt per choice situation, read as order-swap-averaged label
log-probabilities (plus a verbalised probability vector), entered into the mixture as a
utility index whose scale is fitted on validation next to π. It is slate-aware (sees all
alternatives), encodes the trade-off judgement directly rather than a five-sentence proxy,
is cheaper than K × J generations, degrades gracefully, and is the natural place to plug in
a fine-tuned 8–12B LM later. Pair it with the learning-curve / cold-start evaluation.
