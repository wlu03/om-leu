# LLMs for discrete choice / travel mode choice: literature survey (2023–2026)

Compiled 2026-09-05 from a web survey (arXiv abstracts/HTML, publisher records,
authors' pages). Where a paywall blocked the full text, numbers are marked
**unverified**. Groups: (A) LLM as predictor on Swissmetro / Optima / LPMC;
(B) LLM outputs embedded structurally in a random-utility model; (C) LLM as
synthetic respondent; (D) LLM as modelling assistant; (E) adjacent work.

## A. LLM as mode-choice predictor on Swissmetro / Optima / LPMC

### A1. `llm_zero_shot_reasoning` (+ `llm_embedding_classifier`)
Mo, Xu, Ma, Cho, Zhuang, Guo, Zhao (2023/2024). *Large Language Models for Travel
Behavior Prediction.* arXiv:2312.00819. No code.
Zero-shot prompting: task description, alternative attributes as a structured
dictionary, socio-demographics as narrative, a "domain knowledge guide", a
"comparative reasoning guide" (explicit % comparisons of cost/time), JSON output
with prediction + explanation. GPT-3.5, GPT-4, Llama-3.1 8B/70B, T=0. Hard labels,
no logprobs. Second framework: OpenAI text embeddings of the scenario as extra
features for MNL/RF/NN in small-sample regimes.
Data: Swissmetro, balanced 1,000 train / 200 test, 5 draws. Results: GPT-3.5
acc 0.586 / F1 0.572; GPT-4 0.570; Llama-3.1-70B 0.540; baselines (1k train)
MNL 0.605, RF 0.612, NN 0.635; with 10 train samples MNL 0.469 → +embeddings
0.494, NN 0.462 → 0.532. Removing the comparative guide costs GPT-3.5 −5.8 pp.
Limitation: hallucinated explanations; two tasks only.

### A2. `llm_persona_loading_fewshot`
Liu, Li, Yin (2024). *Can LLMs Capture Human Travel Behavior? Evidence and Insights
on Mode Choice.* SSRN 4937575. Zero-shot role prompting is misaligned; few-shot and
"travel behaviour persona loading" (LLM summarises revealed trade-offs into a persona)
"surpass traditional models when domain knowledge is incorporated". Dataset likely
Swissmetro (unverified); numbers unverified.

### A3. `llm_persona_embedding_loader`
Liu, Li, Yin (2025). *Aligning LLM with human travel choices: a persona-based
embedding learning approach.* arXiv:2505.19003.
Expert LLM writes a persona Z_k per base respondent from demographics + 9 observed
choices; a learnable 4-d linear embedding e(d; β) maps a new person to base personas
via cosine similarity + softmax (λ = 40/3); GPT-4o returns a hard label given the
loaded persona; β estimated by Monte-Carlo stochastic EM on
LL(β) = Σ_i log Σ_k P(Z_k | d_i) · 1[LLM(d_i, X_i, Z_k) = Y_i].
Data: Swissmetro 1,004 respondents / 9,036 records; 2,250 for persona inference,
200 to train the loader, 400 test. Results: MNL JSD 0.483 / macro-F1 0.474 /
wF1 0.606; zero-shot 0.216 / 0.407 / 0.543; few-shot 0.108 / 0.429 / 0.594;
proposed 0.021 / 0.556 / 0.683. Limitation: no NLL, no mixed-logit / NN comparison.

### A4. `llm_text_generalist_mode_choice`
Nishida, Ishigaki, Onishi (2025). *LLMs Predict Transportation Mode Choice Behavior
for a Variety of Alternative Sets.* TRR 2679(12), DOI 10.1177/03611981251352499.
One text-prompted model across alternative sets (LPMC, Optima, Swissmetro,
Netherlands); GPT-3.5/GPT-4 zero/few-shot, hard labels; "more versatile than MNL",
GPT-4 ≥ MNL on most sets; exact numbers unverified (paywalled).

### A5. `llm_panel_demo_prompting` (withdrawn)
Zhai, Tian, Li, Zhao (2024). arXiv:2406.13558 — withdrawn for a preprocessing
error. Narrative rows + zero-shot / similar-demo / panel-demo prompts to LLaMA-3 /
Gemma. Reported (invalid) LPMC acc 0.772 vs MNL 0.690; Optima 0.734 vs 0.640.

### A6. `llm_qlora_finetune_local` (LiTransMC)
Alsaleh, Farooq (2025). *Towards Locally Deployable Fine-Tuned Causal LLMs for Mode
Choice Behaviour.* arXiv:2507.21432. Eleven 1–12B open models, zero / random
few-shot / targeted few-shot, direct vs CoT, JSON output with rationale; Gemma-3-12B
fine-tuned with QLoRA and answer-token loss masking. Distribution metrics
(JSD, DistMAE) + Explanation Strength Index. Data: Swissmetro, Brightwater, LPMC
(100 respondents train / 200 obs test). Best wF1: Swissmetro 0.619 (targeted
few-shot), LPMC 0.663; LiTransMC 0.6845 / JSD 0.00025 on Swissmetro.

### A7. `llm_symbolic_utility_discovery` (ATHENA)
Zhao, Zhao, Du, Yang (2025). *Personalized Decision Modeling: Utility Optimization or
Textualized-Symbolic Reasoning.* arXiv:2511.02194. Stage 1: LLM proposes symbolic
utility expressions per demographic group, scored by MNL loss, evolved
(LLM-SR style). Stage 2: personalised semantic template refined with TextGrad; final
label sampled from the LLM given the template and f_g*(X_i). Data: Swissmetro
500 travellers × 2. Results: acc 0.813 / F1 0.766 vs MNL 0.610 / 0.389, XGBoost
0.708 / 0.705, few-shot LLM 0.682. Single seed.

## B. Foundation-model outputs embedded in a random-utility model

### B1. `fm_prob_embedded_mnl`
Wang, Sun, Li, Fan, Zhuang (2026). *Embedding Foundation Model Predictions in
Discrete-Choice Models with Structural Guarantees.* arXiv:2606.26432. Tabular
foundation models (TabPFN, Mitra), not text LLMs. V_k = βᵀφ_k(x) + g_k(q(x)) with
β_cost = −exp(θ) and q(x) the FM probability vector as a fixed feature; two-stage
estimation (β first, correction g second) keeps MRS = β_j/β_j' exact.
Data: Swissmetro (row-level 70/15/15), LPMC, IoT-Wearables. Results: +up to 12.8 pp
over MNL, 100 % monotone, VOT Swissmetro 84.4 CHF/h unchanged; a naive
feature-augmented MNL gains a bit more accuracy but VOT drifts to 678–1,508 CHF/h.

### B2. `llm_latent_variable_hybrid` (SAPA)
Sameen, Zhang, Zhao (2025). arXiv:2509.18181. Llama-3.1-8B writes personas and
scores seven theory-driven latent constructs (time / cost sensitivity, pro-car,
convenience, environmental concern, spontaneity, technology affinity), interacted
with trip attributes; LightGBM predicts ridesourcing choice. PSRC HTS, 58,954 /
15,052 trips. PR-AUC 0.141 → 0.248, F1 0.198 → 0.300. Closest thing to an
LLM-hybrid-choice model, but latent scores feed a GBM, not an ICLV.

## C. LLM as synthetic respondent

- `llm_synthetic_sp_vot` — Yan, Liu, Yin (2025/2026), arXiv:2507.22244 / TBS
  101245. GPT-4o, Gemini-2.5-pro, Claude-Sonnet-4 rank 13 alternatives in a
  768-cell factorial SP; rank-ordered logit VOT: GPT-4o $7.92/h, Claude $5.80/h
  (human 6.0–10.6), Gemini $14.37/h.
- `llm_synthetic_dce_audit` — Voltes-Dorta, Suau-Sanchez (2025), AIT 100034.
  23 LLMs as airline DCE respondents; mixed / latent-class logit WTP audit.
- `llm_survey_respondent_intertemporal` — Goli, Singh (2024), Marketing Science
  43(4). GPT-3.5/4 far more impatient than humans.
- `gpt_synthetic_consumer_wtp` — Brand, Israeli, Ngwe (2023/EC'24). Realistic WTP
  magnitudes, weak heterogeneity.
- `llm_conjoint_data_augmentation` — Wang, Zhang, Zhang (2024–26),
  arXiv:2412.19363. Consistent estimator pooling LLM + human conjoint data.
- `gpt_food_dce` — Califano, Caracciolo (2026), FQP 105930.
- `llm_survey_persona_agents` — Torkayesh et al. (2026), arXiv:2608.07519;
  `llm_guided_persona_survey` — Tzachristas et al. (2025), arXiv:2501.13955.

## D. LLM as modelling assistant

- `llm_mnl_specification_assistant` — Sfeir, Nova, Hess, van Cranenburgh
  (2025/JOCM 2026), arXiv:2507.21790. Twelve LLMs specify MNL utilities on Apollo's
  synthetic inter-city data; only agentic GPT-o3 estimated correctly.

## E. Adjacent predictors on other datasets

- `llm_rag_mode_choice` — Xu, Jiao (2025), arXiv:2508.17527 (PSRC 2023; GPT-4o +
  balanced RAG acc 0.808 vs MNL 0.738).
- `llm_feature_informed_prompting` — Zhang, Xu (2026) TransMode-LLM,
  arXiv:2601.13763 (NHTS).
- `llm_delay_choice` — Chen et al. (2024) DelayPTC-LLM, arXiv:2410.00052.
- `masked_lm_mode_choice` — Yang et al. (2024), TR-A 184, 104074.
- `llm_dual_agent_route_learning` — Liu et al. (2025), arXiv:2511.00993.

## Cross-cutting observations

1. Every Swissmetro LLM paper except B1 elicits hard labels; none uses logprobs,
   so likelihood-based comparison with MNL is absent in the literature.
2. Only B1 and A7 keep an explicit RUM utility; only B1 guarantees interpretable
   marginal rates of substitution.
3. SAPA (B2) is the nearest "LLM hybrid choice model", with LLM-scored latent
   constructs instead of survey indicators.
4. Reported Swissmetro accuracies are not comparable across papers (200-sample
   tests in A1/A6, 400 in A3, 500×2 in A7, row-level 70/15/15 in B1).

Relation to OM-LEU: OM-LEU keeps a utility with attribute heads, obtains
probabilities from a trained model rather than from letter tokens, and uses the
LLM only to generate outcome sentences (a mediating representation), which is
closest to B1's "embed the model's output as a feature under structural
constraints" and to B2's LLM-generated latent constructs.
