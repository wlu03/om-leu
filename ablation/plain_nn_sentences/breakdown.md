**Problem:**  
Replace the designed sentence model by an ordinary feed-forward network on the same LLM consequences: for each alternative the mean of its $K$ sentence embeddings is fed to an MLP that outputs one utility, and the choice is a softmax over the alternatives. Nothing else changes. What is this baseline, what does it hold fixed, and how do its two scores decompose the improvement into "from the LLM" and "from the designed model"?

**Explanation:**  
The baseline member is

$$
\phi_{ij} = \frac{1}{K}\sum_{k=1}^{K} e_{ijk} \in \mathbb{R}^{768}, \qquad u_{ij} = \mathrm{MLP}(\phi_{ij}) = W_3\,\sigma\!\big(W_2\,\sigma(W_1 \phi_{ij})\big), \qquad p(j \mid E_i) = \operatorname{softmax}_j(u_{ij}),
$$

with hidden widths $128$ and $64$, ReLU, dropout $0.2$, trained by the same cross-entropy, optimiser, early stopping and weight decay as the designed members, and averaged over the same five seeds. It sees exactly the same sentences and embeddings as OM-LEU 2. It has no projection-with-probe, no attention over sentences, no attribute heads, no person input and no structural utility: it is the direct meaning of "feed the LLM output to a neural network".

Relative to the designed member, the baseline replaces the whole map $E_{ij} \mapsto \tilde V_{ij}$; the mean over $k$ discards which axis a sentence belongs to, and the absence of $z_i$ means that two people with the same sentences get the same probabilities. Everything the designed model adds — the low-rank contrastive space, the salience weights, the head decomposition, the person weights — is removed at once, so the sentence-only gap

$$
\Delta_{\text{design}} = \mathrm{NLL}_{\text{sent}}(\texttt{plain\_nn\_sentences}) - \mathrm{NLL}_{\text{sent}}(\texttt{full\_model})
$$

is the value of the designed model *on the same information*. Because the sentences are identical, no part of this gap can come from the LLM.

The second score completes the decomposition. Mixing the plain MLP with the structural channel through its own fitted $\pi$ gives $\mathrm{NLL}_{\text{mix}}(\texttt{plain\_nn\_sentences})$, and

$$
\underbrace{\mathrm{NLL}(\texttt{no\_sentences}) - \mathrm{NLL}_{\text{mix}}(\texttt{plain\_nn\_sentences})}_{\text{what the LLM sentences add with no design}}
\;+\;
\underbrace{\mathrm{NLL}_{\text{mix}}(\texttt{plain\_nn\_sentences}) - \mathrm{NLL}_{\text{mix}}(\texttt{full\_model})}_{\text{what the designed sentence model adds}}
= \mathrm{NLL}(\texttt{no\_sentences}) - \mathrm{NLL}_{\text{mix}}(\texttt{full\_model}).
$$

Both terms are paired differences over the same test events and carry bootstrap intervals. A large first term and a small second term would mean the improvement is the LLM's; the reverse would mean it is the model's. The other `plain_nn_*` folders vary the inputs of the same MLP (slot order kept, $z_i$ added, all numeric inputs added, sentences removed) so that the person input and the numeric attributes can be credited separately.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
