**Problem:**  
The plain MLP of `plain_nn_sentences` averages the $K$ sentence embeddings, which erases which axis each sentence came from. Feed instead the concatenation $[e_{ij1}; \dots; e_{ijK}] \in \mathbb{R}^{K d} = \mathbb{R}^{3840}$ so that slot identity is kept. What does the comparison with the mean-pooled baseline and with the designed model tell us?

**Explanation:**  
Because the sentences are generated one per axis in a fixed order, position $k$ *is* the axis label. Mean pooling, $\phi_{ij} = \frac{1}{K}\sum_k e_{ijk}$, is invariant to permuting the slots, so the mean-pooled MLP cannot learn that the financial sentence should be read differently from the reliability sentence; it can only respond to the average content. The flat input

$$
\phi^{\text{flat}}_{ij} = [e_{ij1}; e_{ij2}; \dots; e_{ijK}], \qquad u_{ij} = \mathrm{MLP}(\phi^{\text{flat}}_{ij}),
$$

lets the first layer $W_1 \in \mathbb{R}^{128 \times 3840}$ apply a different linear map to every slot, which is strictly more expressive than the mean (set all five slot blocks equal to recover it). The cost is five times more first-layer parameters on a few thousand training events, which the same weight decay and early stopping must absorb.

The designed model is intermediate between the two. Its salience attention $\alpha_{ijk}$ is also permutation-equivariant in the slots, but the per-axis ablations (`docs/llm_signal_research.md`) show that the members use the axes unequally, so the attention learns the axis from the content of the sentence rather than from its position. If the flat MLP beats the mean-pooled MLP by a large margin, slot identity is informative and a positional signal could be added to the designed model at little cost; if it does not, the average content already carries what a plain network can use, and the designed model's advantage over both baselines is not about the axes at all. The sentence-only paired $\Delta$ against `full_model` answers the second question, the difference between this folder and `plain_nn_sentences` the first.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
