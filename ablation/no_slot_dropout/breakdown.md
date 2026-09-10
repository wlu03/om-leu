**Problem:**  
Train the members without sentence-slot dropout (each of the $K$ slots masked out of the attention with probability 0.15 during training, never all five). Why is this regulariser in the model, and what does its removal show?

**Explanation:**  
Slot dropout is a data augmentation: at each step the member sees a random subset of the five sentences for each alternative, with the attention renormalised over the survivors,

$$
\alpha_{ijk} = \frac{m_{ijk}\exp(v^{\top} H_{ijk})}{\sum_{k'} m_{ijk'}\exp(v^{\top} H_{ijk'})}, \qquad m_{ijk} \sim \mathrm{Bernoulli}(0.85),
$$

so the pooled vector cannot rely on any one slot being present. It discourages the member from depending on a single sentence (for instance the financial one, which is the most informative on LPMC) and encourages it to spread the evidence over the axes, which is also what makes the per-axis ablations meaningful: without it the removal of one axis would hit a member that had never seen that axis missing.

Its predictive value is an empirical question. Members are early-stopped on validation NLL, and with a 32-dimensional projection and linear heads they are not very prone to overfitting; the dropout may be redundant with weight decay. If the sentence-only $\Delta$ is near zero the regulariser is harmless and can be kept for the robustness reason above; if it is negative (no dropout better) the augmentation removes information that a small model needs; if it is positive, the members do overfit to individual slots and the dropout is useful. With mean pooling (`no_salience`) the mask acts on the mean instead, so the two folders are not additive.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
