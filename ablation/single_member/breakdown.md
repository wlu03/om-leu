**Problem:**  
Use one member instead of five: $\bar p_i(j) = p_1(j \mid E_i, z_i)$ instead of $\frac{1}{5}\sum_m p_m$. What does the ensemble contribute, and why does the same ablation appear in both the sentence-only and the mixture columns with different sizes?

**Explanation:**  
The five members differ only in their seed (initialisation, minibatch order, dropout masks). Averaging them in probability is a bagging-like reduction of the variance of a small model trained on few events. For a proper score the gain has a clean lower bound: by concavity of the logarithm,

$$
\log \frac{1}{5}\sum_m p_m(y_i) \;\ge\; \frac{1}{5}\sum_m \log p_m(y_i),
$$

so the ensemble NLL is never worse than the average member NLL, and the gap grows with the disagreement between members on the chosen alternative. The sentence-only $\Delta$ of this folder measures that gap plus the variance reduction relative to *one particular* member; a large value means the members are individually unstable.

Inside the mixture the same change is worth less. The structural channel does not change, and $\pi$ re-optimises around the single member: a noisier sentence channel receives a smaller weight, and the mixture's own averaging with $q_i$ reduces part of the member variance. This is the general pattern of `ablation/`: mixture deltas are smaller than sentence-only deltas, by a factor set by how much of the channel the fitted $\pi$ uses. It also gives the ensemble a fair reading against the plain-MLP baselines, which are averaged over five seeds as well; the design comparison in `plain_nn_sentences` is between two five-member ensembles, so the ensembling itself is not counted as part of the designed model.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
