**Problem:**  
Write down the complete sentence channel of OM-LEU 2 as it is trained and evaluated in `ablation/`, state what the *sentence-only* and *mixture* scores measure, and explain why every other folder in `ablation/` is a one-term change of this model.

**Explanation:**  
For event $i$ (person $p(i)$, covariates $z_i \in \mathbb{R}^P$) and alternative $j$ the frozen LLM writes $K = 5$ first-person consequence sentences, one per axis (cost, time, comfort, access, reliability), and a frozen encoder maps each to $e_{ijk} \in \mathbb{R}^{d}$, $d = 768$. One member of the sentence model computes

$$
\begin{aligned}
H_{ijk} &= \mathrm{LN}(W e_{ijk}), \qquad W \in \mathbb{R}^{r \times d},\; r = 32 &&\text{(projection)}\\
\alpha_{ijk} &= \frac{\exp(v^{\top} H_{ijk})}{\sum_{k'} \exp(v^{\top} H_{ijk'})}, \qquad h_{ij} = \sum_{k} \alpha_{ijk} H_{ijk} &&\text{(salience attention)}\\
A_{ijm} &= a_m^{\top} h_{ij} + b_m, \qquad m = 1, \dots, M = 5 &&\text{(attribute heads)}\\
w(z_i) &= \operatorname{softmax}\big(\mathrm{MLP}_w(z_i)\big) \in \Delta^{M} &&\text{(person weights)}\\
\tilde V_{ij} &= \sum_{m} w_m(z_i)\, A_{ijm}, \qquad p(j \mid E_i, z_i) = \operatorname{softmax}_j(\tilde V_{ij}). &&
\end{aligned}
$$

The member is trained on the training split by minimising

$$
\mathcal{L} = -\log p(y_i \mid E_i, z_i) \;+\; \lambda\, \Big[-\log \operatorname{softmax}_j\big(c^{\top}\bar H_{ij}\big)_{y_i}\Big], \qquad \bar H_{ij} = \tfrac{1}{K}\sum_k H_{ijk},\; \lambda = 0.5,
$$

whose second term is a linear probe on the mean projected sentence (an InfoNCE / conditional-logit contrast of the chosen against the unchosen alternatives), with sentence-slot dropout (each $k$ masked with probability 0.15, never all) and early stopping on validation NLL. Five members with different seeds give

$$
\bar p_i(j) = \frac{1}{5}\sum_{m=1}^{5} p_m(j \mid E_i, z_i),
$$

and the full model mixes this with the structural channel $q_i(j) = \operatorname{softmax}_j(a\,U_{ij})$ of stages 1–2:

$$
p_i(j) = (1 - \pi)\, q_i(j) + \pi\, \bar p_i(j), \qquad \pi = \sigma(\gamma),
$$

with $(\gamma, \log a)$ fitted by L-BFGS on the validation split (`pi=val`) or on out-of-fold predictions (`pi=oof`).

Two numbers are reported for every folder. The **sentence-only NLL** is $-\frac{1}{n}\sum_i \log \bar p_i(y_i)$ on the test events: the sentence channel judged on its own, so it depends only on the sentences and on the sentence model, never on stages 1–2 or on $\pi$. The **mixture NLL** is $-\frac{1}{n}\sum_i \log p_i(y_i)$: what the change is worth inside the full system, where the fitted $\pi$ decides how much of the channel is used. A design change can move the first number a lot and the second very little, because the mixture weight re-optimises around the weaker channel; both are needed to read an ablation.

Each other folder overrides exactly one term above (`CONFIG` in its `model.py`): the person weights $w(z_i)$, the attention $\alpha$, the number of heads $M$, the projection $W$, the probe $\lambda$, the slot dropout, the number of members, or the sentence model as a whole (the plain-MLP baselines). Because everything else, including the sentences, the embeddings, the structural channel, the seeds and the test events, is held fixed, the per-event differences are paired, and the reported $\Delta$ is the mean paired $\Delta$NLL with a bootstrap interval over events.
