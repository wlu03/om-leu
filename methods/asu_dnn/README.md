# ASU-DNN (alternative-specific utility deep neural network)

**Papers.** Wang, Wang, Zhao (2020), "Deep neural networks for choice analysis:
architecture design with alternative-specific utility functions", *TR-C*
112:102470, arXiv:1909.07481 (Singapore SP + R `TRAIN` data; +2–3 pp over a
fully connected DNN, +8 pp over MNL / NL). Re-implemented on Swissmetro by
Haj-Yahia, Mansour, Toledo (2025, *TR-C* 171: test NLL 0.72 / 69.4 %; with
domain-knowledge constraints C-ASU-DNN 0.73 / 67.8 %) and Hernández, Mouter,
van Cranenburgh (2024, arXiv:2404.13198: test LL −1,359 vs linear MNL −1,448);
on LPMC by Zhou et al. (2025, Alt-GNN: ASU-DNN LL −1,413 vs MNL −1,469 on a 10k
subsample).

**Model.** One sub-network per alternative that sees only that alternative's
attributes and the person's characteristics, so no cross-alternative attribute
enters a utility (the DNN analogue of a RUM specification):

    V_j = f_j([x_j, z]),   f_j separate MLPs,   P = softmax(V)

Economic information (VOT, elasticities) is obtained numerically from
probability derivatives; Wang et al. report that these can violate sign
expectations for a share of individuals.

**Implementation.** `model.py::ASUDNN` (torch): J MLPs of 2 × 64 ReLU on
standardised attributes + covariates, dropout 0.1, Adam lr 1e-3, batch 256,
weight decay 1e-4, early stopping on validation NLL.
