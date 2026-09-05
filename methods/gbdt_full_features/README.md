# GBDT on the full feature vector (black-box ML reference)

**Papers.** Hillel, Elshafie, Jin (2018), *Proc. ICE SMIC* 171(1) — XGBoost on
LPMC with the imputed level of service: holdout NLL 0.651, accuracy 74.8 %;
Hillel (2020 / 2021) on household-grouped vs trip-wise sampling (GBDT external
CEL 0.651 vs logistic regression 0.693); Hillel, Bierlaire, Elshafie, Jin
(2021), *JOCM* 38 systematic review; Martín-Baos et al. (2023), *TR-C* 156
(XGBoost accuracy 74.7 %, GMPCA 51.9 on LPMC; tree derivatives give unusable
VOT); Salvadé & Hillel (2025): LightGBM test CEL 0.6537, the best number on LPMC.

**Model.** Plain multiclass LightGBM on [all alternatives' attributes
flattened, person / trip covariates]. Not a RUM: any feature can enter any
class score, so it has no behavioural interpretation; it is the accuracy /
NLL ceiling that the structured models are measured against.

**Implementation.** `model.py::run`: 15 leaves, lr 0.05, feature and bagging
fraction 0.8, L2 1.0, up to 2,000 rounds with early stopping (50) on
validation NLL. Runs in a torch-free child process (see `methods/README.md`).
