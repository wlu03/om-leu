"""Why does removing the personalised weights change the full system so little?

Five diagnostics on one dataset/seed under the person-level split:
  1. how much the learned person weights actually vary across people
  2. how much the semantic channel's predictions change when they are removed
  3. whether the fitted mixture weight compensates (evaluate the ablated channel at the full
     model's pi as well as at its own)
  4. how correlated the two channels' errors are, which bounds what a mixture can gain
  5. where the standalone loss sits, by how confident the structural model is
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "ablation"); sys.path.insert(0, ".")
import _common
from experiments.harness.data import load_bundle
from experiments.models.omleu2 import _log_mean_exp

DS, SEED = sys.argv[1], int(sys.argv[2])
mods = _common.discover()
b = load_bundle(DS, SEED)
out = {"dataset": DS, "seed": SEED}
fit = {}
for name in ("full_model", "no_person_weights", "global_weights"):
    if name not in mods: continue
    model, run = _common.build(mods[name], "person", "oof")(b, SEED)
    info = run(model, b, SEED)
    bb = getattr(model, "data_view", None) or b
    te = bb.idx("test")
    with torch.no_grad():
        sem = _log_mean_exp(torch.stack([m(model.sem_view or bb, te) for m in model.members], 1), 1)
        a = float(model.a)
        struct = torch.log_softmax(a * model.U[te], -1)
        mix = model(bb, te)
    y = bb.y[te]
    fit[name] = {"sem": sem.double().numpy(), "struct": struct.double().numpy(),
                 "mix": mix.double().numpy(), "y": y.numpy(), "pi": float(model.pi), "a": a,
                 "members": model.members, "view": model.sem_view or bb, "te": te}
    print(f"  fitted {name}: pi={float(model.pi):.3f} a={a:.3f}")

def nll(lp, y): return float(-lp[np.arange(len(y)), y].mean())
y = fit["full_model"]["y"]
full, abl = fit["full_model"], fit["no_person_weights"]
glob = fit.get("global_weights")

# 1. spread of the learned person weights
m0 = full["members"][0]
with torch.no_grad():
    z = m0.person(full["view"], full["te"])
    w = torch.softmax(m0.weights(z), -1).numpy()
out["person_weights"] = {"mean": w.mean(0).round(4).tolist(), "sd_across_people": w.std(0).round(4).tolist(),
                         "max_person_range": float((w.max(0) - w.min(0)).max()),
                         "share_of_people_whose_argmax_is_the_modal_head":
                             float((w.argmax(1) == np.bincount(w.argmax(1)).argmax()).mean())}
# 2. how much do the channel's predictions move
out["prediction_shift"] = {
    "mean_abs_prob_change": float(np.abs(np.exp(full["sem"]) - np.exp(abl["sem"])).sum(1).mean() / 2),
    "kendall_agreement_top1": float((full["sem"].argmax(1) == abl["sem"].argmax(1)).mean()),
    "sem_nll_full": nll(full["sem"], y), "sem_nll_ablated": nll(abl["sem"], y)}
# 3. does pi compensate?  evaluate each channel at its own pi and at the other's
def mix_at(struct_lp, sem_lp, pi):
    return np.logaddexp(np.log1p(-pi) + struct_lp, np.log(pi) + sem_lp)
out["pi_compensation"] = {
    "pi_full": full["pi"], "pi_ablated": abl["pi"],
    "mix_full_at_own_pi": nll(full["mix"], y),
    "mix_ablated_at_own_pi": nll(abl["mix"], y),
    "mix_ablated_at_full_pi": nll(mix_at(abl["struct"], abl["sem"], full["pi"]), y),
    "mix_full_at_ablated_pi": nll(mix_at(full["struct"], full["sem"], abl["pi"]), y),
    "structural_only": nll(full["struct"], y)}
# 4. error correlation and how often each channel is better
e_s = -full["struct"][np.arange(len(y)), y]; e_m = -full["sem"][np.arange(len(y)), y]
e_a = -abl["sem"][np.arange(len(y)), y]
out["channel_relationship"] = {
    "error_correlation_struct_vs_sem": float(np.corrcoef(e_s, e_m)[0, 1]),
    "share_events_sem_better": float((e_m < e_s).mean()),
    "share_events_ablated_sem_better": float((e_a < e_s).mean()),
    "error_correlation_sem_full_vs_ablated": float(np.corrcoef(e_m, e_a)[0, 1])}
# 5. where the standalone loss sits, by structural confidence
conf = np.exp(full["struct"]).max(1)
q = np.quantile(conf, [0.25, 0.5, 0.75])
bins = np.digitize(conf, q)
out["by_structural_confidence"] = []
for k in range(4):
    m_ = bins == k
    if m_.sum() < 10: continue
    out["by_structural_confidence"].append({
        "quartile": k, "n": int(m_.sum()), "mean_struct_confidence": float(conf[m_].mean()),
        "sem_loss_from_ablation": float(e_a[m_].mean() - e_m[m_].mean()),
        "mixture_loss_from_ablation": float((-abl["mix"][np.arange(len(y)), y])[m_].mean()
                                            - (-full["mix"][np.arange(len(y)), y])[m_].mean())})
if glob is not None:
    out["learned_global_weights"] = {
        "sem_nll": nll(glob["sem"], y), "mix_nll": nll(glob["mix"], y), "pi": glob["pi"],
        "weights": torch.softmax(glob["members"][0].global_w.detach(), -1).numpy().round(4).tolist()}
print(json.dumps(out, indent=1, default=float))
