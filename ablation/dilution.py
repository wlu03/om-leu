"""Is the small mixture effect just dilution?

For p = (1-pi) s + pi q, the first-order change in the event loss when the semantic channel
moves from q to q' is

    d(-log p_y) ~= -pi (q'_y - q_y) / p_y ,

so the standalone change is multiplied by the "responsibility" pi q_y / p_y, not by pi alone.
This script measures the predicted and the observed change, and reports the responsibility.
"""
import sys, numpy as np, torch
sys.path.insert(0, "ablation"); sys.path.insert(0, ".")
import _common
from experiments.harness.data import load_bundle
from experiments.models.omleu2 import _log_mean_exp

DS, SEED = sys.argv[1], int(sys.argv[2])
mods = _common.discover(); b = load_bundle(DS, SEED)
got = {}
for name in ("full_model", "no_person_weights"):
    model, run = _common.build(mods[name], "person", "oof")(b, SEED)
    run(model, b, SEED)
    bb = getattr(model, "data_view", None) or b
    te = bb.idx("test")
    with torch.no_grad():
        q = _log_mean_exp(torch.stack([m(model.sem_view or bb, te) for m in model.members], 1), 1).exp().double().numpy()
        s = torch.softmax(float(model.log_a.exp()) * model.U[te], -1).double().numpy()
        p = model(bb, te).exp().double().numpy()
    got[name] = dict(q=q, s=s, p=p, pi=float(model.pi), y=bb.y[te].numpy())
y = got["full_model"]["y"]; n = np.arange(len(y))
F, A = got["full_model"], got["no_person_weights"]
pi = F["pi"]
qF, qA, sF, pF = F["q"][n, y], A["q"][n, y], F["s"][n, y], F["p"][n, y]
resp = pi * qF / pF                                   # share of the mixture carried by the semantic channel
d_standalone = float((-np.log(qA)).mean() - (-np.log(qF)).mean())
d_predicted = float((-pi * (qA - qF) / pF).mean())
pA_same_pi = (1 - pi) * sF + pi * A["q"][n, y]
d_observed_same_pi = float((-np.log(pA_same_pi)).mean() - (-np.log(pF)).mean())
d_observed_refit = float((-np.log(A["p"][n, y])).mean() - (-np.log(pF)).mean())
print(f"{DS} seed {SEED}")
print(f"  pi (full / ablated)                  {pi:.3f} / {A['pi']:.3f}")
print(f"  mean responsibility  pi*q_y/p_y      {resp.mean():.3f}   (median {np.median(resp):.3f}, p90 {np.quantile(resp,0.9):.3f})")
print(f"  standalone change                    {d_standalone:+.4f}")
print(f"  first-order prediction for the mixture {d_predicted:+.4f}")
print(f"  observed, pi held fixed               {d_observed_same_pi:+.4f}")
print(f"  observed, pi refitted                 {d_observed_refit:+.4f}")
print(f"  attenuation factor (standalone / observed-refit) {d_standalone / max(d_observed_refit,1e-9):.1f}x")
# where does the standalone loss land?
loss = (-np.log(qA)) - (-np.log(qF))
o = np.argsort(-loss); top = o[:max(1, len(o)//10)]
print(f"  top decile of standalone loss: mean loss {loss[top].mean():+.3f}, "
      f"their mean responsibility {resp[top].mean():.3f}, "
      f"structural already correct on {float((sF[top] == F['s'][top].max(1)).mean())*100:.0f}% of them")
print(f"  rest: mean loss {loss[o[len(o)//10:]].mean():+.3f}, mean responsibility {resp[o[len(o)//10:]].mean():.3f}")
