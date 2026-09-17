"""Figure 1, drawn for a 5.5-inch text column rather than a slide.

    ../venv/bin/python figures/make_fig1.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W, H = 5.5, 2.45
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
GREY, TEAL, AMBER, INDIGO, INK = "#6b7280", "#1f7a6d", "#b26a1b", "#4553b5", "#16181d"
FS, FL = 6.0, 6.6


def box(x, y, w, h, color, fill=0.96, lw=0.7, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=(1, 1, 1) if fill is None else plt.matplotlib.colors.to_rgba(color, 0.07),
                                ec=color, lw=lw, ls=ls, zorder=2))


def arrow(x1, y1, x2, y2, color=INK, lw=0.8, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
                                 color=color, lw=lw, linestyle=ls, zorder=3,
                                 connectionstyle=f"arc3,rad={rad}"))


def txt(x, y, s, size=FS, color=INK, ha="center", va="center", weight="normal", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, style=style, zorder=4)


# ---- inputs -------------------------------------------------------------------
box(1, 12, 15, 20, GREY)
txt(8.5, 29.5, "inputs", size=FL, weight="bold", color=GREY)
txt(8.5, 24.5, "attributes  $x_{ij}$", size=FS)
txt(8.5, 20.5, "covariates  $z_i$", size=FS)
txt(8.5, 16.5, "history  $h_{ij}$", size=FS)

# ---- conventional channel -----------------------------------------------------
box(20, 24, 26, 17, TEAL)
txt(33, 38.0, "1  structural utility", size=FL, weight="bold", color=TEAL)
txt(33, 33.0, r"$V_{ij}=\mathrm{ASC}_j+\beta^{\!\top}x_{ij}$", size=FS)
txt(33, 29.6, r"$-\beta_t(z_i)t_{ij}-\beta_c(z_i)c_{ij}+g_j(z_i)+u_{p(i)j}$", size=5.4)
txt(33, 26.3, "R = 5 fits averaged in probability", size=5.4, style="italic", color=TEAL)

box(50, 24, 22, 17, AMBER)
txt(61, 38.0, "2  boosted residual", size=FL, weight="bold", color=AMBER)
txt(61, 33.0, r"$U_{ij}=\tau L_{ij}+f_j(x_{ij},z_i,h_{ij})$", size=FS)
txt(61, 29.0, "one ensemble per alternative,", size=5.4, style="italic", color=AMBER)
txt(61, 26.5, "monotone in time and cost", size=5.4, style="italic", color=AMBER)

# ---- language channel ---------------------------------------------------------
box(20, 3, 52, 17, INDIGO)
txt(46, 17.2, "3  language channel", size=FL, weight="bold", color=INDIGO)
for i, (x, lab) in enumerate([(26, "frozen\nLLM"), (38, "$K$ sentences\nper alternative"),
                              (51, "frozen encoder\n$768$-d"), (64, "reader\n$\\rightarrow \\bar p_i(j)$")]):
    box(x - 5.5, 6.0, 11, 8.2, INDIGO, lw=0.6)
    txt(x, 10.1, lab, size=5.4, color=INK)
    if i < 3:
        arrow(x + 5.8, 10.1, x + 6.4, 10.1, INDIGO, lw=0.7)
txt(46, 4.2, "no gradient reaches the language model or the encoder", size=5.2, style="italic", color=INDIGO)

# ---- mixture ------------------------------------------------------------------
box(78, 12, 21, 21, INK)
txt(88.5, 30.0, "probability mixture", size=FL, weight="bold")
txt(88.5, 24.8, r"$p_i(j)=(1-\pi)\,q_i(j)$", size=FS)
txt(88.5, 21.2, r"$+\ \pi\,\bar p_i(j)$", size=FS)
txt(88.5, 16.0, "$\\pi$ and $a$ stacked\non out-of-fold predictions", size=5.2, style="italic")

# ---- wiring -------------------------------------------------------------------
arrow(16.3, 26.5, 19.7, 30.0, TEAL)
arrow(16.3, 20.0, 19.7, 11.5, INDIGO)
arrow(46.3, 32.5, 49.7, 32.5, TEAL)
txt(48, 34.4, r"$\tau L_{ij}$", size=5.2, color=TEAL)
arrow(72.3, 32.5, 77.7, 27.0, AMBER)
txt(74.2, 35.2, r"$1-\pi$", size=5.6, color=AMBER)
arrow(72.3, 11.5, 77.7, 18.0, INDIGO)
txt(74.6, 14.2, r"$\pi$", size=5.6, color=INDIGO)
txt(61, 21.9, r"$q_i(j)=\mathrm{softmax}_j(a\,U_i)$", size=5.2, color=AMBER)

fig.savefig("figures/architecture.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig("figures/architecture.png", dpi=220, bbox_inches="tight", pad_inches=0.01)
print("figure written")
