#!/usr/bin/env python3
"""Generate the two figures used in the report, from the measured diagnostics
and leaderboard results gathered during the assignment. All numbers are the
actual values produced by diagnose_watermarks_validated.py, the BM3D
copy-consistency analysis, identify_scheme.py, and the public leaderboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = [f"WM\\_{i}" if False else f"WM_{i}" for i in range(1, 9)]
DEEP = {"WM_2", "WM_7", "WM_8"}  # confirmed deep-scheme groups

# --- Out-of-fold detector AUC per feature family (validated_diagnostics.json) --
FEATURES = ["residual", "Y", "Cb", "Cr", "LSB", "DCT"]
AUC = np.array([
    [0.660, 0.619, 0.999, 0.600, 0.895, 0.513],  # WM_1
    [0.376, 0.413, 0.485, 0.509, 0.566, 0.349],  # WM_2
    [0.976, 0.991, 0.979, 0.970, 0.746, 0.654],  # WM_3
    [0.764, 0.799, 0.516, 0.404, 0.732, 0.716],  # WM_4
    [0.562, 0.496, 0.963, 0.912, 0.995, 0.448],  # WM_5
    [0.519, 0.563, 0.639, 0.549, 0.899, 0.960],  # WM_6
    [0.513, 0.458, 0.467, 0.519, 0.617, 0.421],  # WM_7
    [0.566, 0.512, 0.466, 0.408, 0.449, 0.404],  # WM_8
])

# --- BM3D copy-attack residual consistency (||mean||^2 / mean||.||^2) ----------
CONS_RAW = [0.092, 0.078, 0.164, 0.063, 0.130, 0.091, 0.111, 0.130]
CONS_BM3D = [0.087, 0.084, 0.105, 0.153, 0.388, 0.309, 0.084, 0.090]

# --- Public leaderboard score progression -------------------------------------
STAGES = ["Specialized\npipeline", "Raw avg\n(s=0.3)", "Raw avg\n(s=0.5)",
          "Per-category\ntuning", "+WM_2\nRivaGAN", "+WM_7\nTrustMark-Q",
          "+WM_8\nTrustMark-P"]
SCORES = [0.220, 0.265, 0.295, 0.334, 0.477, 0.602, 0.719]

# --- Scheme identification: source-decode agreement vs clean control ----------
ID_GROUPS = ["WM_2\n(RivaGAN)", "WM_7\n(TrustMark-Q)", "WM_8\n(TrustMark-P)"]
ID_SRC = [0.989, 1.000, 0.999]
ID_CTRL = [0.626, 0.612, 0.608]


def fig_diagnostics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 2.85),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # Panel (a): AUC heatmap, diverging around 0.5 (chance)
    im = ax1.imshow(AUC, cmap="RdBu_r", vmin=0.30, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(len(FEATURES)))
    ax1.set_xticklabels(FEATURES, rotation=30, ha="right")
    ax1.set_yticks(range(len(GROUPS)))
    ylabels = [f"{g} ★" if g in DEEP else g for g in GROUPS]
    ax1.set_yticklabels(ylabels)
    for yi in range(AUC.shape[0]):
        for xi in range(AUC.shape[1]):
            v = AUC[yi, xi]
            ax1.text(xi, yi, f"{v:.2f}", ha="center", va="center",
                     color="white" if (v > 0.82 or v < 0.42) else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("out-of-fold detector AUC", fontsize=8)
    ax1.set_title("(a) Classical-feature separability\n★ = confirmed deep scheme",
                  fontsize=9)

    # Panel (b): BM3D copy-consistency, raw vs bm3d
    x = np.arange(len(GROUPS))
    w = 0.38
    ax2.bar(x - w/2, CONS_RAW, w, label="raw mean-diff", color="#9ecae1")
    ax2.bar(x + w/2, CONS_BM3D, w, label="BM3D regeneration", color="#08519c")
    ax2.axhspan(0.20, 0.42, color="green", alpha=0.08)
    ax2.text(6.9, 0.225, "copyable\nregime", fontsize=7.5, color="green", ha="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.replace("WM_", "") for g in GROUPS])
    ax2.set_xlabel("watermark group (WM$_k$)")
    ax2.set_ylabel("watermark residual consistency")
    ax2.set_title("(b) Copy-attack isolation quality", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_ylim(0, 0.44)

    fig.tight_layout()
    fig.savefig("fig_diagnostics.pdf", bbox_inches="tight")
    fig.savefig("fig_diagnostics.png", dpi=150, bbox_inches="tight")
    print("wrote fig_diagnostics.pdf/.png")


def fig_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 2.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # Panel (a): score progression
    x = np.arange(len(STAGES))
    colors = ["#bdbdbd", "#bdbdbd", "#bdbdbd", "#bdbdbd", "#c6dbef", "#6baed6", "#08519c"]
    ax1.bar(x, SCORES, color=colors)
    for xi, s in zip(x, SCORES):
        ax1.text(xi, s + 0.012, f"{s:.3f}", ha="center", fontsize=7.5)
    ax1.axvspan(-0.5, 3.5, color="grey", alpha=0.06)
    ax1.axvspan(3.5, 6.5, color="#08519c", alpha=0.06)
    ax1.text(1.5, 0.68, "statistical\nestimation", ha="center", fontsize=8, color="#555555")
    ax1.text(5.0, 0.68, "scheme\nidentification", ha="center", fontsize=8, color="#08519c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGES, fontsize=6.8)
    ax1.set_ylabel("public leaderboard score")
    ax1.set_ylim(0, 0.80)
    ax1.set_title("(a) Score progression", fontsize=9)

    # Panel (b): scheme-ID agreement vs control
    xi = np.arange(len(ID_GROUPS))
    w = 0.38
    ax2.bar(xi - w/2, ID_SRC, w, label="watermarked sources", color="#08519c")
    ax2.bar(xi + w/2, ID_CTRL, w, label="clean control", color="#c6dbef")
    ax2.axhline(0.5, ls="--", color="grey", lw=0.8)
    for k in range(len(ID_GROUPS)):
        ax2.text(xi[k] - w/2, ID_SRC[k] + 0.01, f"{ID_SRC[k]:.2f}", ha="center", fontsize=7)
    ax2.set_xticks(xi)
    ax2.set_xticklabels(ID_GROUPS, fontsize=7.5)
    ax2.set_ylabel("source-decode bit agreement")
    ax2.set_ylim(0, 1.08)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.set_title("(b) Scheme identification", fontsize=9)

    fig.tight_layout()
    fig.savefig("fig_results.pdf", bbox_inches="tight")
    fig.savefig("fig_results.png", dpi=150, bbox_inches="tight")
    print("wrote fig_results.pdf/.png")


if __name__ == "__main__":
    fig_diagnostics()
    fig_results()
