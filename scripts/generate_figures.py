"""Generate all paper figures (matplotlib -> PDF).

Usage:
    uv run python scripts/generate_figures.py              # all figures
    uv run python scripts/generate_figures.py --fig 1 5 6  # specific figures
    uv run python scripts/generate_figures.py --fig 5 --png  # PNG output (raster)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
COLORS = {
    "var_full": "#4C72B0",      # blue — variational full-weight
    "var_lora": "#DD8452",      # orange — BLoB LoRA
    "post_full": "#C44E52",     # red — diag. Laplace full
    "post_lora": "#8172B3",     # purple — TFB LoRA
    "post_lora_lap": "#937860", # brown — diag. Laplace LoRA
    "mc_dropout": "#55A868",    # green — MC Dropout
    "deterministic": "#AAAAAA", # grey — deterministic baseline
    "accent": "#E07B39",        # highlight
    "bg_light": "#F5F5F5",
    "success": "#55A868",
    "failure": "#C44E52",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def _gauss(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _save(fig, name, fmt):
    path = OUT_DIR / f"{name}.{fmt}"
    fig.savefig(path, format=fmt, dpi=300 if fmt == "png" else None)
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Fig 1: Point Weights vs Bayesian Weights
# ===================================================================
def fig1_point_vs_bayesian(fmt="pdf"):
    """Side-by-side: deterministic layer (scalar weights) vs Bayesian layer
    (weight distributions). Shows the core concept of the paper."""
    fig, (ax_det, ax_bay) = plt.subplots(1, 2, figsize=(10, 4.5))

    def _draw_layer(ax, title, bayesian=False):
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Input nodes
        in_y = [1.0, 2.5, 4.0]
        out_y = [0.5, 2.5, 4.5]
        in_x, out_x = 0.5, 3.5

        for y in in_y:
            circle = plt.Circle((in_x, y), 0.25, fc="#D4E6F1", ec="black", lw=1.2)
            ax.add_patch(circle)
        for y in out_y:
            circle = plt.Circle((out_x, y), 0.25, fc="#D5F5E3", ec="black", lw=1.2)
            ax.add_patch(circle)

        ax.text(in_x, -0.3, "Input", ha="center", fontsize=9, style="italic")
        ax.text(out_x, -0.3, "Output", ha="center", fontsize=9, style="italic")

        # Connections
        weights = [0.7, -0.3, 0.5, 0.2, -0.8, 0.4, 0.1, -0.6, 0.9]
        idx = 0
        for iy in in_y:
            for oy in out_y:
                if bayesian:
                    # Draw a small Gaussian curve along the connection midpoint
                    mx, my = (in_x + out_x) / 2, (iy + oy) / 2
                    ax.annotate(
                        "", xy=(out_x - 0.3, oy), xytext=(in_x + 0.3, iy),
                        arrowprops=dict(arrowstyle="-", color="#888888",
                                        lw=0.6, linestyle="--"),
                    )
                    # Mini Gaussian
                    t = np.linspace(-1.2, 1.2, 50)
                    g = _gauss(t, 0, 0.4) * 0.25
                    angle = np.arctan2(oy - iy, out_x - in_x)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    # Perpendicular to connection line
                    gx = mx + t * cos_a * 0.15 - g * sin_a
                    gy = my + t * sin_a * 0.15 + g * cos_a
                    ax.fill(gx, gy, alpha=0.5, color=COLORS["accent"], lw=0)
                    ax.plot(gx, gy, color=COLORS["accent"], lw=0.8)
                else:
                    w = weights[idx]
                    lw = 0.5 + abs(w) * 2.0
                    color = "#2E86C1" if w > 0 else "#C0392B"
                    ax.annotate(
                        "", xy=(out_x - 0.3, oy), xytext=(in_x + 0.3, iy),
                        arrowprops=dict(arrowstyle="-", color=color, lw=lw),
                    )
                    mx, my = (in_x + out_x) / 2, (iy + oy) / 2
                    ax.text(mx + 0.1, my + 0.15, f"{w:.1f}", fontsize=6,
                            ha="center", color=color)
                idx += 1

    _draw_layer(ax_det, "Deterministic: $w_{ij}$ = scalar", bayesian=False)
    _draw_layer(ax_bay, r"Bayesian: $w_{ij} \sim \mathcal{N}(\mu, \sigma^2)$",
                bayesian=True)

    # Bottom annotation
    fig.text(0.5, -0.02,
             "N weight samples -> N predictions -> disagreement = epistemic uncertainty (MI)",
             ha="center", fontsize=10, style="italic", color="#555555")

    fig.tight_layout()
    _save(fig, "fig1_point_vs_bayesian", fmt)


# ===================================================================
# Fig 2: The 2×2 Method Matrix
# ===================================================================
def fig2_method_matrix(fmt="pdf"):
    """2×2 grid showing four Bayesian methods with results."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    methods = [
        ("Variational FFN\n(Full Weights)", "var_full",
         "All FFN weights:\n$w_{ij} \\sim \\mathcal{N}(\\mu, \\sigma^2)$\n"
         "Train: ELBO from scratch\n33.6M Bayesian params",
         "AUROC 0.874", True),
        ("BLoB LoRA\n(LoRA Adapters)", "var_lora",
         "LoRA A matrix only:\n$A_{ij} \\sim \\mathcal{N}(\\mu, \\sigma^2)$\n"
         "Train: LoRA fine-tune + KL\n1.97M Bayesian params",
         "AUROC 0.909", True),
        ("Diag. Laplace FFN\n(Full Weights)", "post_full",
         "Post-hoc on all FFN weights:\n$\\mathcal{N}(\\hat{\\theta},"
         " \\mathrm{diag}(F)^{-1})$\nFit: diagonal Fisher\n33.6M params",
         "AUROC 0.536", False),
        ("TFB LoRA\n(LoRA Adapters)", "post_lora",
         "Post-hoc on LoRA via SVD:\n$\\Omega_{ij} = \\sigma_q / d_i$\n"
         "Fit: binary search (7 min)\n1.97M params",
         "AUROC 0.917", True),
    ]

    row_labels = ["Variational\n(train-time)", "Post-hoc\n(no Bayesian training)"]
    col_labels = ["Full Weights", "LoRA Adapters"]

    for i, (ax, (name, color_key, desc, result, success)) in enumerate(
        zip(axes.flat, methods)
    ):
        row, col = divmod(i, 2)
        bg = "#E8F8E8" if success else "#FDEDED"
        ax.set_facecolor(bg)

        ax.text(0.5, 0.88, name, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=COLORS[color_key])
        ax.text(0.5, 0.55, desc, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#333333", linespacing=1.4)

        result_color = COLORS["success"] if success else COLORS["failure"]
        ax.text(0.5, 0.08, result, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=result_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=result_color, alpha=0.9))
        ax.set_xticks([])
        ax.set_yticks([])

    # Row / column labels
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.72 - i * 0.44, label, ha="center", va="center",
                 fontsize=10, fontweight="bold", rotation=90, color="#444444")
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=11, fontweight="bold",
                             color="#444444", pad=10)

    # Also add MC Dropout annotation
    fig.text(0.5, -0.03,
             "Plus MC Dropout baseline (zero training): AUROC 0.898 — "
             "overlapping CI with BLoB LoRA",
             ha="center", fontsize=9, style="italic", color=COLORS["mc_dropout"])

    fig.tight_layout(rect=[0.05, 0.02, 1, 0.96])
    _save(fig, "fig2_method_matrix", fmt)


# ===================================================================
# Fig 3: BLoB LoRA Internals
# ===================================================================
def fig3_blob_lora(fmt="pdf"):
    """ΔW = (α/r) B A — A has distributions, B is deterministic."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1.5, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # ΔW = (α/r) B × A
    def _draw_matrix(ax, x, y, rows, cols, label, bayesian=False, values=None):
        """Draw a matrix as a grid of cells."""
        cell_w, cell_h = 0.7, 0.7
        for r in range(rows):
            for c in range(cols):
                cx = x + c * cell_w
                cy = y - r * cell_h
                if bayesian:
                    # Small Gaussian curve in cell
                    rect = FancyBboxPatch(
                        (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
                        boxstyle="round,pad=0.02", facecolor="#FFF3E0",
                        edgecolor=COLORS["accent"], lw=0.8,
                    )
                    ax.add_patch(rect)
                    t = np.linspace(-0.25, 0.25, 30)
                    g = _gauss(t, 0, 0.08) * 0.015
                    ax.fill(cx + t, cy - 0.15 + g, alpha=0.6,
                            color=COLORS["accent"])
                    ax.plot(cx + t, cy - 0.15 + g, color=COLORS["accent"], lw=0.5)
                else:
                    val = values[r * cols + c] if values else 0
                    fc = "#D4E6F1" if val >= 0 else "#FADBD8"
                    rect = FancyBboxPatch(
                        (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
                        boxstyle="round,pad=0.02", facecolor=fc,
                        edgecolor="#888888", lw=0.8,
                    )
                    ax.add_patch(rect)
                    ax.text(cx, cy, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color="#333333")
        # Label
        mid_x = x + (cols - 1) * cell_w / 2
        ax.text(mid_x, y + cell_h * 0.8, label, ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        # Dimensions
        ax.text(mid_x, y - rows * cell_h + 0.05, f"{rows}×{cols}",
                ha="center", va="top", fontsize=8, color="#888888")
        return mid_x

    # Matrix B (deterministic): d × r  (show 4×2)
    b_vals = [0.3, -0.1, 0.5, 0.2, -0.4, 0.7, 0.1, -0.3]
    _draw_matrix(ax, 1, 4, 4, 2, "$B$ (deterministic)", False, b_vals)

    # Multiplication sign
    ax.text(3.5, 2.5, "×", fontsize=20, ha="center", va="center")

    # Matrix A (Bayesian): r × d  (show 2×4)
    _draw_matrix(ax, 5, 4, 2, 4, "$A$ (Bayesian: $\\mu, \\sigma$)", True)

    # Equals sign
    ax.text(9.0, 2.5, "=", fontsize=20, ha="center", va="center")

    # ΔW result: d × d  (show 4×4, lighter)
    dw_vals = [0.1, -0.2, 0.3, 0.0, -0.1, 0.4, -0.2, 0.1,
               0.2, -0.1, 0.0, 0.3, -0.3, 0.2, 0.1, -0.1]
    _draw_matrix(ax, 10, 4, 4, 4, "$\\Delta W$", False, dw_vals)

    # Annotations
    ax.annotate("$\\frac{\\alpha}{r}$", xy=(3.5, 3.5), fontsize=14,
                ha="center", va="center", color="#555555")

    # Key insight box
    box_text = ("Only $A$ carries uncertainty ($\\mu_{ij}, \\sigma_{ij}$)\n"
                "$B$ is deterministic -> KL factorizes in rank-$r$ A-space\n"
                "1.97M Bayesian params (2.5% of 76M model)")
    ax.text(6.5, -1.0, box_text, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1",
                      edgecolor=COLORS["accent"], lw=1.2),
            linespacing=1.6)

    fig.tight_layout()
    _save(fig, "fig3_blob_lora", fmt)


# ===================================================================
# Fig 4: Why TFB Works and Laplace Doesn't
# ===================================================================
def fig4_tfb_vs_laplace(fmt="pdf"):
    """Split panel: diagonal Fisher flatness vs SVD structure."""
    fig, (ax_lap, ax_tfb) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left panel: Diagonal Laplace (fails) ---
    ax_lap.set_title("Diagonal Laplace (FAILS)", fontsize=11,
                     fontweight="bold", color=COLORS["failure"])

    # Show flat Fisher curvature
    x = np.linspace(0, 100, 200)
    fisher_vals = np.random.exponential(0.001, 200)
    fisher_vals = np.sort(fisher_vals)[::-1]
    ax_lap.fill_between(x, 0, fisher_vals, alpha=0.3, color=COLORS["failure"])
    ax_lap.plot(x, fisher_vals, color=COLORS["failure"], lw=1.5)
    ax_lap.set_xlabel("Parameter index (sorted)")
    ax_lap.set_ylabel("Diagonal Fisher value")
    ax_lap.set_ylim(0, 0.015)
    ax_lap.axhline(y=0.001, color="#888888", ls="--", lw=0.8)
    ax_lap.text(50, 0.0013, "≈ 0 (flat curvature)", ha="center",
                fontsize=8, color="#888888")

    # Annotation
    ax_lap.text(0.5, 0.85, "Posterior: $\\mathcal{N}(\\hat{\\theta},"
                " \\mathrm{diag}(F)^{-1})$",
                transform=ax_lap.transAxes, ha="center", fontsize=9,
                color="#555555")
    ax_lap.text(0.5, 0.72, "Flat $F$ -> huge variance -> random samples\n"
                "-> MI ≈ 0 (no signal)",
                transform=ax_lap.transAxes, ha="center", fontsize=9,
                color=COLORS["failure"], linespacing=1.5)

    result_box = dict(boxstyle="round,pad=0.3", facecolor="#FDEDED",
                      edgecolor=COLORS["failure"])
    ax_lap.text(0.5, 0.05, "AUROC ≈ 0.5 (random)",
                transform=ax_lap.transAxes, ha="center", fontsize=10,
                fontweight="bold", color=COLORS["failure"], bbox=result_box)

    # --- Right panel: TFB (succeeds) ---
    ax_tfb.set_title("TFB: SVD-Structured Variance (SUCCEEDS)", fontsize=11,
                     fontweight="bold", color=COLORS["success"])

    # Show SVD singular values — meaningful structure
    rank = 16
    singular_vals = np.array([2.5, 2.1, 1.8, 1.5, 1.2, 1.0, 0.85, 0.7,
                              0.55, 0.45, 0.35, 0.28, 0.2, 0.15, 0.1, 0.07])
    variance = 0.03 / singular_vals  # σ_q / d_i

    ax_tfb_sv = ax_tfb
    bar_colors = plt.cm.Purples(np.linspace(0.3, 0.9, rank))
    ax_tfb_sv.bar(range(rank), singular_vals, color=bar_colors,
                   edgecolor="#555555", lw=0.5, alpha=0.8)
    ax_tfb_sv.set_xlabel("SVD direction $i$")
    ax_tfb_sv.set_ylabel("Singular value $d_i$", color=COLORS["post_lora"])

    # Overlay variance on secondary axis
    ax_var = ax_tfb_sv.twinx()
    ax_var.plot(range(rank), variance, "o-", color=COLORS["accent"],
                lw=2, markersize=5)
    ax_var.set_ylabel("Variance $\\Omega_i = \\sigma_q / d_i$",
                      color=COLORS["accent"])
    ax_var.tick_params(axis="y", labelcolor=COLORS["accent"])

    ax_tfb_sv.text(0.5, 0.88, "$\\Omega_{ij} = \\sigma_q / d_i$ "
                   "(SVD of $B$)",
                   transform=ax_tfb_sv.transAxes, ha="center", fontsize=9,
                   color="#555555")
    ax_tfb_sv.text(0.5, 0.75, "Strong directions -> low variance (preserve)\n"
                   "Weak directions -> high variance (explore)",
                   transform=ax_tfb_sv.transAxes, ha="center", fontsize=9,
                   color=COLORS["success"], linespacing=1.5)

    result_box_g = dict(boxstyle="round,pad=0.3", facecolor="#E8F8E8",
                        edgecolor=COLORS["success"])
    ax_tfb_sv.text(0.5, 0.05, "AUROC 0.917",
                   transform=ax_tfb_sv.transAxes, ha="center", fontsize=10,
                   fontweight="bold", color=COLORS["success"], bbox=result_box_g)

    fig.tight_layout()
    _save(fig, "fig4_tfb_vs_laplace", fmt)


# ===================================================================
# Fig 5: AUROC Bar Chart with 95% CIs
# ===================================================================
def fig5_auroc_bars(fmt="pdf"):
    """Horizontal bar chart: AUROC per method with bootstrap 95% CIs."""
    methods = [
        ("TFB LoRA",          0.917, 0.900, 0.933, COLORS["post_lora"]),
        ("BLoB LoRA",         0.909, 0.890, 0.925, COLORS["var_lora"]),
        ("MC Dropout",        0.898, 0.877, 0.917, COLORS["mc_dropout"]),
        ("Variational FFN",   0.874, 0.852, 0.895, COLORS["var_full"]),
        ("Deterministic",     0.591, 0.556, 0.626, COLORS["deterministic"]),
        ("Diag. Laplace FFN", 0.536, 0.500, 0.572, COLORS["post_full"]),
        ("Diag. Laplace LoRA",0.494, 0.459, 0.529, COLORS["post_lora_lap"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(methods))

    for i, (name, auroc, lo, hi, color) in enumerate(methods):
        xerr_lo = auroc - lo
        xerr_hi = hi - auroc
        ax.barh(i, auroc, color=color, alpha=0.85, edgecolor="white", lw=0.5,
                height=0.65)
        ax.errorbar(auroc, i, xerr=[[xerr_lo], [xerr_hi]], fmt="none",
                    ecolor="black", capsize=4, lw=1.2)
        # Value label
        ax.text(auroc + 0.008, i, f"{auroc:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in methods])
    ax.set_xlabel("AUROC (higher is better)")
    ax.set_xlim(0.4, 1.0)
    ax.axvline(x=0.5, color="#CCCCCC", ls="--", lw=0.8, label="Random baseline")
    ax.invert_yaxis()

    # Bracket annotation: "CIs overlap"
    ax.annotate("", xy=(0.935, 0), xytext=(0.935, 2),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.0))
    ax.text(0.948, 1, "CIs\noverlap", fontsize=7, ha="center", va="center",
            color="#888888")

    fig.tight_layout()
    _save(fig, "fig5_auroc_bars", fmt)


# ===================================================================
# Fig 6: N vs AUROC — "N=3 is the knee"
# ===================================================================
def fig6_n_vs_auroc(fmt="pdf"):
    """Line plot: MC samples N vs AUROC for each method."""
    # Data from Table 4 in paper
    ns = [1, 3, 5, 10, 20]

    data = {
        "Full Var. MC":   [0.500, 0.850, 0.855, 0.866, 0.869],
        "BLoB LoRA MC":   [0.500, 0.861, 0.879, 0.880, 0.888],
        "TFB LoRA MC":    [0.500, 0.847, 0.859, 0.881, 0.886],
    }

    colors_map = {
        "Full Var. MC": COLORS["var_full"],
        "BLoB LoRA MC": COLORS["var_lora"],
        "TFB LoRA MC": COLORS["post_lora"],
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, aurocs in data.items():
        ax.plot(ns, aurocs, "o-", color=colors_map[name], lw=2,
                markersize=7, label=name)

    # Highlight N=3 knee
    ax.axvline(x=3, color="#CCCCCC", ls="--", lw=1.0)
    ax.annotate("N=3: the knee\n(97% of full signal)",
                xy=(3, 0.855), xytext=(7, 0.72),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                          edgecolor=COLORS["accent"]))

    # Shaded "diminishing returns" region
    ax.axvspan(5, 20, alpha=0.05, color="#888888")
    ax.text(12.5, 0.52, "Diminishing returns", fontsize=8,
            ha="center", color="#888888", style="italic")

    ax.set_xlabel("Number of MC samples (N)")
    ax.set_ylabel("AUROC")
    ax.set_xticks(ns)
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig6_n_vs_auroc", fmt)


# ===================================================================
# Fig 7: Scaling Inversion — 4L vs 16L
# ===================================================================
def fig7_scaling_inversion(fmt="pdf"):
    """Grouped bar chart: MI ratio at 4L vs 16L for each method."""
    methods = ["Var. FFN", "BLoB LoRA", "TFB", "Diag. Laplace"]
    mi_4l =   [1.43,       1.13,        1.10,  1.00]
    mi_16l =  [1.32,       1.53,        1.35,  1.00]

    colors_4l = ["#A9CCE3", "#F5CBA7", "#D2B4DE", "#F5B7B1"]
    colors_16l = [COLORS["var_full"], COLORS["var_lora"],
                  COLORS["post_lora"], COLORS["post_full"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.32

    bars_4l = ax.bar(x - width / 2, mi_4l, width, label="4L (16M params)",
                     color=colors_4l, edgecolor="white", lw=0.5)
    bars_16l = ax.bar(x + width / 2, mi_16l, width, label="16L (76M params)",
                      color=colors_16l, edgecolor="white", lw=0.5)

    # Value labels
    for bars in [bars_4l, bars_16l]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}×", ha="center", va="bottom", fontsize=9)

    # Arrows showing direction of change
    for i, (v4, v16) in enumerate(zip(mi_4l, mi_16l)):
        if v4 != v16:
            direction = "↑" if v16 > v4 else "↓"
            color = COLORS["success"] if v16 > v4 else COLORS["failure"]
            diff = v16 - v4
            ax.text(x[i] + width / 2 + 0.18, max(v4, v16) + 0.02,
                    f"{direction} {diff:+.2f}", fontsize=8,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("MI Ratio (OOD / ID)")
    ax.axhline(y=1.0, color="#CCCCCC", ls="--", lw=0.8)
    ax.text(3.5, 1.02, "No signal", fontsize=8, ha="center", color="#AAAAAA")
    ax.set_ylim(0.8, 1.75)
    ax.legend(loc="upper left")

    # Scaling inversion annotation
    ax.annotate("Scaling inversion:\nLoRA gains, full-weight loses",
                xy=(1.16, 1.53), xytext=(2.5, 1.65),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                          edgecolor=COLORS["accent"]))

    fig.tight_layout()
    _save(fig, "fig7_scaling_inversion", fmt)


# ===================================================================
# Main
# ===================================================================
FIGURE_FUNCS = {
    1: fig1_point_vs_bayesian,
    2: fig2_method_matrix,
    3: fig3_blob_lora,
    4: fig4_tfb_vs_laplace,
    5: fig5_auroc_bars,
    6: fig6_n_vs_auroc,
    7: fig7_scaling_inversion,
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--fig", type=int, nargs="+",
                        help="Figure numbers to generate (default: all)")
    parser.add_argument("--png", action="store_true",
                        help="Output PNG instead of PDF")
    args = parser.parse_args()

    fmt = "png" if args.png else "pdf"
    figs = args.fig or sorted(FIGURE_FUNCS.keys())

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Generating {len(figs)} figure(s) -> {OUT_DIR}/ (.{fmt})")

    for n in figs:
        if n not in FIGURE_FUNCS:
            print(f"  Unknown figure: {n}")
            continue
        print(f"  Fig {n}...", end="")
        FIGURE_FUNCS[n](fmt)

    print("Done.")


if __name__ == "__main__":
    main()
