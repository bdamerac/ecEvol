import os
import matplotlib.pyplot as plt

# per-parameter colours
PHI = "#4C72B0"      # descendant fraction
S = "#1B9E77"        # selection
MU = "#E76F51"       # mutation rate
INTRO = "#7E5DA8"    # emergence / introduction generation
COHORT_PCAWG = "#2C6FBB"
COHORT_HMF = "#C0392B"

MU_CMAP = "Oranges"
INTRO_CMAP = "Purples"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    })


def save_panel(fig, outdir, name):
    """Write {name}.pdf, {name}.svg, and {name}_preview.png to outdir."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}_preview.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(outdir, name))
