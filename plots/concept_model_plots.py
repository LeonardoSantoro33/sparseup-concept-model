"""
concept_model_plots.py — Figures of the SparseUp Concept Model (same four figures as the SPLADE project).

Fill in the numbers below from docs/RESULTS.md (None = not available yet: that point / figure is skipped).
    python plots/concept_model_plots.py      # writes plots/figures/fig1...fig4 (.png and .pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# figures always go to plots/figures/, whatever folder the script is launched from
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = [256, 512, 1024, 2048, 4096, 8192]  # x axis
N = [None] * len(K)

# --- Reference -------------------------------------------------------------------
REF_NAME = "SparseUp"
REF_DEV = None        # MS MARCO dev MRR@10 of SparseUp (04_baseline_sparseup.py)
REF_BEIR = None       # BEIR mean nDCG@10 of SparseUp (10_eval_beir.py --sparseup)
REF_GB = 13.4         # SparseUp document index, raw (int32 + float32 per non-zero entry); update with the real size

# --- Concept Model alone ---------------------------------------------------------
DEV = {
    "Random W": list(N),
    "Frozen":   list(N),
    "e2e":      list(N),
}
BEIR = {
    "Random W": list(N),
    "Frozen":   list(N),
    "e2e":      list(N),
}

# --- Hybrids (lambda* chosen on MS MARCO validation) -----------------------------
HYB_DEV = {
    "Frozen hybrid": list(N),
    "e2e hybrid":    list(N),
    "hybnoW":        list(N),
}
HYB_BEIR = {
    "Frozen hybrid": list(N),
    "e2e hybrid":    list(N),
    "hybnoW":        list(N),
}

# --- Document storage (GB, raw sizes) ---------------------------------------------
E2E_GB = [round(8.84e6 * 2 * k / 1e9, 1) for k in K]      # dense fp16 vectors g(d)
HYB_E2E_GB = list(N)                                      # SparseUp index + fine-tuned encoder index
HYB_FROZEN_GB = [round(REF_GB + 34_000 * k * 4 / 1e9, 2) for k in K]   # SparseUp index + W (fp32, ~34k x k)


def complete(*seqs):
    """True if every value is available (no None)."""
    return all(v is not None for seq in seqs for v in (seq if isinstance(seq, (list, tuple)) else [seq]))


# --- Style -------------------------------------------------------------------------
# validated categorical palette (blue, orange, aqua): one fixed color per system
COLORS = {
    "Random W": "#2a78d6", "Frozen": "#eb6834", "e2e": "#1baf7a",
    "Frozen hybrid": "#eb6834", "e2e hybrid": "#1baf7a", "hybnoW": "#2a78d6",
    "e2e hybrid (space)": "#2a78d6",
}
MARKERS = {
    "Random W": "s", "Frozen": "o", "e2e": "^",
    "Frozen hybrid": "o", "e2e hybrid": "^", "hybnoW": "D",
    "e2e hybrid (space)": "D",
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
})


def k_axis(ax):
    """Base-2 logarithmic x axis labelled with the k values."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("k (number of concepts)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ref_line(ax, y, text=REF_NAME):
    ax.axhline(y, color=TEXT_2, linestyle="--", linewidth=1.2, zorder=1)
    # label outside the data area, to the right of the line
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", color=TEXT_2, fontsize=9, annotation_clip=False)


def series(ax, x, y, name):
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pts:
        return
    x, y = zip(*pts)
    ax.plot(x, y, color=COLORS[name], marker=MARKERS[name], linewidth=2,
            markersize=6, markeredgecolor="white", markeredgewidth=1, label=name, zorder=3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / name}.png and .pdf")


# ===== Figure 1: Concept Model alone, % of SparseUp =================================
if complete(REF_DEV, REF_BEIR):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, data, ref, title in [
        (a1, DEV, REF_DEV, "In-domain: MS MARCO dev (MRR@10)"),
        (a2, BEIR, REF_BEIR, "Out-of-domain: BEIR (mean nDCG@10)"),
    ]:
        for name, values in data.items():
            series(ax, K, [None if v is None else 100 * v / ref for v in values], name)
        ref_line(ax, 100, f"{REF_NAME} = 100%")
        k_axis(ax)
        ax.set_title(title, color=TEXT, fontsize=11)
    a1.set_ylabel(f"% of {REF_NAME}")
    a1.legend(frameon=False, loc="lower right")
    save(fig, "fig1_percent_of_sparseup")
else:
    print("fig1 skipped: REF_DEV / REF_BEIR not filled in yet")

# ===== Figure 2: hybrids, absolute values ===========================================
if complete(REF_DEV, REF_BEIR):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, ref, title, ylab in [
        (a1, HYB_DEV, REF_DEV, "In-domain: MS MARCO dev", "MRR@10"),
        (a2, HYB_BEIR, REF_BEIR, "Out-of-domain: BEIR", "mean nDCG@10"),
    ]:
        for name, values in data.items():
            series(ax, K, values, name)
        ref_line(ax, ref)
        k_axis(ax)
        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_ylabel(ylab)
    a1.legend(frameon=False, loc="lower left")
    save(fig, "fig2_hybrids")
else:
    print("fig2 skipped: REF_DEV / REF_BEIR not filled in yet")

# ===== Figure 3: storage vs quality (in-domain) =====================================
if complete(REF_DEV, DEV["e2e"], HYB_DEV["e2e hybrid"], HYB_E2E_GB):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    series(ax, E2E_GB, DEV["e2e"], "e2e")
    ax.lines[-1].set_label("e2e alone (dense fp16 vectors)")
    ax.scatter(HYB_E2E_GB, HYB_DEV["e2e hybrid"], color=COLORS["e2e hybrid (space)"],
               marker="D", s=40, edgecolor="white", linewidth=1, zorder=4,
               label=f"e2e hybrid ({REF_NAME} index + encoder index)")
    for gb, v, k in zip(E2E_GB, DEV["e2e"], K):
        ax.annotate(f"k={k}", (gb, v), textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=TEXT_2)
    # label positions for the hybrid points (they are very close to each other)
    HYB_OFFSETS = {256: (-7, 3, "right"), 512: (-7, -9, "right"), 1024: (7, -9, "left"),
                   2048: (7, 0, "left"), 4096: (0, 9, "center"), 8192: (-7, -9, "right")}
    for gb, v, k in zip(HYB_E2E_GB, HYB_DEV["e2e hybrid"], K):
        dx, dy, ha = HYB_OFFSETS[k]
        ax.annotate(f"k={k}", (gb, v), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8, color=TEXT_2)
    ax.scatter([REF_GB], [REF_DEV], color=TEXT, marker="*", s=160, zorder=5,
               label=REF_NAME)
    ax.set_xscale("log")
    ticks = [5, 10, 20, 50, 100, 150]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(3.5, 200)
    ax.set_xlabel("document storage (GB, log scale)")
    ax.set_ylabel("MS MARCO dev MRR@10")
    ax.set_title("Storage vs in-domain quality", color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig3_space_vs_quality")
else:
    print("fig3 skipped: e2e / hybrid numbers not filled in yet")

# ===== Figure 4: total document storage for each k ==================================
if complete(HYB_E2E_GB):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(K))
    width = 0.27
    groups = [
        (E2E_GB, "#1baf7a", "Concept Model alone, frozen or e2e (dense fp16 vectors)"),
        (HYB_FROZEN_GB, "#eb6834", f"Frozen hybrid ({REF_NAME} index + W)"),
        (HYB_E2E_GB, "#2a78d6", f"e2e hybrid ({REF_NAME} index + encoder index)"),
    ]
    for j, (values, color, name) in enumerate(groups):
        bars = ax.bar(x + (j - 1) * (width + 0.02), values, width, color=color, label=name, zorder=3)
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7.5, color=TEXT_2)
    ax.axhline(REF_GB, color=TEXT_2, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate(f"{REF_NAME} ({REF_GB} GB)", xy=(1.0, REF_GB),
                xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", color=TEXT_2, fontsize=9, annotation_clip=False)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100, 150])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "150"])
    ax.set_ylim(3.5, 260)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("k (number of concepts)")
    ax.set_ylabel("document storage (GB, log scale)")
    ax.set_title("Total document storage (8.84M passages)", color=TEXT, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "fig4_total_space")
else:
    print("fig4 skipped: HYB_E2E_GB not filled in yet")
