"""
Clean regeneration of all five System III result figures.
Style goals:
  - No curved arcs, no AI-looking callouts
  - Horizontal dot/lollipop charts with simple bracket or bracket-arrow gap annotations
  - Consistent seaborn-muted palette, tight layout, no right/top spines
  - All data taken from Table 7.5 / thesis results (edit DATA section if values change)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10.5,
    "axes.titlesize":     11.5,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.color":         "#cccccc",
    "axes.axisbelow":     True,
    "figure.dpi":         180,
})

PAL     = sns.color_palette("muted")
C_GRAY  = "#888888"
C_BLUE  = PAL[0]
C_ORG   = PAL[1]
C_GREEN = PAL[2]
C_RED   = PAL[3]
C_PURP  = PAL[4]
C_BROWN = PAL[5]

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

# ── Table 7.5 (thesis) ───────────────────────────────────────────────────────
CV_METHODS = [
    "Image only\n(BiomedCLIP)",
    "Qwen2-VL 7B\nzero-shot",
    "FG checkbox\nonly",
    "FG text\n(PubMedBERT)",
    "Full Fusion\nXGBoost ★",
    "Pseudo-label\nexpansion",
]
CV_WF1  = [0.504, 0.375, 0.957, 0.949, 0.958, 0.982]
CV_COLS = [C_GRAY, C_GRAY, C_BLUE, C_BLUE, C_RED, C_GREEN]

TEST_METHODS = [
    "Image only\n(frozen CLIP)",
    "Image only\n(fine-tuned)",
    "Pred FG +\nImage (YOLOv8)",
]
TEST_WF1  = [0.639, 0.678, 0.665]
TEST_COLS = [C_GRAY, C_BLUE, C_ORG]

CLASSES       = ["Normal", "Pre-caries", "Caries", "Decolor"]
IMG_F1_PC     = [0.513, 0.548, 0.405, 0.502]
FUSION_F1_PC  = [1.000, 0.929, 0.980, 0.940]

# ── Feature knockout / prevalence ────────────────────────────────────────────
FEAT_LABELS  = ["chalky", "brown", "defect", "fill", "stain", "wear"]
KNOCK_DELTA  = [-0.0029, 0.0057, 0.0141, 0.0000, 0.0029, 0.0000]
# Feature prevalence (%) per class – row = class, col = feature (chalky,brown,defect,fill,stain,wear)
PREVALENCE   = np.array([
    [0,  0,  0,  3,  0,  0],    # Normal
    [24, 52, 8, 11, 18,  2],    # Pre-caries
    [4,  23, 65, 27, 18,  3],   # Caries
    [1,  15,  3,  7, 66,  6],   # Decolor
])

# ── End-to-end pipeline ──────────────────────────────────────────────────────
PIPE_METHODS = [
    "Image only\n(frozen CLIP)",
    "Pred FG + Image\n(CLIP+YOLOv8)",
    "Pred FG + ADA\ntext + Image",
    "Pred FG + LLaVA-1.5\n+ Image",
]
PIPE_WF1    = [0.639, 0.665, 0.368, 0.649]
PIPE_PREC   = [0.338, 0.347, 0.261, 0.359]
PIPE_COLS   = [C_GRAY, C_ORG, C_PURP, C_RED]

CV_UB = 0.958   # cross-validation upper bound

# ── Confusion matrices (from thesis / experiment) ────────────────────────────
CM_IMG = np.array([
    [54, 20,  0, 26],
    [17, 50,  3, 29],
    [ 7, 29, 37, 27],
    [19, 19, 10, 52],
])
CM_FUS = np.array([
    [100,  0,  0,  0],
    [  0, 93,  0,  7],
    [  0,  0, 98,  2],
    [  0,  5,  0, 95],
])
SPEC_VALS = {
    "img_4":  0.481, "fus_4":  0.958,
    "img_3":  0.577, "fus_3":  0.958,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def lollipop_h(ax, values, labels, colors, xlim=(0, 1.05), vline=None, vline_label=None,
               highlight=None, gap_pair=None, gap_label=None):
    """Clean horizontal lollipop chart. No arcs."""
    y = np.arange(len(labels))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(xlim)
    ax.set_xlabel("Weighted F1")
    for yi, (v, c) in enumerate(zip(values, colors)):
        ax.hlines(yi, 0, v, color=c, linewidth=1.4, alpha=0.7, zorder=2)
        ax.scatter(v, yi, color=c, s=55, zorder=3)
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=9, color=c)
    if vline is not None:
        ax.axvline(vline, color="#555555", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
        if vline_label:
            ax.text(vline + 0.015, ax.get_ylim()[1] * 0.98, vline_label,
                    va="top", ha="left", fontsize=8.5, color="#555555", style="italic")
    if highlight is not None:
        ax.hlines(highlight, 0, values[highlight], color=colors[highlight],
                  linewidth=2.5, alpha=1.0, zorder=2)
        ax.scatter(values[highlight], highlight, color=colors[highlight], s=80, zorder=4, marker="*")
    if gap_pair is not None:
        lo_idx, hi_idx = gap_pair
        xlo, xhi = values[lo_idx], values[hi_idx]
        ymid = (lo_idx + hi_idx) / 2
        # Simple double-headed bracket to the right of the dots
        bx = max(values) + 0.07
        ax.annotate("", xy=(bx, hi_idx), xytext=(bx, lo_idx),
                    arrowprops=dict(arrowstyle="<->", color="#cc0000",
                                    lw=1.4, mutation_scale=10))
        ax.vlines(bx, lo_idx, hi_idx, color="#cc0000", lw=1.4)
        ax.text(bx + 0.01, ymid, gap_label, va="center", ha="left",
                fontsize=9, color="#cc0000", fontweight="bold")

def save(fig, path, dims):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    from PIL import Image
    img = Image.open(path)
    print(f"  saved {path}  size={img.size}  target≈{dims}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – fg_7.2.png  ──  Main results comparison (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fg_7.2.png ...")
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8),
                          gridspec_kw={"width_ratios": [1.6, 1.3, 1.2]})
fig.suptitle(
    "Fine-Grain Annotation Improves Dental Caries Classification\n"
    "Cross-validation (annotated) vs Test set (unannotated)",
    fontsize=12.5, fontweight="bold", y=1.01,
)

# Panel 1 – CV lollipop
ax = axes[0]
ax.set_title("5-fold CV\n(FG-annotated crops)", pad=6)
lollipop_h(ax, CV_WF1, CV_METHODS, CV_COLS, xlim=(0, 1.12),
           highlight=4,
           gap_pair=(0, 4),
           gap_label="+0.454\nannotation\nvalue")
ax.set_xlim(0, 1.12)

# Panel 2 – Test set lollipop
ax = axes[1]
ax.set_title("Held-out Test Set\n(623 unannotated crops)", pad=6)
lollipop_h(ax, TEST_WF1, TEST_METHODS, TEST_COLS,
           xlim=(0, 1.0),
           vline=CV_UB, vline_label="CV upper bound\n(with annotation)")
ax.set_xlim(0, 1.0)

# Panel 3 – Per-class grouped bar
ax = axes[2]
ax.set_title("Per-Class F1\n(CV)", pad=6)
x = np.arange(len(CLASSES))
w = 0.35
b1 = ax.barh(x - w/2, IMG_F1_PC,   w, color=C_GRAY,  alpha=0.85, label="Image only")
b2 = ax.barh(x + w/2, FUSION_F1_PC, w, color=C_RED,   alpha=0.85, label="Full Fusion ★")
ax.set_yticks(x)
ax.set_yticklabels(CLASSES, fontsize=9)
ax.set_xlim(0, 1.15)
ax.set_xlabel("F1 Score")
for bar, val in zip(list(b1) + list(b2),
                    IMG_F1_PC + FUSION_F1_PC):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", ha="left", fontsize=8.5)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=1.4)
save(fig, "figs/fg_7.2.png", "(2722,1382)")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – fg_7.6.png  ──  Feature prevalence heatmap + knockout
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fg_7.6.png ...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                          gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("Fine-Grain Feature Distribution and Clinical Importance",
             fontsize=12.5, fontweight="bold", y=1.02)

# Heatmap
ax = axes[0]
ax.set_title("Feature Prevalence by Gross Label (%)", pad=6)
im = ax.imshow(PREVALENCE, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(6)); ax.set_xticklabels(FEAT_LABELS, fontsize=10)
ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES, fontsize=10)
for (i, j), v in np.ndenumerate(PREVALENCE):
    col = "white" if v > 55 else "#333333"
    ax.text(j, i, str(v), ha="center", va="center", fontsize=10, color=col, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.75, label="% crops with feature", pad=0.02)
ax.spines[:].set_visible(False)

# Knockout dot plot
ax = axes[1]
ax.set_title("Feature Knockout: Δ Weighted F1\n(performance drop when feature removed)", pad=6)
feat_sorted = list(zip(FEAT_LABELS, KNOCK_DELTA))
feat_sorted.sort(key=lambda x: x[1])
labels_s, deltas_s = zip(*feat_sorted)
y = np.arange(len(labels_s))
cols_k = [C_RED if d > 0.005 else C_BLUE if d > 0 else C_GRAY for d in deltas_s]
ax.set_yticks(y); ax.set_yticklabels(labels_s, fontsize=10)
ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)
for yi, (d, c) in enumerate(zip(deltas_s, cols_k)):
    ax.hlines(yi, 0, d, color=c, linewidth=2, alpha=0.8, zorder=2)
    ax.scatter(d, yi, color=c, s=60, zorder=3)
    ax.text(d + (0.0005 if d >= 0 else -0.0005),
            yi, f"{d:+.4f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=9.5, color=c, fontweight="bold")
ax.set_xlabel("Performance drop (Δ Weighted F1)")
ax.set_xlim(-0.014, 0.026)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout(pad=1.5)
save(fig, "figs/fg_7.6.png", "(2779,988)")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – fg_7.8.png  ──  Clinically-weighted confusion + caries spectrum
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fg_7.8.png ...")
CMAP = "Blues"
SHORT = ["Norm", "Pre-", "Cari", "Deco"]
fig = plt.figure(figsize=(13, 5.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38, hspace=0.1)
fig.suptitle(
    "Clinically-Weighted Evaluation: Caries Spectrum Analysis\n"
    "Pre-caries + Caries treated as a unified diagnostic category",
    fontsize=11.5, fontweight="bold", y=1.03,
)

for idx, (cm, title, wf1) in enumerate([
    (CM_IMG, f"Image Only — 4-class\nwF1={SPEC_VALS['img_4']:.3f}", SPEC_VALS["img_4"]),
    (CM_FUS, f"Full Fusion ★ — 4-class\nwF1={SPEC_VALS['fus_4']:.3f}", SPEC_VALS["fus_4"]),
]):
    ax = fig.add_subplot(gs[0, idx])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_norm, cmap=CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(SHORT, fontsize=9)
    if idx == 0: ax.set_ylabel("True", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_title(title, fontsize=9.5, pad=5)
    for (i, j), v in np.ndenumerate(cm):
        col = "white" if cm_norm[i, j] > 55 else "#333333"
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9.5, color=col)
    plt.colorbar(im, ax=ax, shrink=0.78, pad=0.03)

# 4-class vs 3-class bar comparison
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title("4-class vs 3-class Performance\n(caries spectrum unified)", fontsize=9.5, pad=5)
cats   = ["Image only\n(4-class)", "Full Fusion\n(4-class)", "Image only\n(3-class)", "Full Fusion\n(3-class)"]
vals   = [SPEC_VALS["img_4"], SPEC_VALS["fus_4"], SPEC_VALS["img_3"], SPEC_VALS["fus_3"]]
colors = [C_GRAY, C_RED, C_GRAY, C_RED]
bars   = ax3.bar(range(4), vals, color=colors, alpha=0.85, width=0.6)
ax3.set_xticks(range(4)); ax3.set_xticklabels(cats, fontsize=8.5)
ax3.set_ylim(0, 1.12); ax3.set_ylabel("Weighted F1")
for bar, v in zip(bars, vals):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.015,
             f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
# Line connecting fusion results
ax3.plot([1, 3], [SPEC_VALS["fus_4"], SPEC_VALS["fus_3"]], "r-o",
         linewidth=1.2, markersize=5, alpha=0.7)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

plt.tight_layout(pad=1.2)
save(fig, "figs/fg_7.8.png", "(1894,814)")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – fg_7.10.png  ──  End-to-end pipeline automatic vs annotated
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fg_7.10.png ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
fig.suptitle(
    "End-to-End Pipeline: Automatic vs Annotated Results\n"
    "Test Set (623 held-out crops)",
    fontsize=12, fontweight="bold", y=1.02,
)

for ax, scores, title, xlabel in [
    (axes[0], PIPE_WF1,  "Weighted F1 — All Classes",     "Weighted F1"),
    (axes[1], PIPE_PREC, "F1 — Pre-caries\n(hardest class)", "F1 Score"),
]:
    y = np.arange(len(PIPE_METHODS))
    ax.set_title(title, pad=6)
    ax.set_yticks(y); ax.set_yticklabels(PIPE_METHODS, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(xlabel)
    for yi, (v, c) in enumerate(zip(scores, PIPE_COLS)):
        ax.hlines(yi, 0, v, color=c, linewidth=1.6, alpha=0.75, zorder=2)
        ax.scatter(v, yi, color=c, s=60, zorder=3)
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=9, color=c)
    ax.axvline(CV_UB, color="#888888", linestyle="--", linewidth=1.1, alpha=0.75, zorder=1)
    ax.text(CV_UB + 0.01, len(PIPE_METHODS) - 0.6,
            f"CV upper bound ({CV_UB})", fontsize=8.5, color="#888888",
            va="top", ha="left", style="italic")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # Legend patch
    handles = [
        mpatches.Patch(color=C_GRAY,  label="Image only"),
        mpatches.Patch(color=C_ORG,   label="+ Predicted FG (YOLOv8)"),
        mpatches.Patch(color=C_PURP,  label="+ ADA text"),
        mpatches.Patch(color=C_RED,   label="+ LLaVA-1.5 text"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.8)

plt.tight_layout(pad=1.5)
save(fig, "figs/fg_7.10.png", "(2383,929)")

print("\nAll done. fg_7.5.png (PCA) not regenerated — requires raw PC coordinates.")
print("Run regen_pca() with your actual PCA output if you want to restyle it.")
