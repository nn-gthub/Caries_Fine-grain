# =============================================================================
# EXPERIMENT 3: LOW-LABEL / FEW-SHOT REGIME (REAL DATA + GPU PIPELINE)
# Proves fine-grain features are especially valuable when labeled data is scarce
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, cohen_kappa_score
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE   = 42
N_REPEATS      = 5  # Reduced slightly from 10 to ensure it finishes comfortably overnight
TRAIN_FRACS    = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90]
GROSS_LABEL_NAMES = ["Normal", "Pre-caries", "Caries", "Decolor"]
FG_FEATURE_NAMES  = ["stain", "defect", "brown", "chalky", "fill", "wear"]

print("Loading real dataset and pre-computed embeddings...")
# --- DATA LOADING ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"

df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])
for col in FG_FEATURE_NAMES:
    df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, np.nan:0}).fillna(0).astype(int)
df = df[df['gross_label'].isin(GROSS_LABEL_NAMES)].copy().reset_index(drop=True)

label_map = {label: idx for idx, label in enumerate(GROSS_LABEL_NAMES)}
gross_labels = df['gross_label'].map(label_map).values
fg_features = df[FG_FEATURE_NAMES].values.astype(float)

try:
    image_embeddings = np.load('clip_embeddings.npy')
    text_embeddings = np.load('pubmed_embs.npy')
except FileNotFoundError:
    print("ERROR: .npy files not found. Ensure you are in the right directory.")
    exit()

print(f"Real Data Loaded: {len(gross_labels)} samples.")

# --- BUILD FEATURE SETS ---
feature_sets = {
    "Image only\n(BiomedCLIP)":          image_embeddings,
    "Text only\n(PubMedBERT)":          text_embeddings,
    "Image + Text\n(CLIP + PubMedBERT)": np.hstack([image_embeddings, text_embeddings]),
    "Full fusion\n(CLIP + FG + PubMedBERT)": np.hstack([image_embeddings, fg_features, text_embeddings]),
}

# --- FEW-SHOT ENGINE ---
def run_fewshot(X, y, train_fracs, n_repeats, label):
    rows = []
    for frac in train_fracs:
        wf1s, kappas = [], []
        sss = StratifiedShuffleSplit(n_splits=n_repeats, train_size=frac, random_state=RANDOM_STATE)
        
        for tr, te in sss.split(X, y):
            if len(np.unique(y[tr])) < 4:  # Skip degenerate splits missing a class
                continue
                
            # Configured explicitly to execute on GPU 2 to prevent collisions
            clf = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=100,
                tree_method='hist',
                device='cuda:2',
                random_state=RANDOM_STATE
            )
            
            clf.fit(X[tr], y[tr], verbose=False)
            pred = clf.predict(X[te])
            wf1s.append(f1_score(y[te], pred, average="weighted"))
            kappas.append(cohen_kappa_score(y[te], pred))
            
        if len(wf1s) > 0:
            rows.append({
                "model":        label,
                "train_frac":   frac,
                "train_pct":    int(frac * 100),
                "wf1_mean":     np.mean(wf1s),
                "wf1_std":      np.std(wf1s),
                "kappa_mean":   np.mean(kappas),
                "kappa_std":    np.std(kappas),
                "n_train_avg":  int(frac * len(y)),
            })
    return rows

print("\nStarting Few-Shot Training Loops across all configurations...")
all_rows = []
for model_name, X in feature_sets.items():
    print(f"  Training: {model_name.replace(chr(10),' ')}...")
    all_rows.extend(run_fewshot(X, gross_labels, TRAIN_FRACS, N_REPEATS, model_name))

fs_df = pd.DataFrame(all_rows)

# --- PLOTTING ---
print("\nGenerating Few-Shot Learning Curves...")
MODEL_COLORS = {
    "Image only\n(BiomedCLIP)":          "#e74c3c",
    "Text only\n(PubMedBERT)":          "#3498db",
    "Image + Text\n(CLIP + PubMedBERT)":     "#f39c12",
    "Full fusion\n(CLIP + FG + PubMedBERT)": "#2ecc71",
}
MODEL_STYLES = {
    "Image only\n(BiomedCLIP)":          ("--", "o"),
    "Text only\n(PubMedBERT)":          (":",  "s"),
    "Image + Text\n(CLIP + PubMedBERT)":     ("-.", "^"),
    "Full fusion\n(CLIP + FG + PubMedBERT)": ("-",  "D"),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#f8f4ef")

for ax_idx, (metric, metric_std, ylabel, title) in enumerate([
    ("wf1_mean",   "wf1_std",   "Weighted F1",    "Learning Curves — Weighted F1"),
    ("kappa_mean", "kappa_std", "Cohen's Kappa",   "Learning Curves — Cohen's Kappa"),
]):
    ax = axes[ax_idx]
    for model_name, grp in fs_df.groupby("model"):
        grp = grp.sort_values("train_pct")
        color = MODEL_COLORS[model_name]
        ls, marker = MODEL_STYLES[model_name]
        lw = 3 if "Full fusion" in model_name else 1.8

        ax.plot(grp["train_pct"], grp[metric],
                color=color, linestyle=ls, marker=marker,
                linewidth=lw, markersize=7, label=model_name.replace("\n", " "),
                zorder=3 if "Full fusion" in model_name else 2)
        ax.fill_between(grp["train_pct"],
                        grp[metric] - grp[metric_std],
                        grp[metric] + grp[metric_std],
                        alpha=0.12, color=color)

    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 95)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="both", alpha=0.25)
    ax.set_facecolor("#f8f4ef")

plt.suptitle("Few-Shot Experiment: Does Fine-Grain Fusion Need Less Data?", fontsize=14, y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig("exp3_fewshot_curves.png", dpi=150, bbox_inches="tight", facecolor="#f8f4ef")

# --- GAP PLOT ---
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#f8f4ef")
full_wf1 = fs_df[fs_df["model"]=="Full fusion\n(CLIP + FG + PubMedBERT)"][["train_pct","wf1_mean"]].set_index("train_pct")
img_wf1  = fs_df[fs_df["model"]=="Image only\n(BiomedCLIP)"][["train_pct","wf1_mean"]].set_index("train_pct")
gap = (full_wf1["wf1_mean"] - img_wf1["wf1_mean"]).reset_index()
gap.columns = ["train_pct", "gap"]

colors_gap = ["#e74c3c" if g < 0 else "#2ecc71" for g in gap["gap"]]
bars = ax.bar(gap["train_pct"].astype(str) + "%", gap["gap"], color=colors_gap, edgecolor="white", linewidth=1.2, width=0.5)
ax.axhline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, gap["gap"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.003 if val >= 0 else -0.015), f"{val:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Training Set Size (%)", fontsize=12)
ax.set_ylabel("F1 Gap: Full Fusion − Image Only", fontsize=12)
ax.set_title("Advantage of Fine-Grain Fusion Over Image-Only Across Dataset Sizes", fontsize=12, fontweight="bold")
ax.set_facecolor("#f8f4ef")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("exp3_fewshot_gap.png", dpi=150, bbox_inches="tight", facecolor="#f8f4ef")

print("\nSUCCESS! Saved 'exp3_fewshot_curves.png' and 'exp3_fewshot_gap.png' safely.")