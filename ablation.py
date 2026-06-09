# =============================================================================
# EXPERIMENT 1: ABLATION — Isolate Each Fine-Grain Feature's Role (VERBOSE)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
GROSS_LABEL_NAMES = ["Normal", "Pre-caries", "Caries", "Decolor"]
FG_FEATURE_NAMES  = ["stain", "defect", "brown", "chalky", "fill", "wear"]

# --- 1. LOAD DATA ---
print("[1/5] Loading real dataset and embeddings...")
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
fg_features = df[FG_FEATURE_NAMES]

try:
    image_embeddings = np.load('clip_embeddings.npy')
    text_embeddings = np.load('pubmed_embs.npy')
except FileNotFoundError:
    print("ERROR: .npy files not found. Run this in the same directory as your extracted embeddings.")
    exit()

print(f"✓ Data loaded successfully: {len(gross_labels)} samples.")

# --- 2. VERBOSE EVALUATION ENGINE ---
def evaluate_features_verbose(X, y, n_splits=5, label=""):
    """5-fold stratified CV with real-time print tracking."""
    print(f"\n[Evaluating Configuration]: {label}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    clf = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=100,
        tree_method='hist',   # <-- ADD THIS
        device='cuda',        # <-- ADD THIS
        random_state=RANDOM_STATE
    )
    
    all_true, all_pred = [], []
    
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        print(f"   -> Training Cross-Validation Fold {fold}/{n_splits}...", end="\r")
        clf.fit(X[tr], y[tr], verbose=False)
        all_pred.extend(clf.predict(X[te]))
        all_true.extend(y[te])
    
    print(f"   ✓ Fold training complete for {label}. Computing metrics...      ")
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    per_class = f1_score(all_true, all_pred, average=None, labels=[0,1,2,3])
    wf1 = f1_score(all_true, all_pred, average="weighted")
    kappa = cohen_kappa_score(all_true, all_pred)

    result = {"label": label, "Weighted F1": round(wf1, 4), "Kappa": round(kappa, 4)}
    for i, cls in enumerate(GROSS_LABEL_NAMES):
        result[f"F1_{cls}"] = round(per_class[i], 4)
    return result

# --- 3. PART A: FEATURE KNOCKOUT ---
print("\n[2/5] Initiating Feature Knockout Analysis...")
full_X = np.hstack([image_embeddings, text_embeddings, fg_features.values.astype(float)])
knockout_results = []

knockout_results.append(evaluate_features_verbose(image_embeddings, gross_labels, label="Image only (baseline)"))
knockout_results.append(evaluate_features_verbose(full_X, gross_labels, label="Full fusion (all FG)"))

for idx, feat in enumerate(FG_FEATURE_NAMES, start=1):
    print(f"\n Knockout Progress: [{idx}/{len(FG_FEATURE_NAMES)}] handling features...")
    remaining_fg = fg_features.drop(columns=[feat]).values.astype(float)
    X_ko = np.hstack([image_embeddings, text_embeddings, remaining_fg])
    knockout_results.append(evaluate_features_verbose(X_ko, gross_labels, label=f"Remove '{feat}'"))

ko_df = pd.DataFrame(knockout_results)

# --- 4. PART B: INCREMENTAL ADDITION ---
print("\n[3/5] Initiating Incremental Addition Analysis...")
FG_ORDER = ["stain", "defect", "brown", "chalky", "fill", "wear"]
incremental_results = []

incremental_results.append(evaluate_features_verbose(image_embeddings, gross_labels, label="Base Layer: Image only"))
incremental_results.append(evaluate_features_verbose(np.hstack([image_embeddings, text_embeddings]), gross_labels, label="Base + Text (structured)"))

running_fg_cols = []
for idx, feat in enumerate(FG_ORDER, start=1):
    print(f"\n Addition Progress: [{idx}/{len(FG_ORDER)}] injecting fine-grain signals...")
    running_fg_cols.append(feat)
    X_running = np.hstack([image_embeddings, text_embeddings, fg_features[running_fg_cols].values.astype(float)])
    incremental_results.append(evaluate_features_verbose(X_running, gross_labels, label=f"+ Stepwise feature: {feat}"))

inc_df = pd.DataFrame(incremental_results)

# --- 5. PLOTTING ---
# --- 5. PLOTTING ---
print("\n[4/5] Saving raw data and plotting ablation charts...")

# SAFETY NET: Save the raw results to CSVs just in case!
ko_df.to_csv("exp1_knockout_raw_results.csv", index=False)
inc_df.to_csv("exp1_incremental_raw_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#f8f4ef")
full_wf1 = ko_df.loc[ko_df["label"]=="Full fusion (all FG)", "Weighted F1"].values[0]

# Create the plotting table
ko_only = ko_df[ko_df["label"].str.startswith("Remove")].copy()
ko_only["WF1_drop"] = full_wf1 - ko_only["Weighted F1"]

# Extract the feature names properly into a 1D column
ko_only["feat"] = ko_only["label"].str.extract(r"'(.+)'").iloc[:, 0]

colors = ["#c0392b" if d > 0.005 else "#2980b9" for d in ko_only["WF1_drop"]]
ax = axes[0]
bars = ax.bar(ko_only["feat"], ko_only["WF1_drop"], color=colors, edgecolor="white", linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Weighted F1 Drop When Feature Removed", fontsize=12, fontweight="bold")
ax.set_xlabel("Removed Fine-Grain Feature")
ax.set_ylabel("Δ Weighted F1 vs Full Fusion")
for bar, val in zip(bars, ko_only["WF1_drop"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, f"{val:+.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# THE FIX: Use ko_only instead of ko_df to build the heatmap!
hm_data = ko_only.set_index("feat")[[f"F1_{c}" for c in GROSS_LABEL_NAMES]].rename(columns=lambda c: c.replace("F1_",""))
sns.heatmap(hm_data.astype(float), annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0, ax=axes[1], linewidths=0.5)
axes[1].set_title("Per-Class F1 — Feature Knockout Heatmap", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("exp1a_knockout.png", dpi=150, bbox_inches="tight", facecolor="#f8f4ef")

fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#f8f4ef")
x_ticks = range(len(inc_df))
ax1.plot(x_ticks, inc_df["Weighted F1"], marker="o", color="#2ecc71", linewidth=2.5, label="Weighted F1")
ax1.axvspan(1.5, len(inc_df)-0.5, alpha=0.08, color="#e74c3c", label="FG feature additions")
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(inc_df["label"].tolist(), rotation=35, ha="right")
ax1.set_title("Incremental Feature Addition Performance", fontweight="bold")
ax1.set_ylabel("Weighted F1 Score")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("exp1b_incremental.png", dpi=150, bbox_inches="tight", facecolor="#f8f4ef")

print("\n[5/5] All tasks complete! Saved 'exp1a_knockout.png' and 'exp1b_incremental.png' successfully.")