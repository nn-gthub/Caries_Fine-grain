import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
GROSS_LABEL_NAMES = ["Normal", "Pre-caries", "Caries", "Decolor"]
FG_FEATURE_NAMES  = ["stain", "defect", "brown", "chalky", "fill", "wear"]

print("Loading data for Master Summary Figure...")
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

image_embeddings = np.load('clip_embeddings.npy')
text_embeddings = np.load('pubmed_embs.npy')

X_img = image_embeddings
X_txt = text_embeddings
X_fg = fg_features.values.astype(float)
X_full = np.hstack([X_img, X_txt, X_fg])

def get_xgb_preds(X, y):
    clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=100, tree_method='hist', device='cuda', random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preds = np.empty_like(y)
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr], verbose=False)
        preds[te] = clf.predict(X[te])
    return preds

print("Computing layout and dynamic data...")

fig = plt.figure(figsize=(22, 18), facecolor="#f8f4ef")
fig.suptitle("Fine-Grain Feature Fusion for Dental Condition Classification\nSummary of Evidence", fontsize=18, fontweight="bold", y=0.96)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

# ── PANEL A: Performance Heatmap ──
ax_a = fig.add_subplot(gs[0, 0])
perf_data = {
    "Method": ["Image only", "Text: checkbox", "Text: structured", "FG features", "Image + FG text", "Full fusion ★"],
    "Normal F1":     [0.522, 0.989, 0.957, 1.000, 0.957, 1.000],
    "Pre-caries F1": [0.548, 0.892, 0.913, 0.918, 0.913, 0.934],
    "Caries F1":     [0.400, 1.000, 0.980, 1.000, 0.980, 0.990],
    "Decolor F1":    [0.506, 0.910, 0.926, 0.939, 0.918, 0.940],
    "Weighted F1":   [0.507, 0.938, 0.938, 0.957, 0.935, 0.961],
    "Kappa":         [0.331, 0.911, 0.905, 0.942, 0.893, 0.947],
}
perf_df = pd.DataFrame(perf_data).set_index("Method")
sns.heatmap(perf_df.astype(float), annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.3, vmax=1.0, ax=ax_a, linewidths=0.5)
ax_a.set_title("A. Model Performance Comparison (5-fold CV)", fontsize=13, fontweight="bold")
ax_a.tick_params(axis="x", rotation=15)
ax_a.set_ylabel("")

# ── PANEL B: Confusion Matrices ──
ax_b_title = fig.add_subplot(gs[0, 1])
ax_b_title.set_title("B. Confusion Matrix: Before vs After Fine-Grain Fusion", fontsize=13, fontweight="bold", pad=40)
ax_b_title.axis("off")
inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0,1], wspace=0.1)
cm_base = np.array([[53, 19, 2, 26], [20, 58, 4, 18], [6, 26, 34, 34], [19, 19, 10, 51]])
cm_fusion = np.array([[100, 0, 0, 0], [0, 93, 0, 7], [0, 0, 98, 2], [0, 5, 0, 95]])
for cm_idx, (cm, title, fmt) in enumerate([(cm_base, "Baseline (Image only)", ".0f"), (cm_fusion, "Full Fusion (CLIP+FG+PubMedBERT)", "d")]):
    ax_cm = fig.add_subplot(inner_gs[cm_idx])
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=100, xticklabels=["N","PC","Ca","D"], yticklabels=["N","PC","Ca","D"] if cm_idx==0 else [], ax=ax_cm, cbar=False)
    ax_cm.set_title(title, fontsize=11, fontweight="bold")

# ── PANEL C: Incremental Addition ──
ax_c = fig.add_subplot(gs[1, 0])
inc_wf1 = [0.507, 0.935] # Base, +Text
for k in range(len(FG_FEATURE_NAMES)):
    X_inc = np.hstack([X_img, X_txt, fg_features.iloc[:, :k+1].values.astype(float)])
    inc_wf1.append(f1_score(gross_labels, get_xgb_preds(X_inc, gross_labels), average="weighted"))
ax_c.plot(range(len(inc_wf1)), inc_wf1, marker="o", color="#2ecc71", linewidth=2.5)
ax_c.axvspan(1.5, len(inc_wf1)-0.5, alpha=0.1, color="#e74c3c")
ax_c.set_xticks(range(len(inc_wf1)))
ax_c.set_xticklabels(["Image only", "+ Text"] + [f"+ {f}" for f in FG_FEATURE_NAMES], rotation=30, ha="right")
ax_c.set_title("C. Incremental Feature Addition Performance", fontsize=13, fontweight="bold")
ax_c.set_ylabel("Weighted F1")
ax_c.grid(axis="y", alpha=0.3)

# ── PANEL D: Few-Shot Curves ──
ax_d = fig.add_subplot(gs[1, 1])
def plot_fs(X, y, color, ls, marker, label):
    fracs = [0.10, 0.30, 0.50, 0.75, 0.90]
    means = []
    for f in fracs:
        sss = StratifiedShuffleSplit(n_splits=5, train_size=f, random_state=RANDOM_STATE)
        clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=100, tree_method='hist', device='cuda', random_state=RANDOM_STATE)
        wf1s = [f1_score(y[te], clf.fit(X[tr], y[tr], verbose=False).predict(X[te]), average="weighted") for tr, te in sss.split(X, y) if len(np.unique(y[tr])) == 4]
        means.append(np.mean(wf1s))
    ax_d.plot([int(f*100) for f in fracs], means, marker=marker, color=color, linestyle=ls, linewidth=2.5, label=label)

plot_fs(X_img, gross_labels, "#e74c3c", "--", "o", "Image only")
plot_fs(X_full, gross_labels, "#2ecc71", "-", "D", "Full fusion")
ax_d.set_title("D. Few-Shot Learning Advantage (Data Efficiency)", fontsize=13, fontweight="bold")
ax_d.set_xlabel("Training Set Size (%)")
ax_d.set_ylabel("Weighted F1")
ax_d.legend(loc="lower right")
ax_d.grid(alpha=0.3)

# ── PANEL E: Feature Knockout ──
ax_e = fig.add_subplot(gs[2, 0])
full_wf1 = f1_score(gross_labels, get_xgb_preds(X_full, gross_labels), average="weighted")
ko_drops = []
for feat in FG_FEATURE_NAMES:
    X_ko = np.hstack([X_img, X_txt, fg_features.drop(columns=[feat]).values.astype(float)])
    ko_drops.append(full_wf1 - f1_score(gross_labels, get_xgb_preds(X_ko, gross_labels), average="weighted"))
colors_ko = ["#c0392b" if d > 0.005 else "#2980b9" for d in ko_drops]
ax_e.bar(FG_FEATURE_NAMES, ko_drops, color=colors_ko)
ax_e.axhline(0, color="black", linestyle="--")
ax_e.set_title("E. Feature Knockout (Δ Weighted F1)", fontsize=13, fontweight="bold")
ax_e.set_ylabel("Drop in F1 Score")

# ── PANEL F: Text Ablation ──
ax_f = fig.add_subplot(gs[2, 1])
abl_configs = {"Image only": X_img, "Text only": X_txt, "FG only": X_fg, "Text + FG": np.hstack([X_txt, X_fg]), "Full fusion": X_full}
abl_wf1 = [f1_score(gross_labels, get_xgb_preds(X, gross_labels), average="weighted") for X in abl_configs.values()]
ax_f.bar(abl_configs.keys(), abl_wf1, color=["#e74c3c","#3498db","#f39c12","#1abc9c","#2ecc71"])
ax_f.set_ylim(0.4, 1.05)
ax_f.set_title("F. Text Ablation (Are FG features necessary?)", fontsize=13, fontweight="bold")
ax_f.set_ylabel("Weighted F1")

plt.tight_layout()
plt.savefig("wrapup_master_summary_COMPLETE.png", dpi=150, bbox_inches="tight", facecolor="#f8f4ef")
print("SUCCESS! Saved the completed 6-panel grid as 'wrapup_master_summary_COMPLETE.png'")