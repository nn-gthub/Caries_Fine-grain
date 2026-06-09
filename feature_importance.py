import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import xgboost as xgb

# --- NAS BACKUP PATHS ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
PAL = ['#2D6A4F', '#E9A039', '#C1392B'] # Green, Orange, Red for plotting

print("Loading dataset...")
df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])
for col in FG_COLS:
    df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, np.nan:0}).fillna(0).astype(int)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)

# Encode Labels
label_map = {label: idx for idx, label in enumerate(CLASS_ORDER)}
y = df['gross_label'].map(label_map).values

# --- 1. LOAD PRE-COMPUTED EMBEDDINGS ---
print("Loading pre-computed embeddings...")
clip_embs = np.load('clip_embeddings.npy')
text_embs = np.load('pubmed_embs.npy')
fg_embs = df[FG_COLS].values

X_fusion = np.hstack([clip_embs, fg_embs, text_embs])

# --- 2. TRAIN FINAL XGBOOST ---
print("Training final XGBoost model...")
clf_fg = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)
clf_fg.fit(X_fusion, y, verbose=False)

# --- 3. NATIVE FEATURE IMPORTANCE (CRASH-PROOF) ---
print("Extracting native modality contributions...")
# Get the native 'Gain' importance for all 1030 features
importances = clf_fg.feature_importances_

# Aggregate the importances by their source modality
num_img = clip_embs.shape[1]
num_fg  = len(FG_COLS)

img_importance = np.sum(importances[0 : num_img])
fg_importance  = np.sum(importances[num_img : num_img + num_fg])
txt_importance = np.sum(importances[num_img + num_fg : ])

# Convert to percentages
total = img_importance + fg_importance + txt_importance
modalities = {
    'Visual Features\n(Intraoral Images)': (img_importance / total) * 100,
    'Clinical Knowledge\n(Structured Text)': (txt_importance / total) * 100,
    'Fine-Grain Tags\n(Checkbox Vectors)': (fg_importance / total) * 100
}

# Also grab the individual importance of the 6 fine-grain tags for a sub-plot
fg_individual = {FG_COLS[i]: importances[num_img + i] for i in range(num_fg)}
fg_individual = dict(sorted(fg_individual.items(), key=lambda item: item[1], reverse=True))

# --- PLOTTING ---
print("Generating Final Paper Plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#F8F7F4')

# Plot A: Modality Contribution
sns.barplot(x=list(modalities.keys()), y=list(modalities.values()), ax=axes[0], palette=PAL)
axes[0].set_title('Overall Modality Contribution to Model Accuracy', fontweight='bold', pad=15)
axes[0].set_ylabel('Relative Importance (%)')
for i, v in enumerate(modalities.values()):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Plot B: Fine-Grain Breakdown
sns.barplot(x=list(fg_individual.keys()), y=list(fg_individual.values()), ax=axes[1], palette='Blues_r')
axes[1].set_title('Most Predictive Fine-Grain Clinical Features', fontweight='bold', pad=15)
axes[1].set_ylabel('Native XGBoost Gain')

plt.tight_layout()
plt.savefig('c1_plot5_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#F8F7F4')

print("\nSUCCESS! Saved pure, crash-proof native plot as 'c1_plot5_feature_importance.png'.")