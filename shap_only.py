import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import xgboost as xgb
import shap
from transformers import AutoTokenizer, AutoModel
import open_clip

# --- NAS BACKUP PATHS ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']

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

# --- 1. LOAD SAVED IMAGE EMBEDDINGS ---
print("Loading pre-computed CLIP embeddings (skipping image processing)...")
try:
    clip_embs = np.load('clip_embeddings.npy')
except FileNotFoundError:
    print("ERROR: 'clip_embeddings.npy' not found in this folder. You must run this in the same folder as your main script.")
    exit()

# --- 2. QUICKLY REBUILD TEXT FEATURES ---
print("Rebuilding text embeddings...")
def build_structured_text(row):
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note = str(row.get('clinical_note', '')).strip()
    note = '' if note in ['nan', 'none', ''] else note
    parts = []
    if present: parts.append(f"Fine-grain features identified: {', '.join(present)}.")
    else: parts.append("No fine-grain features identified.")
    action = str(row.get('recommended_action', '')).strip().lower()
    if action and action not in ['nan', '']: parts.append(f"Recommended action: {action}.")
    if note: parts.append(note)
    return ' '.join(parts)

df['structured_text'] = df.apply(build_structured_text, axis=1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# We use PubMedCLIP text encoder to match your best performing 'Full Fusion' result
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
clip_tokenizer = open_clip.get_tokenizer(model_name)
clip_model = clip_model.to(DEVICE).eval()

text_embs = []
# Fast batch processing for text
with torch.no_grad():
    for text in df['structured_text']:
        text_tensor = clip_tokenizer([text]).to(DEVICE)
        t_feat = clip_model.encode_text(text_tensor)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        text_embs.append(t_feat.squeeze().cpu().numpy())
text_embs = np.array(text_embs)

# Extract raw Fine-Grain checkboxes
fg_embs = df[FG_COLS].values

# Combine all into Full Fusion X
X_fusion = np.hstack([clip_embs, fg_embs, text_embs])

# --- 3. TRAIN FINAL XGBOOST ---
print("Training final full-dataset XGBoost model...")
clf_fg = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)
clf_fg.fit(X_fusion, y, verbose=False)

# --- 4. THE SHAP FIX & PLOT ---
print("Running SHAP Explainer (with compatibility fix)...")
# The crucial fix to prevent the ValueError crash
booster = clf_fg.get_booster()
booster.set_param({"base_score": 0.5})

# Generate SHAP
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_fusion)

print("Generating and saving SHAP plot safely...")
# Feature names for the plot
feat_names = [f"CLIP_img_{i}" for i in range(clip_embs.shape[1])] + \
             FG_COLS + \
             [f"CLIP_txt_{i}" for i in range(text_embs.shape[1])]

plt.figure(figsize=(10, 8))
# SHAP returns a list of arrays for multi-class; we plot the summary for all classes
shap.summary_plot(shap_values, features=X_fusion, feature_names=feat_names, 
                  class_names=CLASS_ORDER, show=False)
plt.title("Feature Importance: Full Multimodal Fusion", fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('c1_plot5_shap.png', dpi=150, bbox_inches='tight', facecolor='white')

print("\nSUCCESS! SHAP analysis complete. Saved purely as 'c1_plot5_shap.png' without modifying previous results.")