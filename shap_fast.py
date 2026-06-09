import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import json
import re

warnings.filterwarnings('ignore')

import xgboost as xgb
import shap

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

# --- 1. LOAD PRE-COMPUTED EMBEDDINGS ---
print("Loading pre-computed embeddings directly from disk (Lightning Fast)...")
try:
    clip_embs = np.load('clip_embeddings.npy')
    text_embs = np.load('pubmed_embs.npy')
except FileNotFoundError:
    print("ERROR: Could not find .npy files. Make sure you are in the right directory.")
    exit()

# Extract raw Fine-Grain checkboxes
fg_embs = df[FG_COLS].values

# Combine all into Full Fusion X
X_fusion = np.hstack([clip_embs, fg_embs, text_embs])

# --- 2. TRAIN FINAL XGBOOST ---
print("Training final full-dataset XGBoost model...")
clf_fg = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)
clf_fg.fit(X_fusion, y, verbose=False)

# --- 3. THE "MONKEY PATCH" SHAP FIX ---
print("Applying in-memory Monkey Patch to intercept SHAP...")

booster = clf_fg.get_booster()

# Store the original function XGBoost uses to dump its memory
original_save_raw = booster.save_raw

# Create our own function that intercepts SHAP's request
def patched_save_raw(*args, **kwargs):
    # Get the raw memory dump from XGBoost
    model_bytes = original_save_raw(*args, **kwargs)
    
    # If SHAP is asking for a JSON format, intercept and sanitize it mid-air
    if (args and args[0] == "json") or kwargs.get("raw_format") == "json":
        model_str = model_bytes.decode('utf-8')
        # Erase the array and replace it with a safe float
        model_str = re.sub(r'"base_score":\s*"\[.*?\]"', '"base_score": "0.5"', model_str)
        return bytearray(model_str, 'utf-8')
    
    return model_bytes

# Hijack the booster's function
booster.save_raw = patched_save_raw

print("Running SHAP Explainer...")
# Now when SHAP tries to read the booster, it gets tricked by our patched function!
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_fusion)

print("Generating and saving SHAP plot...")
# Feature names for the plot
feat_names = [f"CLIP_img_{i}" for i in range(clip_embs.shape[1])] + \
             FG_COLS + \
             [f"CLIP_txt_{i}" for i in range(text_embs.shape[1])]

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, features=X_fusion, feature_names=feat_names, 
                  class_names=CLASS_ORDER, show=False)
plt.title("Feature Importance: Full Multimodal Fusion", fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('c1_plot5_shap.png', dpi=150, bbox_inches='tight', facecolor='white')

print("\nSUCCESS! SHAP analysis complete. Saved as 'c1_plot5_shap.png'.")