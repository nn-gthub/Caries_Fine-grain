import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
import open_clip
from sklearn.metrics.pairwise import cosine_similarity

# --- CORRECTED ABSOLUTE PATHS ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
PAL = {'Normal': '#2D6A4F', 'Pre-caries': '#E9A039', 'Caries': '#C1392B', 'Decolor': '#6C3483', 'bg': '#F8F7F4'}

print("Loading data...")
df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])
for col in FG_COLS:
    df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, np.nan:0}).fillna(0).astype(int)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy()

# SMART REPRESENTATIVE SAMPLING
must_keep_idx = set()
for fg in FG_COLS:
    has_fg = df[df[fg] == 1]
    if len(has_fg) > 0:
        must_keep_idx.update(has_fg.sample(min(len(has_fg), 2), random_state=42).index)

df_keep = df.loc[list(must_keep_idx)]
df_pool = df.drop(index=list(must_keep_idx))

final_sample = [df_keep]
for gl in CLASS_ORDER:
    current_count = len(df_keep[df_keep['gross_label'] == gl])
    needed = max(0, 15 - current_count)
    pool_gl = df_pool[df_pool['gross_label'] == gl]
    if needed > 0 and len(pool_gl) > 0:
        final_sample.append(pool_gl.sample(min(len(pool_gl), needed), random_state=42))

df = pd.concat(final_sample).reset_index(drop=True)

# STRUCTURED TEXT BUILDER
def build_structured_text(row):
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note    = str(row.get('clinical_note', '')).strip()
    note    = '' if note in ['nan', 'none', ''] else note
    parts = []
    if present: parts.append(f"Fine-grain features identified: {', '.join(present)}.")
    else: parts.append("No fine-grain features identified.")
    action = str(row.get('recommended_action', '')).strip().lower()
    if action and action not in ['nan', '']: parts.append(f"Recommended action: {action}.")
    if note: parts.append(note)
    return ' '.join(parts)

df['structured_text'] = df.apply(build_structured_text, axis=1)

DEVICE = 'cpu'

# LOAD BIOMEDCLIP (Shared Space)
print("Loading BiomedCLIP to calculate shared-space similarity...")
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_name)
clip_tokenizer = open_clip.get_tokenizer(model_name)
clip_model = clip_model.to(DEVICE).eval()

print("Extracting Image and Text Embeddings...")
image_embs, text_embs, valid_indices = [], [], []

for i, row in df.iterrows():
    img_path = CROP_DIR / row['crop name']
    if img_path.exists():
        img_tensor = clip_preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        text_tensor = clip_tokenizer([row['structured_text']]).to(DEVICE)
        
        with torch.no_grad():
            i_feat = clip_model.encode_image(img_tensor)
            t_feat = clip_model.encode_text(text_tensor)
            
            i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            
            image_embs.append(i_feat.squeeze().numpy())
            text_embs.append(t_feat.squeeze().numpy())
            valid_indices.append(i)

df = df.iloc[valid_indices].reset_index(drop=True)
image_embs = np.array(image_embs)
text_embs = np.array(text_embs)

# 1. Calculate 1-to-1 Cosine Similarities
df['semantic_similarity'] = [cosine_similarity(image_embs[i].reshape(1, -1), text_embs[i].reshape(1, -1))[0][0] for i in range(len(df))]

# 2. Calculate Cross-Modal Heatmap Data (Image Class vs Text Class)
heatmap_data = np.zeros((4, 4))
for i, img_label in enumerate(CLASS_ORDER):
    for j, txt_label in enumerate(CLASS_ORDER):
        img_mask = df['gross_label'] == img_label
        txt_mask = df['gross_label'] == txt_label
        if img_mask.sum() > 0 and txt_mask.sum() > 0:
            img_cent = image_embs[img_mask].mean(axis=0).reshape(1, -1)
            txt_cent = text_embs[txt_mask].mean(axis=0).reshape(1, -1)
            heatmap_data[i, j] = cosine_similarity(img_cent, txt_cent)[0][0]

# --- PLOTTING ---
print("Generating Plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PAL['bg'])

# Plot A: Cross-Modal Heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Blues', 
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, 
            ax=axes[0], cbar_kws={'shrink': 0.8}, linewidths=1, linecolor=PAL['bg'])
axes[0].set_title('Cross-Modal Semantic Similarity\n(Images of Class X vs Text of Class Y)', fontweight='bold', pad=15)
axes[0].set_ylabel('Image Embeddings', fontweight='bold')
axes[0].set_xlabel('Text Embeddings', fontweight='bold')

# Plot B: Similarity Score Distribution
sns.boxplot(data=df, x='gross_label', y='semantic_similarity', order=CLASS_ORDER, palette=PAL, ax=axes[1])
sns.stripplot(data=df, x='gross_label', y='semantic_similarity', order=CLASS_ORDER, color='black', alpha=0.5, ax=axes[1])
axes[1].set_title('1-to-1 Semantic Alignment Score by Class\n(How well does a tooth match its own text?)', fontweight='bold', pad=15)
axes[1].set_ylabel('Cosine Similarity Score')
axes[1].set_xlabel('Gross Label')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('semantic_alignment_plots.png', dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
print("DONE! Plot saved as 'semantic_alignment_plots.png'.")