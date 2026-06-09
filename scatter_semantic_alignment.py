import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
import open_clip
from umap import UMAP

# --- CORRECTED ABSOLUTE PATHS ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
PAL = {'Normal': '#2D6A4F', 'Pre-caries': '#E9A039', 'Caries': '#C1392B', 'Decolor': '#6C3483', 'bg': '#F8F7F4'}

print("Loading and sampling data...")
df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])
for col in FG_COLS:
    df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, np.nan:0}).fillna(0).astype(int)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy()

# SMART REPRESENTATIVE SAMPLING (Keeping it smaller so the plot isn't too cluttered)
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

# LOAD BIOMEDCLIP
print("Loading BiomedCLIP...")
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_name)
clip_tokenizer = open_clip.get_tokenizer(model_name)
clip_model = clip_model.to(DEVICE).eval()

print("Extracting Embeddings...")
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

# --- THE MAGIC TRICK FOR SHARED UMAP ---
# We stack both arrays together before UMAP so they are forced into the EXACT same 2D space mapping
print("Computing Shared UMAP Space...")
combined_for_umap = np.vstack([image_embs, text_embs])
umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
combined_proj = umap_model.fit_transform(combined_for_umap)

# Split them back apart now that they share a coordinate system
n_samples = len(df)
proj_images = combined_proj[:n_samples]
proj_texts = combined_proj[n_samples:]

# --- PLOTTING ---
print("Generating Scatter Plot...")
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(PAL['bg'])
ax.set_facecolor(PAL['bg'])

for i in range(n_samples):
    gl = df['gross_label'].iloc[i]
    color = PAL[gl]
    
    # 1. Draw the connecting line first (so it stays in the background)
    ax.plot([proj_images[i, 0], proj_texts[i, 0]], 
            [proj_images[i, 1], proj_texts[i, 1]], 
            color=color, alpha=0.3, linestyle='--', linewidth=1.5, zorder=1)
    
    # 2. Plot Image Embedding (Circle)
    ax.scatter(proj_images[i, 0], proj_images[i, 1], 
               c=color, marker='o', s=120, edgecolor='white', linewidth=1, zorder=2, alpha=0.8)
    
    # 3. Plot Text Embedding (Star)
    ax.scatter(proj_texts[i, 0], proj_texts[i, 1], 
               c=color, marker='*', s=250, edgecolor='white', linewidth=1, zorder=3, alpha=0.9)
    
    # 4. (Optional) Add Fine-Grain text labels slightly above the stars
    fg_present = [c for c in FG_COLS if df[c].iloc[i] == 1]
    if fg_present:
        ax.text(proj_texts[i, 0], proj_texts[i, 1] + 0.15, '\n'.join(fg_present), 
                fontsize=7, color=color, ha='center', va='bottom', fontweight='bold')

# Formatting
ax.set_title('Shared-Space UMAP: Image vs. Text Semantic Alignment\nConnecting lines show distance between an image and its corresponding text', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks([]); ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_color('#E8E5DF')

# Legend
legend_elements = [
    mlines.Line2D([], [], color='#1A1A1A', marker='o', linestyle='None', markersize=10, label='Image Embedding'),
    mlines.Line2D([], [], color='#1A1A1A', marker='*', linestyle='None', markersize=15, label='Text Embedding'),
    mlines.Line2D([], [], color='#1A1A1A', linestyle='--', alpha=0.5, label='Semantic Gap (Shorter = Better)')
]
# Add color key
for gl in CLASS_ORDER:
    legend_elements.append(mlines.Line2D([], [], color=PAL[gl], marker='s', linestyle='None', markersize=10, label=f'Class: {gl}'))

ax.legend(handles=legend_elements, loc='best', fontsize=10, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig('scatter_semantic_alignment.png', dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
print("DONE! Plot saved as 'scatter_semantic_alignment.png'.")