import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import open_clip
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# --- YOUR ABSOLUTE PATHS ---
# --- YOUR ABSOLUTE PATHS ---
# MAIN_CSV  = '/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/annotation_clean.csv' 
# LOG_CSV   = '/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/gross_label_corrections.csv'  
# CROP_DIR  = Path('/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/tooth_crops/')

# =========================================================
# CORRECT SERVER PATHS
# =========================================================

from pathlib import Path

# Metadata CSVs
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"

LOG_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"

# Unified image directory
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")



CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
PAL = {'Normal': '#2D6A4F', 'Pre-caries': '#E9A039', 'Caries': '#C1392B', 'Decolor': '#6C3483', 'bg': '#F8F7F4'}

print("Loading and sampling data...")
df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

# Correct labels and normalize checkboxes
label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])
for col in FG_COLS:
    df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, np.nan:0}).fillna(0).astype(int)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy()

# --- SMART REPRESENTATIVE SAMPLING ---
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
print(f"Sampled {len(df)} images. Ensuring all fine-grain features are present.")

# --- EXACT TEXT BUILDER FROM MAIN SCRIPT ---
def build_structured_text(row):
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note    = str(row.get('clinical_note', '')).strip()
    note    = '' if note in ['nan', 'none', ''] else note

    parts = []
    if present:
        feat_str = ', '.join(present)
        parts.append(f"Fine-grain features identified: {feat_str}.")
    else:
        parts.append("No fine-grain features identified.")

    action = str(row.get('recommended_action', '')).strip().lower()
    if action and action not in ['nan', '']:
        parts.append(f"Recommended action: {action}.")

    if note: parts.append(note)
    return ' '.join(parts)

df['structured_text'] = df.apply(build_structured_text, axis=1)

# FORCE CPU
DEVICE = 'cpu'

# --- 1. BIOBERT EXTRACTION ---
print("\nExtracting BioBERT (Text) Embeddings on CPU...")
bert_tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
bert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(DEVICE).eval()

enc = bert_tok(df['structured_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
with torch.no_grad():
    out = bert_model(**enc)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    bert_embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    bert_embs = torch.nn.functional.normalize(bert_embs, dim=-1).numpy()

# --- 2. BIOMEDCLIP EXTRACTION ---
print("Extracting BiomedCLIP (Image) Embeddings on CPU...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
clip_model = clip_model.to(DEVICE).eval()

clip_embs = []
for p in [CROP_DIR / row['crop name'] for _, row in df.iterrows()]:
    if p.exists():
        img = clip_preprocess(Image.open(p).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            feat = clip_model.encode_image(img)
            clip_embs.append((feat / feat.norm(dim=-1, keepdim=True)).squeeze().numpy())
    else:
        clip_embs.append(np.zeros(512))
clip_embs = np.array(clip_embs)

# Scale & Combine
bert_norm = StandardScaler().fit_transform(bert_embs)
clip_norm = StandardScaler().fit_transform(clip_embs)
combined_norm = np.hstack([clip_norm, bert_norm])

# --- UMAP PROJECTION ---
print("Computing UMAP...")
umap_model = UMAP(n_components=2, n_neighbors=5, min_dist=0.3, random_state=42)
proj = umap_model.fit_transform(combined_norm)

# --- PLOT: MULTIMODAL IMAGE MAP ---
print("\nRendering High-Res Multimodal Image Map (This might take a few seconds)...")
fig, ax = plt.subplots(figsize=(26, 22)) # Massive canvas so images don't overlap too much
fig.patch.set_facecolor(PAL['bg'])
ax.set_facecolor(PAL['bg'])

x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
y_range = y_max - y_min
ax.set_xlim(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1)
ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)

for i in range(len(df)):
    x, y_coord = proj[i, 0], proj[i, 1]
    gl = df['gross_label'].iloc[i]
    
    # Get active fine-grain features
    fg = [c for c in FG_COLS if df[c].iloc[i] == 1]
    fg_text = "\n".join(fg) if fg else "None" 

    img_path = CROP_DIR / df['crop name'].iloc[i]
    if img_path.exists():
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((120, 120), Image.Resampling.LANCZOS) # Resize thumbnail
        
        imagebox = OffsetImage(img, zoom=0.9)
        imagebox.image.axes = ax

        # Draw Image with colored Gross Label Border
        ab = AnnotationBbox(imagebox, (x, y_coord), frameon=True,
                            bboxprops=dict(edgecolor=PAL[gl], linewidth=5, facecolor='none'),
                            pad=0.1)
        ax.add_artist(ab)

        # Draw Fine-Grain Text Label beneath the image
        ax.text(x, y_coord - (y_range * 0.035), fg_text, fontsize=9, ha='center', va='top',
                fontweight='bold', color='#1A1A1A',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=PAL[gl], boxstyle='round,pad=0.3'))

# Legend & Formatting
handles = [mpatches.Patch(color=PAL[gl], label=f"Gross Label: {gl}") for gl in CLASS_ORDER]
ax.legend(handles=handles, loc='upper right', fontsize=18, framealpha=0.9)
ax.set_title('Multimodal UMAP Projection (CLIP Image + BioBERT Text)\nImages outlined by Gross Label | Text tags display Fine-Grain Features', 
             fontsize=24, fontweight='bold', pad=20)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig('preview_image_map.png', dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
print("\nDONE! Open 'preview_image_map.png' to see your crops laid out in space.")