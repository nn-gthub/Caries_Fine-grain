import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import open_clip
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- YOUR ABSOLUTE PATHS ---
# MAIN_CSV  = '/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv' 
# LOG_CSV   = '/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/gross_label_corrections.csv'  
# CROP_DIR  = Path('/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/tooth_crops/') 

# Metadata CSVs
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"

LOG_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"

# Unified image directory
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")


CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']

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
    needed = max(0, 10 - current_count) # 10 per class for a clean HTML table
    pool_gl = df_pool[df_pool['gross_label'] == gl]
    if needed > 0 and len(pool_gl) > 0:
        final_sample.append(pool_gl.sample(min(len(pool_gl), needed), random_state=42))

df = pd.concat(final_sample).reset_index(drop=True)

# --- EXACT TEXT BUILDER FROM MAIN SCRIPT ---
def build_structured_text(row):
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note    = str(row.get('clinical_note', '')).strip()
    note    = '' if note in ['nan', 'none', ''] else note

    parts = []
    if present:
        parts.append(f"Fine-grain features identified: {', '.join(present)}.")
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
print("Extracting BioBERT (Text) Embeddings on CPU...")
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

# Scale & Combine Multimodal Space
bert_norm = StandardScaler().fit_transform(bert_embs)
clip_norm = StandardScaler().fit_transform(clip_embs)
combined_norm = np.hstack([clip_norm, bert_norm])

# --- MATHEMATICAL EXPLAINABILITY: CENTROID SIMILARITY ---
print("Calculating Mathematical Centroid Similarities...")
centroids = {}
for gl in CLASS_ORDER:
    mask = df['gross_label'] == gl
    if mask.sum() > 0:
        cent = combined_norm[mask].mean(axis=0)
        centroids[gl] = cent.reshape(1, -1)

# Calculate how close each sample is to its OWN class centroid
similarities = []
for i, row in df.iterrows():
    gl = row['gross_label']
    sample_emb = combined_norm[i].reshape(1, -1)
    sim = cosine_similarity(sample_emb, centroids[gl])[0][0]
    similarities.append(sim)

df['centroid_similarity'] = similarities

# Sort to show the "most mathematically perfect" examples first
df = df.sort_values(by=['gross_label', 'centroid_similarity'], ascending=[True, False])

# --- BUILD HTML DASHBOARD ---
print("Building HTML Explainability Report...")
html = """
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #F8F7F4; padding: 20px; }
        table { border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #E8E5DF; padding: 12px; text-align: left; vertical-align: top; }
        th { background-color: #2D6A4F; color: white; font-weight: bold; }
        .score { font-size: 18px; font-weight: bold; color: #C1392B; }
        img { border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
    </style>
</head>
<body>
    <h2>Multimodal Space Validation Report</h2>
    <p>This table sorts crops by how mathematically close their combined Image+Text embedding is to the ideal "Centroid" of their diagnosed class. A higher score means the multimodal space strongly agrees with the clinical diagnosis.</p>
    <table>
        <tr>
            <th width="150">Intraoral Image</th>
            <th width="120">Gross Label</th>
            <th width="150">Multimodal Similarity<br>(Cosine Math)</th>
            <th>Structured Clinical Text (BioBERT Input)</th>
        </tr>
"""

for _, row in df.iterrows():
    img_path = CROP_DIR / row['crop name']
    html += f"""
        <tr>
            <td><img src="{img_path}" width="140"></td>
            <td><b>{row['gross_label']}</b></td>
            <td class="score">{row['centroid_similarity']:.3f}</td>
            <td>{row['structured_text']}</td>
        </tr>
    """

html += """
    </table>
</body>
</html>
"""

with open("multimodal_explainability_report.html", "w") as f:
    f.write(html)

print("\nDONE! Open 'multimodal_explainability_report.html' in VS Code or your browser.")