import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import base64
from io import BytesIO
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
    needed = max(0, 10 - current_count)
    pool_gl = df_pool[df_pool['gross_label'] == gl]
    if needed > 0 and len(pool_gl) > 0:
        final_sample.append(pool_gl.sample(min(len(pool_gl), needed), random_state=42))

df = pd.concat(final_sample).reset_index(drop=True)

# --- STRUCTURED TEXT BUILDER ---
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

DEVICE = 'cpu'

# --- LOAD BIOMEDCLIP (For both Image and Text) ---
print("Loading BiomedCLIP to calculate shared-space similarity...")
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_name)
clip_tokenizer = open_clip.get_tokenizer(model_name)
clip_model = clip_model.to(DEVICE).eval()

print("Extracting Embeddings...")
image_embs = []
text_embs = []
valid_indices = []
base64_images = []

for i, row in df.iterrows():
    img_path = CROP_DIR / row['crop name']
    if img_path.exists():
        # Process Image
        raw_img = Image.open(img_path).convert('RGB')
        img_tensor = clip_preprocess(raw_img).unsqueeze(0).to(DEVICE)
        
        # Process Text
        text_tensor = clip_tokenizer([row['structured_text']]).to(DEVICE)
        
        with torch.no_grad():
            i_feat = clip_model.encode_image(img_tensor)
            t_feat = clip_model.encode_text(text_tensor)
            
            i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            
            image_embs.append(i_feat.squeeze().numpy())
            text_embs.append(t_feat.squeeze().numpy())
            valid_indices.append(i)
            
            # Convert image to Base64 for bulletproof HTML rendering
            raw_img.thumbnail((150, 150))
            buffered = BytesIO()
            raw_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(f"data:image/jpeg;base64,{img_str}")

df = df.iloc[valid_indices].reset_index(drop=True)
image_embs = np.array(image_embs)
text_embs = np.array(text_embs)

# --- CALCULATE DIRECT IMAGE-TO-TEXT SIMILARITY ---
print("Calculating Semantic Similarity...")
similarities = [cosine_similarity(image_embs[i].reshape(1, -1), text_embs[i].reshape(1, -1))[0][0] for i in range(len(df))]
df['semantic_similarity'] = similarities

df['b64_img'] = base64_images
df = df.sort_values(by=['gross_label', 'semantic_similarity'], ascending=[True, False])

# --- BUILD SELF-CONTAINED HTML ---
print("Building Base64 HTML Report...")
html = """
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #F8F7F4; padding: 20px; }
        table { border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #E8E5DF; padding: 12px; text-align: left; vertical-align: top; }
        th { background-color: #2563EB; color: white; font-weight: bold; }
        .score { font-size: 18px; font-weight: bold; color: #1A1A1A; }
        img { border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
    </style>
</head>
<body>
    <h2>Direct Image-Text Semantic Similarity</h2>
    <p>This calculates the Cosine Similarity between the actual image crop and its corresponding clinical text block in the shared BiomedCLIP space. Higher scores indicate the model sees strong semantic alignment between the visual features and the text description.</p>
    <table>
        <tr>
            <th width="160">Intraoral Image</th>
            <th width="120">Gross Label</th>
            <th width="150">Image vs Text<br>Cosine Similarity</th>
            <th>Structured Clinical Text</th>
        </tr>
"""

for _, row in df.iterrows():
    html += f"""
        <tr>
            <td><img src="{row['b64_img']}" width="150"></td>
            <td><b>{row['gross_label']}</b></td>
            <td class="score">{row['semantic_similarity']:.3f}</td>
            <td>{row['structured_text']}</td>
        </tr>
    """

html += """
    </table>
</body>
</html>
"""

with open("direct_semantic_similarity_report.html", "w") as f:
    f.write(html)

print("\nDONE! You can now safely download and open 'direct_semantic_similarity_report.html' anywhere.")