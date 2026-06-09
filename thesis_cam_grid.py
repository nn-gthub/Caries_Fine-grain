import torch
import open_clip
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']

print("Loading BiomedCLIP Model (Offline Mode)...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(DEVICE).eval()

# --- PRE-PROCESS DATASET ---
print("Scanning dataset for ideal row candidates...")
df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)
label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])
for col in FG_COLS:
    df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, np.nan:0}).fillna(0).astype(int)

# Target selections to showcase specific features where they organically happen
targets = {
    'Normal':     {'features': [('fill', 'dental filling/restoration'), ('wear', 'incisal/occlusal tooth wear')], 'query': df[df['gross_label'] == 'Normal']},
    'Pre-caries': {'features': [('chalky', 'chalky white spot lesion'), ('brown', 'brown carious discoloration')], 'query': df[(df['gross_label'] == 'Pre-caries') & ((df['chalky'] == 1) | (df['brown'] == 1))]},
    'Caries':     {'features': [('defect', 'structural cavity defect'), ('fill', 'adjacent filling material')], 'query': df[(df['gross_label'] == 'Caries') & (df['defect'] == 1)]},
    'Decolor':    {'features': [('stain', 'extrinsic surface stain'), ('wear', 'associated tooth wear')], 'query': df[(df['gross_label'] == 'Decolor') & (df['stain'] == 1)]}
}

# --- OCCLUSION MAPPING ENGINE ---
def generate_heatmap(image, feature_phrase, patch_size=32, stride=16):
    text_input = tokenizer([f"Clinical intraoral photograph showing {feature_phrase}."]).to(DEVICE)
    with torch.no_grad():
        base_img_emb = model.encode_image(preprocess(image).unsqueeze(0).to(DEVICE))
        text_emb = model.encode_text(text_input)
        base_img_emb /= base_img_emb.norm(dim=-1, keepdim=True)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        base_score = (base_img_emb @ text_emb.T).item()
    
    width, height = image.size
    heatmap_w = (width - patch_size) // stride + 1
    heatmap_h = (height - patch_size) // stride + 1
    heatmap = np.zeros((heatmap_h, heatmap_w))
    
    masked_images, coords = [], []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            masked = image.copy()
            draw = ImageDraw.Draw(masked)
            draw.rectangle([x, y, x + patch_size, y + patch_size], fill=(128, 128, 128))
            masked_images.append(preprocess(masked))
            coords.append((y // stride, x // stride))
            
    masked_batch = torch.stack(masked_images).to(DEVICE)
    with torch.no_grad():
        masked_embs = model.encode_image(masked_batch)
        masked_embs /= masked_embs.norm(dim=-1, keepdim=True)
        scores = (masked_embs @ text_emb.T).squeeze().cpu().numpy()
        
    for idx, (cy, cx) in enumerate(coords):
        heatmap[cy, cx] = max(0, base_score - scores[idx]) 
        
    heatmap = gaussian_filter(heatmap, sigma=1.5)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap

# --- MASTER GRID PLOTTING ---
fig, axes = plt.subplots(4, 3, figsize=(14, 16), facecolor='#F8F7F4')
plt.suptitle("Multimodal Spatial Grounding Across Clinical Phenotypes", fontsize=16, fontweight='bold', y=0.94)

for row_idx, cls_name in enumerate(CLASS_ORDER):
    print(f"Processing Row {row_idx + 1}/4: Class [{cls_name}]...")
    cfg = targets[cls_name]
    
    # Grab candidate row gracefully
    row_data = cfg['query'].iloc[0] if len(cfg['query']) > 0 else df[df['gross_label'] == cls_name].iloc[0]
    img_path = CROP_DIR / row_data['crop name']
    img = Image.open(img_path).convert('RGB')
    
    # Col 0: Original Image
    ax_img = axes[row_idx, 0]
    ax_img.imshow(img.resize((224, 224)))
    ax_img.set_ylabel(cls_name, fontsize=14, fontweight='bold', labelpad=15)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    if row_idx == 0:
        ax_img.set_title("Original Intraoral Crop", fontsize=12, fontweight='bold', pad=10)
        
    # Col 1 & 2: Feature Heatmaps
    for col_idx, (feat_name, feat_phrase) in enumerate(cfg['features'], start=1):
        heatmap = generate_heatmap(img, feat_phrase)
        ax_hm = axes[row_idx, col_idx]
        ax_hm.imshow(img.resize((224, 224)))
        ax_hm.imshow(heatmap, cmap='jet', alpha=0.5, extent=(0, 224, 224, 0))
        ax_hm.axis('off')
        
        if row_idx == 0:
            ax_hm.set_title(f"Target Feature Map", fontsize=12, fontweight='bold', pad=10)
        # Add a sub-label directly onto the patch to tell the viewer what feature they are looking at
        ax_hm.text(10, 210, f"Feature: {feat_name}", color='white', weight='bold', 
                   bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'), fontsize=10)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
output_name = 'c1_plot6_master_cam_grid.png'
plt.savefig(output_name, dpi=200, bbox_inches='tight', facecolor='#F8F7F4')
print(f"\nSUCCESS! High-resolution master grid saved cleanly as '{output_name}'.")