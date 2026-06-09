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

# --- NAS BACKUP PATHS ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")

print("Loading BiomedCLIP Model (Offline Mode Safe)...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(DEVICE).eval()

print("Loading Dataset to find a test image...")
df = pd.read_csv(MAIN_CSV)
# Let's find an image that we know has BOTH a 'defect' and a 'stain' so we can see the model look at different spots
test_df = df[(df['defect'] == 1) & (df['stain'] == 1)]

if len(test_df) == 0:
    # Fallback to any image if the combo isn't found
    target_row = df.iloc[0]
else:
    target_row = test_df.iloc[0]

img_path = CROP_DIR / target_row['crop name']
print(f"Target Image Selected: {target_row['crop name']}")

original_img = Image.open(img_path).convert('RGB')
img_tensor = preprocess(original_img).unsqueeze(0).to(DEVICE)

# --- OCCLUSION HEATMAP ENGINE ---
def generate_heatmap(image, feature_text, patch_size=32, stride=16):
    print(f"  -> Scanning image for: '{feature_text}'...")
    
    # 1. Get baseline similarity (Full Image vs Text)
    text_input = tokenizer([f"Clinical intraoral photograph showing {feature_text}."]).to(DEVICE)
    with torch.no_grad():
        base_img_emb = model.encode_image(preprocess(image).unsqueeze(0).to(DEVICE))
        text_emb = model.encode_text(text_input)
        
        base_img_emb /= base_img_emb.norm(dim=-1, keepdim=True)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        base_score = (base_img_emb @ text_emb.T).item()
    
    # 2. Slide a grey box across the image
    width, height = image.size
    heatmap_w = (width - patch_size) // stride + 1
    heatmap_h = (height - patch_size) // stride + 1
    heatmap = np.zeros((heatmap_h, heatmap_w))
    
    # Batch the masked images to process them incredibly fast on the GPU
    masked_images = []
    coords = []
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            masked = image.copy()
            draw = ImageDraw.Draw(masked)
            draw.rectangle([x, y, x + patch_size, y + patch_size], fill=(128, 128, 128))
            masked_images.append(preprocess(masked))
            coords.append((y // stride, x // stride))
            
    # Run all masked images through CLIP
    masked_batch = torch.stack(masked_images).to(DEVICE)
    with torch.no_grad():
        masked_embs = model.encode_image(masked_batch)
        masked_embs /= masked_embs.norm(dim=-1, keepdim=True)
        scores = (masked_embs @ text_emb.T).squeeze().cpu().numpy()
        
    # Calculate how much the score DROPPED when we covered that spot
    for idx, (cy, cx) in enumerate(coords):
        drop = base_score - scores[idx]
        # Only record positive drops (places where masking HURT the score)
        heatmap[cy, cx] = max(0, drop) 
        
    # 3. Smooth the grid into a nice organic heatmap
    heatmap = gaussian_filter(heatmap, sigma=1.5)
    
    # Normalize to 0-1 for plotting
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
        
    return heatmap

# Generate two separate heatmaps to prove the model tracks DIFFERENT things
heatmap_defect = generate_heatmap(original_img, "a structural tooth defect or cavity")
heatmap_stain = generate_heatmap(original_img, "a brown stain or discoloration")

# --- PLOTTING ---
print("Generating Final Overlay Plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#F8F7F4')

# 1. Original Image
axes[0].imshow(original_img.resize((224, 224)))
axes[0].set_title("Original Intraoral Crop", fontweight='bold')
axes[0].axis('off')

# 2. Defect Heatmap Overlay
axes[1].imshow(original_img.resize((224, 224)))
im1 = axes[1].imshow(heatmap_defect, cmap='jet', alpha=0.5, extent=(0, 224, 224, 0))
axes[1].set_title("Heatmap: Tracking 'Defect / Cavity'", fontweight='bold', color='#C1392B')
axes[1].axis('off')

# 3. Stain Heatmap Overlay
axes[2].imshow(original_img.resize((224, 224)))
im2 = axes[2].imshow(heatmap_stain, cmap='jet', alpha=0.5, extent=(0, 224, 224, 0))
axes[2].set_title("Heatmap: Tracking 'Brown Stain'", fontweight='bold', color='#E9A039')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('c1_plot6_feature_cam.png', dpi=150, bbox_inches='tight', facecolor='#F8F7F4')
print("\nSUCCESS! Saved pure visual proof as 'c1_plot6_feature_cam.png'.")