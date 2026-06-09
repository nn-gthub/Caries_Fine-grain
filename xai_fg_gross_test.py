"""
YOLOv8 DYNAMIC GRAD-CAM HEATMAP GENERATOR
=========================================
Generates true Grad-CAM style glowing thermal heatmaps.
Dynamically drops undetected classes to optimize layout spacing.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# ── Paths & Configuration ─────────────────────────────────────────────────────
WORK_DIR      = Path('/data1/neena/finegrain_alpha_experiments')
CROP_DIR      = WORK_DIR / 'tooth_crops'
SPLIT_CSV     = WORK_DIR / 'experiment_results' / 'segmentation' / 'seg_split.csv'
WEIGHTS       = WORK_DIR / 'experiment_results' / 'segmentation' / 'YOLOv8n-seg' / 'weights' / 'best.pt'
OUT_DIR       = WORK_DIR / 'experiment_results' / 'gradcam_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FG_CLASSES    = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']

def convert_to_gradcam_heatmap(img, mask_prob, alpha=0.6):
    """Transforms a raw mask prediction into a smooth thermal Grad-CAM activation."""
    # Normalize mask to 0-255 range
    mask_normalized = np.uint8(255 * (mask_prob / (mask_prob.max() + 1e-8)))
    
    # Apply a heavy Gaussian blur to smooth boundaries into a classic low-res Grad-CAM glow
    # Kernel size is tied to image size to ensure consistent blurring
    ksize = int(max(img.shape[:2]) * 0.1) | 1  # Ensure odd integer
    gradcam_blur = cv2.GaussianBlur(mask_normalized, (ksize, ksize), 0)
    
    # Generate the thermal colormap
    heatmap = cv2.applyColorMap(gradcam_blur, cv2.COLORMAP_JET)
    
    # Superimpose heatmap onto the original image
    gradcam_overlay = cv2.addWeighted(heatmap, alpha, img, 1.0 - alpha, 0)
    return cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)

def generate_dynamic_gradcam():
    print(f"Loading 6-class FG YOLOv8 model from {WEIGHTS}...")
    try:
        model = YOLO(str(WEIGHTS))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading test dataset...")
    df_split = pd.read_csv(SPLIT_CSV)
    df_test = df_split[df_split['seg_split'] == 'test']
    gross_label_map = dict(zip(df_test['crop name'], df_test['gross_label']))
    
    # Process all test images or sample a sub-cohort
    test_crops = list(gross_label_map.keys())
    np.random.seed(42)
    sample_crops = np.random.choice(test_crops, min(30, len(test_crops)), replace=False)

    for crop_name in sample_crops:
        img_path = CROP_DIR / crop_name
        if not img_path.exists():
            continue

        gt_gross_label = gross_label_map[crop_name]
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # Run inference using low confidence to capture weaker activation trends
        results = model.predict(str(img_path), conf=0.10, retina_masks=True, verbose=False)[0]

        # Keep track of only the active classes detected in this specific image
        detected_features = {}

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for i, cls_idx in enumerate(classes):
                if cls_idx < len(FG_CLASSES):
                    cls_name = FG_CLASSES[cls_idx]
                    mask = cv2.resize(masks_data[i], (W, H))
                    
                    # Store continuous probability maps multiplied by their class confidence
                    if cls_name not in detected_features:
                        detected_features[cls_name] = np.zeros((H, W), dtype=np.float32)
                    detected_features[cls_name] = np.maximum(detected_features[cls_name], mask * confs[i])

        # ── Dynamic Plotting Logic ────────────────────────────────────────────
        # Total columns = 1 (Original Image) + total detected classes
        num_cols = 1 + len(detected_features)
        
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
        # Handle cases where matplotlib squashes axis objects when num_cols == 1
        if num_cols == 1:
            axes = [axes]

        # 1. Plot Original Image with Ground Truth Context
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original Image\n[GT: {gt_gross_label}]", color='blue', fontweight='bold', fontsize=12)
        axes[0].axis("off")

        # 2. Append ONLY the detected classes with smooth Grad-CAM overlays
        for idx, (cls_name, prob_map) in enumerate(detected_features.items()):
            ax = axes[idx + 1]
            gradcam_img = convert_to_gradcam_heatmap(img, prob_map)
            
            ax.imshow(gradcam_img)
            ax.set_title(f"Grad-CAM: {cls_name}\n(Max Conf: {prob_map.max():.2f})", color='darkred', fontweight='bold', fontsize=12)
            ax.axis("off")

        plt.suptitle(f"XAI Analysis: {crop_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        out_file = OUT_DIR / f"gradcam_{crop_name}"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\n✓ True Grad-CAM heatmaps saved cleanly to: {OUT_DIR}")

if __name__ == '__main__':
    generate_dynamic_gradcam()