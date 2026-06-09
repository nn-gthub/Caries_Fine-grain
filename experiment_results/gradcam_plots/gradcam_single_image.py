import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
CLASS_COLORS = {
    'defect':  (255, 0, 0),     # Red
    'brown':   (255, 165, 0),   # Orange
    'chalky':  (0, 255, 255),   # Cyan
    'fill':    (0, 255, 0),     # Green
    'stain':   (255, 0, 255),   # Magenta
    'wear':    (255, 255, 0)    # Yellow
}
TARGET_CROPS  = ["p022_002_tooth000.jpg"]

def generate_single_image_gradcam():
    print(f"Loading model from {WEIGHTS}...")
    try:
        model = YOLO(str(WEIGHTS))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    df_split = pd.read_csv(SPLIT_CSV) if SPLIT_CSV.exists() else pd.DataFrame()

    for crop_name in TARGET_CROPS:
        img_path = CROP_DIR / crop_name
        if not img_path.exists():
            print(f"❌ Could not find image: {img_path}")
            continue

        row = df_split[df_split['crop name'] == crop_name]
        gt_gross_label = row.iloc[0]['gross_label'] if not row.empty else "Unknown"

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        results = model.predict(str(img_path), conf=0.10, retina_masks=True, verbose=False)[0]
        combined_heatmap = np.zeros((H, W, 3), dtype=np.float32)
        detected_in_image = []

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for i, cls_idx in enumerate(classes):
                if cls_idx < len(FG_CLASSES):
                    cls_name = FG_CLASSES[cls_idx]
                    mask = cv2.resize(masks_data[i], (W, H))
                    prob_map = mask * confs[i]
                    if prob_map.max() == 0:
                        continue
                    if cls_name not in detected_in_image:
                        detected_in_image.append(cls_name)

                    ksize = int(max(H, W) * 0.1) | 1
                    smoothed_map = cv2.GaussianBlur(prob_map, (ksize, ksize), 0)
                    if smoothed_map.max() > 0:
                        smoothed_map = smoothed_map / smoothed_map.max()

                    rgb_color = CLASS_COLORS[cls_name]
                    class_color_heatmap = np.zeros((H, W, 3), dtype=np.float32)
                    for c in range(3):
                        class_color_heatmap[:, :, c] = smoothed_map * rgb_color[c]
                    combined_heatmap = np.maximum(combined_heatmap, class_color_heatmap)

        alpha = 0.45
        combined_heatmap = np.uint8(combined_heatmap)
        gray_heatmap = cv2.cvtColor(combined_heatmap, cv2.COLOR_RGB2GRAY)
        fg_mask = gray_heatmap > 15
        
        final_overlay = img_rgb.copy()
        final_overlay[fg_mask] = cv2.addWeighted(
            combined_heatmap[fg_mask], alpha, img_rgb[fg_mask], 1.0 - alpha, 0
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original Image\n[GT Gross Label: {gt_gross_label}]", fontsize=12, fontweight='bold', color='blue')
        axes[0].axis("off")

        axes[1].imshow(final_overlay)
        axes[1].set_title("Unified Fine-Grain XAI Map", fontsize=12, fontweight='bold', color='darkred')
        axes[1].axis("off")

        legend_patches = []
        for cls_name in detected_in_image:
            patch_color = tuple(val / 255.0 for val in CLASS_COLORS[cls_name])
            legend_patches.append(mpatches.Patch(color=patch_color, label=f"Grad-CAM: {cls_name}"))
        if legend_patches:
            axes[1].legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1.0))

        plt.suptitle(f"Single-Image Unified XAI Evaluation: {crop_name}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        out_file = OUT_DIR / f"unified_gradcam_{crop_name}"
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Unified multi-class Grad-CAM saved cleanly to: {out_file}")

if __name__ == '__main__':
    generate_single_image_gradcam()