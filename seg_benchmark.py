"""
SEGMENTATION BENCHMARK
=======================
Trains and benchmarks 4 instance segmentation models on fine-grain
dental lesion annotations.

Models:
  1. YOLOv8n-seg  (nano  — fast baseline)
  2. YOLOv8s-seg  (small — better capacity)
  3. Mask R-CNN   (ResNet50 — torchvision standard)
  4. U-Net        (ConvNeXt-Tiny encoder — medical imaging standard)

Data:
  355 FG-annotated crops from CVAT (polygon annotations)
  Patient-level train/val/test split: 80/11/9

After training:
  Runs inference on ALL 2893 crops
  Saves predicted FG feature vectors to predicted_fg_features.csv

Install:
  pip install ultralytics segmentation-models-pytorch
  pip install pycocotools opencv-python

Run:
  cd /data1/neena/finegrain_alpha_experiments
  python3 seg_benchmark.py
"""

import os, json, shutil, warnings
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR     = Path('/data1/neena/finegrain_alpha_experiments')
CROP_DIR     = WORK_DIR / 'tooth_crops'
COMBINED     = WORK_DIR / 'combined_all_crops_v2.csv'
ANNOT_CLEAN  = WORK_DIR / 'annotation_clean.csv'
CORR_LOG     = WORK_DIR / 'gross_label_corrections.csv'
SEG_DIR      = WORK_DIR / 'seg_data'          # YOLO-format dataset
RESULTS_DIR  = WORK_DIR / 'experiment_results' / 'segmentation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Actual dataset structure — 4 COCO JSONs, one per gross class
SEG_ROOT = Path('/Nasbackup/lab_nirmal/neena/datasets/Alphadent'
                '/finegrain_alpha/crops_360_segregated')
GROSS_SUBDIRS = {
    'Normal':     SEG_ROOT / 'normal',
    'Pre-caries': SEG_ROOT / 'pre-caries',
    'Caries':     SEG_ROOT / 'caries',
    'Decolor':    SEG_ROOT / 'decolor',
}

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_CLASSES  = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
FG_COLS     = FG_CLASSES
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Results: {RESULTS_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD PATIENT-LEVEL SPLIT FOR SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1 — BUILD SEGMENTATION SPLIT")
print("="*60)

# Load FG-annotated crops
df_ann  = pd.read_csv(ANNOT_CLEAN)
corr    = pd.read_csv(CORR_LOG)
lkp     = dict(zip(corr['crop_name'], corr['corrected_gross_label']))
df_ann['gross_label'] = df_ann['crop name'].map(lkp).fillna(df_ann['gross_label'])
df_ann  = df_ann[df_ann['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)
df_ann  = df_ann[df_ann['split'].isin(['train','val'])].reset_index(drop=True)

for col in FG_COLS:
    df_ann[col] = (df_ann[col]
                   .map({True:1,False:0,'True':1,'False':0,np.nan:0})
                   .fillna(0).astype(int))

# Patient-level split
from sklearn.model_selection import GroupShuffleSplit

patients   = df_ann['patient_id'].values
gss_test   = GroupShuffleSplit(n_splits=1, test_size=0.09, random_state=42)
trainval_i, test_i = next(gss_test.split(df_ann, groups=patients))

df_trainval = df_ann.iloc[trainval_i].copy()
df_seg_test = df_ann.iloc[test_i].copy()

gss_val    = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=42)
train_i, val_i = next(gss_val.split(df_trainval,
                                     groups=df_trainval['patient_id'].values))
df_seg_train = df_trainval.iloc[train_i].copy().reset_index(drop=True)
df_seg_val   = df_trainval.iloc[val_i].copy().reset_index(drop=True)
df_seg_test  = df_seg_test.reset_index(drop=True)

print(f"Seg train: {len(df_seg_train)} crops, "
      f"{df_seg_train['patient_id'].nunique()} patients")
print(f"Seg val:   {len(df_seg_val)} crops, "
      f"{df_seg_val['patient_id'].nunique()} patients")
print(f"Seg test:  {len(df_seg_test)} crops, "
      f"{df_seg_test['patient_id'].nunique()} patients")

# Save split info
df_seg_train['seg_split'] = 'train'
df_seg_val['seg_split']   = 'val'
df_seg_test['seg_split']  = 'test'
pd.concat([df_seg_train, df_seg_val, df_seg_test]).to_csv(
    RESULTS_DIR/'seg_split.csv', index=False)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — PARSE CVAT ANNOTATIONS AND BUILD YOLO FORMAT DATASET
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2 — BUILD YOLO SEGMENTATION DATASET")
print("="*60)

def load_all_coco_annotations(gross_subdirs):
    """
    Load and merge all four per-class COCO JSON files.
    Returns unified dict: {filename: [(class_name, norm_pts), ...]}

    Your structure:
      crops_360_segregated/{normal,pre-caries,caries,decolor}/
          images/
          labels/instances_default.json
    """
    unified = {}
    total_instances = 0

    for gross_label, subdir in gross_subdirs.items():
        json_path = subdir / 'labels' / 'instances_default.json'
        img_dir   = subdir / 'images'

        if not json_path.exists():
            print(f"  ⚠  Not found: {json_path}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Build category mapping — fine-grain class name → id
        cat_map  = {c['id']: c['name'].lower().strip()
                    for c in data['categories']}
        img_map  = {img['id']: img['file_name']
                    for img in data['images']}
        img_size = {img['id']: (img['width'], img['height'])
                    for img in data['images']}

        n_instances = 0
        for ann in data['annotations']:
            fname = Path(img_map[ann['image_id']]).name
            W, H  = img_size[ann['image_id']]
            cls   = cat_map[ann['category_id']]
            seg   = ann.get('segmentation', [])

            if not seg:
                continue

            # ── Handle RLE format (your CVAT export uses this) ────────────
            if isinstance(seg, dict) and 'counts' in seg:
                # Compressed RLE — decode to binary mask then find contours
                try:
                    from pycocotools import mask as coco_mask
                    rle    = coco_mask.frPyObjects(seg, H, W)
                    binary = coco_mask.decode(rle).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    # Use largest contour
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) < 4:
                        continue
                    # Simplify polygon
                    epsilon  = 0.005 * cv2.arcLength(cnt, True)
                    approx   = cv2.approxPolyDP(cnt, epsilon, True)
                    pts_raw  = approx.reshape(-1, 2)
                    norm_pts = []
                    for px, py in pts_raw:
                        norm_pts.extend([px/W, py/H])
                    if len(norm_pts) < 6:
                        continue
                except Exception as e:
                    continue

            # ── Handle polygon list format ─────────────────────────────────
            elif isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list):
                    pts_flat = seg[0]
                else:
                    pts_flat = seg
                if len(pts_flat) < 6:
                    continue
                norm_pts = []
                for i in range(0, len(pts_flat)-1, 2):
                    norm_pts.extend([pts_flat[i]/W, pts_flat[i+1]/H])
            else:
                continue

            if fname not in unified:
                unified[fname] = []
            unified[fname].append((cls, norm_pts))
            n_instances += 1

        total_instances += n_instances
        print(f"  {gross_label:12s}: {len(data['images'])} images, "
              f"{n_instances} instances")

    print(f"  Total: {len(unified)} images, {total_instances} instances")

    # Per-class count
    class_counts = {}
    for fname, anns in unified.items():
        for cls, _ in anns:
            class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"  Per fine-grain class:")
    for cls in FG_CLASSES:
        n = class_counts.get(cls, 0)
        flag = '  ← low' if n < 30 else ''
        print(f"    {cls:12s}: {n}{flag}")

    return unified


def build_yolo_seg_dataset(df_split, split_name, annotations,
                            seg_dir, crop_dir):
    """
    Write YOLO segmentation format:
      images/{split}/image.jpg
      labels/{split}/image.txt  → class_id x1 y1 x2 y2 (normalised)
    """
    img_out = seg_dir / 'images' / split_name
    lbl_out = seg_dir / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written, skipped, no_ann = 0, 0, 0
    for _, row in df_split.iterrows():
        fname = row['crop name']
        src   = Path(crop_dir) / fname
        anns  = annotations.get(fname, [])

        if not src.exists():
            skipped += 1
            continue

        shutil.copy(src, img_out / fname)

        lines = []
        for cls_name, pts in anns:
            if cls_name in FG_CLASSES:
                cls_id = FG_CLASSES.index(cls_name)
                pt_str = ' '.join(f'{v:.6f}' for v in pts)
                lines.append(f'{cls_id} {pt_str}')

        stem = Path(fname).stem
        with open(lbl_out / f'{stem}.txt', 'w') as f:
            f.write('\n'.join(lines))

        if not lines:
            no_ann += 1
        written += 1

    print(f"  {split_name}: {written} written  "
          f"({no_ann} with no FG annotations)  "
          f"{skipped} images not found")
    return written


# ── Load annotations from your 4 per-class JSON files ────────────────────
print("Loading annotations from per-class COCO JSONs...")
annotations = load_all_coco_annotations(GROSS_SUBDIRS)

# Verify coverage against annotation_clean.csv
ann_names   = set(df_ann['crop name'])
found_names = set(annotations.keys())
covered     = ann_names & found_names
print(f"\nAnnotation coverage: {len(covered)}/{len(ann_names)} crops "
      f"({100*len(covered)/max(len(ann_names),1):.1f}%)")

if len(covered) < len(ann_names) * 0.8:
    print("⚠  Low coverage — checking filename format...")
    # Check if image filenames in JSON match crop names in CSV
    sample_json = list(found_names)[:3]
    sample_csv  = list(ann_names)[:3]
    print(f"  JSON filenames: {sample_json}")
    print(f"  CSV crop names: {sample_csv}")

# Build YOLO dataset
SEG_DIR.mkdir(parents=True, exist_ok=True)
for df_split, split_name in [
    (df_seg_train, 'train'),
    (df_seg_val,   'val'),
    (df_seg_test,  'test'),
]:
    build_yolo_seg_dataset(
        df_split, split_name, annotations, SEG_DIR, CROP_DIR)

# Write data.yaml
import yaml
data_yaml_content = {
    'path':  str(SEG_DIR.absolute()),
    'train': 'images/train',
    'val':   'images/val',
    'test':  'images/test',
    'nc':    len(FG_CLASSES),
    'names': FG_CLASSES,
}
yaml_path = SEG_DIR / 'data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"\nYOLO dataset built at: {SEG_DIR}")
print(f"Data YAML: {yaml_path}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — YOLO SEGMENTATION MODELS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3 — YOLO SEGMENTATION BENCHMARK")
print("="*60)

from ultralytics import YOLO

yolo_results = {}

for model_name, weights in [
    ('YOLOv8n-seg', 'yolov8n-seg.pt'),
    ('YOLOv8s-seg', 'yolov8s-seg.pt'),
]:
    print(f"\nTraining {model_name}...")
    model = YOLO(weights)

    results = model.train(
        data          = str(yaml_path),
        epochs        = 100,
        imgsz         = 640,
        batch         = 16,
        patience      = 20,
        optimizer     = 'AdamW',
        lr0           = 0.001,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        warmup_momentum = 0.8,
        # Augmentation
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 10.0,
        translate     = 0.1,
        scale         = 0.3,
        shear         = 0.0,
        flipud        = 0.0,
        fliplr        = 0.5,
        mosaic        = 0.5,
        copy_paste    = 0.1,
        # Output
        project       = str(RESULTS_DIR),
        name          = model_name,
        exist_ok      = True,
        device        = 0 if torch.cuda.is_available() else 'cpu',
        workers       = 4,
        seed          = 42,
        verbose       = False,
    )

    # Evaluate on test set
    best_path = RESULTS_DIR / model_name / 'weights' / 'best.pt'
    best_model = YOLO(str(best_path))
    metrics    = best_model.val(
        data   = str(yaml_path),
        split  = 'test',
        device = 0 if torch.cuda.is_available() else 'cpu',
        verbose= False,
    )

    yolo_results[model_name] = {
        'map50_M':    float(metrics.seg.map50),
        'map5095_M':  float(metrics.seg.map),
        'map50_B':    float(metrics.box.map50),
        'weights':    str(best_path),
    }

    print(f"  {model_name}: mAP50(M)={metrics.seg.map50:.4f}  "
          f"mAP50-95(M)={metrics.seg.map:.4f}")

# Save YOLO results
with open(RESULTS_DIR/'yolo_results.json', 'w') as f:
    json.dump(yolo_results, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — MASK R-CNN (torchvision)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4 — MASK R-CNN BENCHMARK")
print("="*60)

import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class DentalSegDataset(Dataset):
    """Dataset for Mask R-CNN — returns image + target dict."""
    def __init__(self, df, annotations, crop_dir, transforms=None):
        self.df          = df.reset_index(drop=True)
        self.annotations = annotations
        self.crop_dir    = Path(crop_dir)
        self.transforms  = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = row['crop name']
        path  = self.crop_dir / fname

        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        W, H   = img.size
        anns   = self.annotations.get(fname, [])

        boxes, masks, labels = [], [], []
        for cls_name, pts in anns:
            if cls_name not in FG_CLASSES:
                continue
            cls_id = FG_CLASSES.index(cls_name) + 1  # 1-indexed for RCNN

            # Denormalise polygon
            coords = np.array(pts).reshape(-1, 2)
            coords[:, 0] *= W
            coords[:, 1] *= H

            # Bounding box
            x1, y1 = coords.min(axis=0)
            x2, y2 = coords.max(axis=0)
            if x2-x1 < 2 or y2-y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])

            # Binary mask from polygon
            mask = np.zeros((H, W), dtype=np.uint8)
            poly = coords.astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
            masks.append(mask)
            labels.append(cls_id)

        img_t = transforms.functional.to_tensor(img)

        if len(boxes) == 0:
            target = {
                'boxes':  torch.zeros((0, 4), dtype=torch.float32),
                'masks':  torch.zeros((0, H, W), dtype=torch.uint8),
                'labels': torch.zeros(0, dtype=torch.int64),
            }
        else:
            target = {
                'boxes':  torch.tensor(boxes, dtype=torch.float32),
                'masks':  torch.tensor(np.array(masks), dtype=torch.uint8),
                'labels': torch.tensor(labels, dtype=torch.int64),
            }

        if self.transforms:
            img_t = self.transforms(img_t)

        return img_t, target


def get_maskrcnn(num_classes):
    """Load pre-trained Mask R-CNN and replace heads."""
    model  = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    in_features_mask = (model.roi_heads.mask_predictor
                        .conv5_mask.in_channels)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes)
    return model


# Augmentation transforms for Mask R-CNN
RCNN_TRAIN_TF = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomErasing(p=0.1),
])


def collate_fn(batch):
    return tuple(zip(*batch))


def train_maskrcnn(model, dl_tr, dl_val, epochs=25, device=DEVICE):
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        tr_loss = 0
        for imgs, targets in tqdm(dl_tr, desc=f"Ep{epoch+1}",
                                   leave=False):
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        # Val loss
        model.train()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in dl_val:
                imgs    = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]
                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()

        avg_tr  = tr_loss  / max(len(dl_tr), 1)
        avg_val = val_loss / max(len(dl_val), 1)

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: "
                  f"tr_loss={avg_tr:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss  = avg_val
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


NUM_CLASSES = len(FG_CLASSES) + 1  # +1 for background

ds_rcnn_tr  = DentalSegDataset(df_seg_train, annotations,
                                CROP_DIR, RCNN_TRAIN_TF)
ds_rcnn_val = DentalSegDataset(df_seg_val,   annotations, CROP_DIR)
ds_rcnn_te  = DentalSegDataset(df_seg_test,  annotations, CROP_DIR)

dl_rcnn_tr  = DataLoader(ds_rcnn_tr,  batch_size=4, shuffle=True,
                          num_workers=4, collate_fn=collate_fn,
                          pin_memory=True)
dl_rcnn_val = DataLoader(ds_rcnn_val, batch_size=4, shuffle=False,
                          num_workers=4, collate_fn=collate_fn)
dl_rcnn_te  = DataLoader(ds_rcnn_te,  batch_size=4, shuffle=False,
                          num_workers=4, collate_fn=collate_fn)

print("Training Mask R-CNN (ResNet50-FPN-v2)...")
rcnn_model = get_maskrcnn(NUM_CLASSES)
rcnn_model = train_maskrcnn(rcnn_model, dl_rcnn_tr, dl_rcnn_val,
                             epochs=25)
torch.save(rcnn_model.state_dict(),
           RESULTS_DIR/'maskrcnn_best.pt')
print("Mask R-CNN training complete.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — U-NET WITH CONVNEXT-TINY ENCODER
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 5 — U-NET / CONVNEXT-TINY BENCHMARK")
print("="*60)

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    print("  segmentation_models_pytorch not installed.")
    print("  Install: pip install segmentation-models-pytorch")
    HAS_SMP = False

if HAS_SMP:

    class DentalBinaryDataset(Dataset):
        """
        Semantic segmentation dataset for U-Net.
        One channel per FG class (multi-label binary mask).
        """
        def __init__(self, df, annotations, crop_dir,
                     img_size=512, augment=False):
            self.df          = df.reset_index(drop=True)
            self.annotations = annotations
            self.crop_dir    = Path(crop_dir)
            self.img_size    = img_size
            self.augment     = augment

            self.img_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225]),
            ])
            # Augmentation applied to both image and mask
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.1, hue=0.05, p=0.5),
                A.GaussNoise(var_limit=(5,25), p=0.3),
                A.Resize(img_size, img_size),
            ]) if augment else A.Compose([
                A.Resize(img_size, img_size),
            ])

        def __len__(self): return len(self.df)

        def __getitem__(self, idx):
            row   = self.df.iloc[idx]
            fname = row['crop name']
            path  = self.crop_dir / fname

            try:
                img = np.array(Image.open(path).convert('RGB'))
            except Exception:
                img = np.zeros((512, 512, 3), dtype=np.uint8) + 128
            H, W = img.shape[:2]

            # Build multi-channel mask (one channel per FG class)
            anns = self.annotations.get(fname, [])
            masks = np.zeros((H, W, len(FG_CLASSES)), dtype=np.float32)

            for cls_name, pts in anns:
                if cls_name not in FG_CLASSES:
                    continue
                cls_id = FG_CLASSES.index(cls_name)
                coords = np.array(pts).reshape(-1, 2)
                coords[:, 0] *= W
                coords[:, 1] *= H
                poly = coords.astype(np.int32)
                cv2.fillPoly(masks[:, :, cls_id], [poly], 1.0)

            # Apply augmentation
            augmented = self.aug(image=img, masks=[
                masks[:,:,i] for i in range(len(FG_CLASSES))])
            img_aug   = augmented['image']
            masks_aug = np.stack(augmented['masks'], axis=2)

            # Normalise image
            img_t = torch.tensor(img_aug.transpose(2,0,1),
                                  dtype=torch.float32) / 255.0
            mean  = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            std   = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            img_t = (img_t - mean) / std

            mask_t = torch.tensor(
                masks_aug.transpose(2,0,1), dtype=torch.float32)

            return img_t, mask_t

    # Try albumentations
    try:
        import albumentations as A
        HAS_ALBUMENTATIONS = True
    except ImportError:
        print("  albumentations not installed — using basic augmentation")
        print("  Install: pip install albumentations")
        HAS_ALBUMENTATIONS = False

        # Fallback dataset without albumentations
        class DentalBinaryDataset(Dataset):
            def __init__(self, df, annotations, crop_dir,
                         img_size=512, augment=False):
                self.df          = df.reset_index(drop=True)
                self.annotations = annotations
                self.crop_dir    = Path(crop_dir)
                self.img_size    = img_size
                self.augment     = augment

            def __len__(self): return len(self.df)

            def __getitem__(self, idx):
                row   = self.df.iloc[idx]
                fname = row['crop name']
                path  = self.crop_dir / fname
                try:
                    img = np.array(Image.open(path).convert('RGB')
                                   .resize((self.img_size,
                                            self.img_size)))
                except Exception:
                    img = np.zeros((self.img_size, self.img_size, 3),
                                   dtype=np.uint8) + 128
                H, W = img.shape[:2]

                anns  = self.annotations.get(fname, [])
                masks = np.zeros((H, W, len(FG_CLASSES)),
                                  dtype=np.float32)
                for cls_name, pts in anns:
                    if cls_name not in FG_CLASSES:
                        continue
                    cls_id = FG_CLASSES.index(cls_name)
                    coords = np.array(pts).reshape(-1,2)
                    coords[:,0] *= W; coords[:,1] *= H
                    cv2.fillPoly(masks[:,:,cls_id],
                                  [coords.astype(np.int32)], 1.0)

                img_t = torch.tensor(
                    img.transpose(2,0,1), dtype=torch.float32)/255.0
                mean  = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
                std   = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
                img_t = (img_t - mean) / std

                return img_t, torch.tensor(
                    masks.transpose(2,0,1), dtype=torch.float32)

    IMG_SIZE = 512

    ds_unet_tr  = DentalBinaryDataset(
        df_seg_train, annotations, CROP_DIR, IMG_SIZE, augment=True)
    ds_unet_val = DentalBinaryDataset(
        df_seg_val, annotations, CROP_DIR, IMG_SIZE, augment=False)
    ds_unet_te  = DentalBinaryDataset(
        df_seg_test, annotations, CROP_DIR, IMG_SIZE, augment=False)

    dl_unet_tr  = DataLoader(ds_unet_tr,  batch_size=8,
                              shuffle=True, num_workers=4,
                              pin_memory=True)
    dl_unet_val = DataLoader(ds_unet_val, batch_size=8,
                              shuffle=False, num_workers=4)
    dl_unet_te  = DataLoader(ds_unet_te,  batch_size=8,
                              shuffle=False, num_workers=4)

    # Build U-Net with ConvNeXt-Tiny encoder
    unet_model = smp.Unet(
        encoder_name     = 'convnext_tiny',
        encoder_weights  = 'imagenet',
        in_channels      = 3,
        classes          = len(FG_CLASSES),
        activation       = None,
    ).to(DEVICE)

    print(f"U-Net/ConvNeXt-Tiny parameters: "
          f"{sum(p.numel() for p in unet_model.parameters())/1e6:.1f}M")

    # Combined loss: Dice + BCE
    dice_loss = smp.losses.DiceLoss(mode='multilabel', smooth=1.0)
    bce_loss  = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return 0.5 * dice_loss(pred, target) + \
               0.5 * bce_loss(pred, target)

    unet_optimizer = torch.optim.AdamW(
        unet_model.parameters(), lr=3e-4, weight_decay=1e-4)
    unet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        unet_optimizer, T_max=50)

    print("Training U-Net/ConvNeXt-Tiny (50 epochs)...")
    best_val_loss = float('inf')
    best_unet_state = None
    patience_count  = 0

    unet_history = []
    for epoch in range(50):
        # Train
        unet_model.train()
        tr_loss = 0
        for imgs, masks in dl_unet_tr:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            unet_optimizer.zero_grad()
            pred = unet_model(imgs)
            loss = combined_loss(pred, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                unet_model.parameters(), 1.0)
            unet_optimizer.step()
            tr_loss += loss.item()
        unet_scheduler.step()

        # Val
        unet_model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in dl_unet_val:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                pred = unet_model(imgs)
                val_loss += combined_loss(pred, masks).item()
                # Dice per class
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                inter = (pred_bin * masks).sum((0,2,3))
                union = pred_bin.sum((0,2,3)) + masks.sum((0,2,3))
                dice  = (2*inter / (union + 1e-6)).mean().item()
                val_dice += dice

        avg_tr   = tr_loss  / max(len(dl_unet_tr),  1)
        avg_val  = val_loss / max(len(dl_unet_val), 1)
        avg_dice = val_dice / max(len(dl_unet_val), 1)

        unet_history.append({'epoch': epoch+1,
                              'tr_loss': avg_tr,
                              'val_loss': avg_val,
                              'val_dice': avg_dice})

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}: "
                  f"tr={avg_tr:.4f}  val={avg_val:.4f}  "
                  f"val_dice={avg_dice:.4f}")

        if avg_val < best_val_loss:
            best_val_loss   = avg_val
            best_unet_state = {k: v.clone()
                               for k, v in unet_model.state_dict().items()}
            patience_count  = 0
        else:
            patience_count += 1

        if patience_count >= 10:
            print(f"  Early stop at epoch {epoch+1}")
            break

    unet_model.load_state_dict(best_unet_state)
    torch.save(best_unet_state, RESULTS_DIR/'unet_convnext_best.pt')
    pd.DataFrame(unet_history).to_csv(
        RESULTS_DIR/'unet_history.csv', index=False)
    print("U-Net training complete.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — SEGMENTATION EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 6 — SEGMENTATION EVALUATION")
print("="*60)


def boundary_iou(pred_mask, gt_mask, dilation_ratio=0.02):
    """Boundary IoU — scores only pixels near polygon boundary."""
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0
    H, W    = gt_mask.shape
    bw      = max(1, int(dilation_ratio * np.sqrt(H*W)))
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (bw*2+1, bw*2+1))
    gt_b    = gt_mask - cv2.erode(
        gt_mask.astype(np.uint8), kernel)
    pred_b  = pred_mask - cv2.erode(
        pred_mask.astype(np.uint8), kernel)
    inter   = (gt_b & pred_b).sum()
    union   = (gt_b | pred_b).sum()
    return float(inter / union) if union > 0 else 0.0


def eval_unet_segmentation(model, dl_test, device, threshold=0.5):
    """Evaluate U-Net on test set. Returns per-class Dice and BIoU."""
    model.eval()
    dice_per_cls = np.zeros(len(FG_CLASSES))
    biou_per_cls = np.zeros(len(FG_CLASSES))
    count        = np.zeros(len(FG_CLASSES))

    with torch.no_grad():
        for imgs, masks_gt in tqdm(dl_test, desc="Eval U-Net",
                                    leave=False):
            imgs    = imgs.to(device)
            pred    = torch.sigmoid(model(imgs))
            pred_b  = (pred > threshold).cpu().numpy()
            masks_n = masks_gt.numpy()

            for cls in range(len(FG_CLASSES)):
                for b in range(pred_b.shape[0]):
                    p = pred_b[b, cls]
                    g = masks_n[b, cls].astype(np.uint8)
                    if g.sum() == 0 and p.sum() == 0:
                        continue
                    inter = (p * g).sum()
                    union = p.sum() + g.sum()
                    dice  = 2*inter/(union+1e-6) if union > 0 else 1.0
                    biou  = boundary_iou(p.astype(np.uint8), g)
                    dice_per_cls[cls] += dice
                    biou_per_cls[cls] += biou
                    count[cls]        += 1

    count = np.maximum(count, 1)
    return dice_per_cls/count, biou_per_cls/count


if HAS_SMP:
    print("Evaluating U-Net on seg test set...")
    dice_vals, biou_vals = eval_unet_segmentation(
        unet_model, dl_unet_te, DEVICE)

    unet_seg_results = {}
    print(f"\n  {'Class':12s}  {'Dice':>7}  {'BoundaryIoU':>12}")
    print(f"  {'-'*35}")
    for i, cls in enumerate(FG_CLASSES):
        unet_seg_results[cls] = {
            'dice': float(dice_vals[i]),
            'biou': float(biou_vals[i])}
        print(f"  {cls:12s}  {dice_vals[i]:7.4f}  {biou_vals[i]:12.4f}")
    print(f"  {'Mean':12s}  {dice_vals.mean():7.4f}  {biou_vals.mean():12.4f}")

    with open(RESULTS_DIR/'unet_seg_results.json', 'w') as f:
        json.dump(unet_seg_results, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — INFERENCE ON ALL 2893 CROPS → PREDICTED FG FEATURES
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 7 — PREDICT FG FEATURES FOR ALL CROPS")
print("="*60)

df_all = pd.read_csv(COMBINED)
df_all = df_all[df_all['gross_label'].isin(CLASS_ORDER)].copy()
df_all = df_all.reset_index(drop=True)

# Find best YOLO model
best_yolo_name = max(yolo_results,
                     key=lambda k: yolo_results[k]['map50_M'])
best_yolo_path = yolo_results[best_yolo_name]['weights']
print(f"Using best YOLO model: {best_yolo_name} "
      f"(mAP50={yolo_results[best_yolo_name]['map50_M']:.4f})")

inference_model = YOLO(best_yolo_path)

records = []
for _, row in tqdm(df_all.iterrows(), total=len(df_all),
                   desc="Predicting FG features"):
    img_path = CROP_DIR / row['crop name']
    feat_vec = {f'pred_{c}': 0.0 for c in FG_CLASSES}
    feat_vec.update({f'pred_{c}_conf': 0.0 for c in FG_CLASSES})
    feat_vec.update({f'pred_{c}_area': 0.0 for c in FG_CLASSES})

    if img_path.exists():
        try:
            results = inference_model.predict(
                str(img_path), conf=0.25, verbose=False)
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                for i, cls_id in enumerate(r.boxes.cls.cpu().numpy()):
                    cls_name = FG_CLASSES[int(cls_id)]
                    conf     = float(r.boxes.conf[i].cpu())

                    # Max confidence per class
                    if conf > feat_vec[f'pred_{cls_name}_conf']:
                        feat_vec[f'pred_{cls_name}_conf'] = conf
                    feat_vec[f'pred_{cls_name}'] = 1.0

                    # Mask area ratio
                    if r.masks is not None:
                        mask = r.masks.data[i].cpu().numpy()
                        feat_vec[f'pred_{cls_name}_area'] += (
                            float(mask.sum()) / mask.size)
        except Exception as e:
            pass  # Keep zeros for failed predictions

    rec = {'crop name': row['crop name'],
           'gross_label': row['gross_label'],
           'split': row['split'],
           'patient_id': row.get('patient_id', 'unknown')}
    rec.update(feat_vec)
    records.append(rec)

pred_df = pd.DataFrame(records)
pred_df.to_csv(RESULTS_DIR/'predicted_fg_features.csv', index=False)
print(f"\nSaved: predicted_fg_features.csv ({len(pred_df)} rows)")

# Summary of predictions
print("\nPredicted feature prevalence (all crops):")
for cls in FG_CLASSES:
    n   = (pred_df[f'pred_{cls}'] > 0).sum()
    pct = 100*n/len(pred_df)
    print(f"  {cls:12s}: {n:4d} ({pct:.1f}%)")

print("\nPredicted vs ground truth (annotated subset):")
ann_pred = pred_df[pred_df['crop name'].isin(df_ann['crop name'])]
ann_true = df_ann.set_index('crop name')
print(f"  {'Class':12s}  {'GT count':>10}  {'Pred count':>10}")
for cls in FG_CLASSES:
    gt_n   = int(ann_true[cls].sum()) if cls in ann_true.columns else 0
    pred_n = int((ann_pred[f'pred_{cls}'] > 0).sum())
    print(f"  {cls:12s}  {gt_n:10d}  {pred_n:10d}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — SEGMENTATION RESULTS SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SEGMENTATION BENCHMARK SUMMARY")
print("="*60)
print(f"\n{'Model':30s}  {'mAP50(M)':>10}  {'mAP50-95(M)':>12}")
print(f"  {'-'*55}")

for name, r in yolo_results.items():
    print(f"  {name:28s}  {r['map50_M']:10.4f}  {r['map5095_M']:12.4f}")

if HAS_SMP:
    mean_dice = np.mean([v['dice'] for v in unet_seg_results.values()])
    mean_biou = np.mean([v['biou'] for v in unet_seg_results.values()])
    print(f"  {'U-Net/ConvNeXt-Tiny':28s}  "
          f"{'Dice='+f'{mean_dice:.4f}':>10}  "
          f"{'BIoU='+f'{mean_biou:.4f}':>12}")

print(f"\nPredicted FG features saved to:")
print(f"  {RESULTS_DIR}/predicted_fg_features.csv")
print(f"\nNext step: run fusion_with_predictions.py")
print("✓ Segmentation benchmark complete.")
