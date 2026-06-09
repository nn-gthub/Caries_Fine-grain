"""
PRE-FLIGHT CHECK
=================
Run before master_experiments.py to verify everything is ready.
Fixes common problems automatically where possible.

Usage:
  cd /data1/neena/finegrain_alpha_experiments
  python3 preflight_check.py
"""

import os, re
import pandas as pd
import numpy as np
from pathlib import Path

BASE      = Path('/Nasbackup/lab_nirmal/neena/datasets/Alphadent')
WORK_DIR  = Path('/data1/neena/finegrain_alpha_experiments')

FG_COLS     = ['chalky','brown','defect','fill','stain','wear']
CLASS_ORDER = ['Normal','Pre-caries','Caries','Decolor']

issues   = []
warnings = []

def chk(ok, msg, fix=None):
    status = '  ✓' if ok else '  ✗'
    print(f"{status}  {msg}")
    if not ok:
        issues.append(msg)
        if fix: print(f"     Fix: {fix}")

def warn(msg):
    print(f"  ⚠  {msg}")
    warnings.append(msg)

print("=" * 60)
print("PRE-FLIGHT CHECK — master_experiments.py")
print(f"Working dir: {WORK_DIR}")
print("=" * 60)

# ── CHECK 1: Required files ───────────────────────────────────
print("\n[1] Required files")

chk((WORK_DIR/'annotation_clean.csv').exists(),
    'annotation_clean.csv',
    f'cp {BASE}/finegrain_alpha/metadata/annotation_clean.csv {WORK_DIR}/')

chk((WORK_DIR/'gross_label_corrections.csv').exists(),
    'gross_label_corrections.csv',
    f'cp {BASE}/finegrain_alpha/metadata/gross_label_corrections.csv {WORK_DIR}/')

chk((WORK_DIR/'combined_all_crops_v2.csv').exists(),
    'combined_all_crops_v2.csv',
    'Run build_combined_csv.py first')

# ── CHECK 2: tooth_crops folder ───────────────────────────────
print("\n[2] tooth_crops/ image folder")

crop_dir = WORK_DIR / 'tooth_crops'
if crop_dir.exists():
    n = len(list(crop_dir.glob('*.jpg'))) + len(list(crop_dir.glob('*.png')))
    status = n >= 2000
    chk(status, f'tooth_crops/ — {n} images found',
        'Expected ~2893 images. Run: bash setup_crops.sh')
    if 0 < n < 2000:
        warn(f'Only {n} images in tooth_crops/ — expected ~2893')
else:
    chk(False, 'tooth_crops/ folder not found',
        'Run: bash setup_crops.sh')

# ── CHECK 3: annotation_clean.csv columns ────────────────────
print("\n[3] annotation_clean.csv columns")

ac_path = WORK_DIR / 'annotation_clean.csv'
if ac_path.exists():
    df = pd.read_csv(ac_path)
    for col in ['crop name','gross_label','split','patient_id',
                'clinical_note','recommended_action'] + FG_COLS:
        chk(col in df.columns, f"column '{col}'")

    if 'gross_label' in df.columns:
        review = (df['gross_label'] == 'REVIEW_NEEDED').sum()
        valid  = df['gross_label'].isin(CLASS_ORDER).sum()
        print(f"  ✓  Valid labels: {valid}/{len(df)}")
        if review > 0:
            warn(f"{review} REVIEW_NEEDED rows — will be auto-excluded from training")

    if 'split' in df.columns:
        print(f"  ✓  Splits: {df['split'].value_counts().to_dict()}")

# ── CHECK 4: combined_all_crops_v2.csv ───────────────────────
print("\n[4] combined_all_crops_v2.csv")

comb_path = WORK_DIR / 'combined_all_crops_v2.csv'
if comb_path.exists():
    dc = pd.read_csv(comb_path)
    print(f"  ✓  Rows: {len(dc)}")
    if 'is_fg_annotated' in dc.columns:
        fg_n = dc['is_fg_annotated'].sum()
        print(f"  ✓  FG-annotated: {fg_n}  Gross-only: {len(dc)-fg_n}")
    if 'split' in dc.columns:
        print(f"  ✓  Splits: {dc['split'].value_counts().to_dict()}")
    if 'pseudo_text' in dc.columns:
        avg = dc['pseudo_text'].str.len().mean()
        print(f"  ✓  pseudo_text avg chars: {avg:.0f}")
    else:
        warn("pseudo_text column missing — run build_combined_csv.py again")
    if 'gross_label' in dc.columns:
        review = (dc['gross_label']=='REVIEW_NEEDED').sum()
        if review > 0:
            warn(f"{review} REVIEW_NEEDED in combined — filtered automatically")

# ── CHECK 5: Crop name matching ───────────────────────────────
print("\n[5] Crop name → file matching")

if ac_path.exists() and crop_dir.exists():
    df  = pd.read_csv(ac_path)
    if 'crop name' in df.columns:
        found   = sum(1 for n in df['crop name'] if (crop_dir/str(n)).exists())
        pct     = 100*found/max(len(df),1)
        ok      = pct >= 95
        chk(ok, f'{found}/{len(df)} crops found ({pct:.1f}%)',
            'Check tooth_crops/ folder has all images')
        if not ok and found > 0:
            missing = [n for n in df['crop name'] if not (crop_dir/str(n)).exists()]
            print(f"     First 3 missing: {missing[:3]}")

# ── CHECK 6: GPU and environment ─────────────────────────────
print("\n[6] GPU and environment")

import torch
gpu = torch.cuda.is_available()
if gpu:
    print(f"  ✓  GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓  GPU memory: {mem:.1f} GB")
    if mem < 8:
        warn(f"Only {mem:.1f}GB GPU — LLaVA section will fail, reduce batch_size in Section 0")
else:
    warn("No GPU — embedding extraction will be slow on CPU")

oai = os.environ.get('OPENAI_API_KEY','')
if oai:
    print(f"  ✓  OPENAI_API_KEY set ({oai[:8]}...)")
else:
    warn("OPENAI_API_KEY not set — GPT-4o section skipped. Set with: export OPENAI_API_KEY=sk-...")

import shutil
free = shutil.disk_usage(WORK_DIR).free / 1e9
print(f"  ✓  Free disk: {free:.1f} GB")
if free < 20:
    warn(f"Only {free:.1f}GB free — model downloads need ~15GB")

# ── CHECK 7: Python packages ──────────────────────────────────
print("\n[7] Python packages")

pkgs = ['torch','transformers','open_clip','sklearn',
        'xgboost','shap','umap','statsmodels',
        'pandas','numpy','matplotlib','seaborn','PIL','tqdm']
for pkg in pkgs:
    try:
        import importlib
        importlib.import_module(pkg if pkg != 'PIL' else 'PIL')
        print(f"  ✓  {pkg}")
    except ImportError:
        chk(False, f"{pkg} not installed",
            f"pip install {pkg}")

# ── VERDICT ───────────────────────────────────────────────────
print("\n" + "="*60)
print("VERDICT")
print("="*60)

if issues:
    print(f"\n✗  NOT READY — {len(issues)} issue(s):")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print(f"\n✓  READY — run: python3 master_experiments.py")
    if warnings:
        print(f"\n⚠  {len(warnings)} warning(s) (non-blocking):")
        for w in warnings:
            print(f"   • {w}")

# ── SETUP SCRIPT ──────────────────────────────────────────────
setup_sh = WORK_DIR / 'setup_crops.sh'
if not setup_sh.exists():
    with open(setup_sh, 'w') as f:
        f.write(f"""#!/bin/bash
# Run once to create tooth_crops/ flat folder
mkdir -p {WORK_DIR}/tooth_crops

# FG-annotated crops
cp {BASE}/finegrain_alpha/annot_crops_combined_360/*.jpg {WORK_DIR}/tooth_crops/ 2>/dev/null
cp {BASE}/finegrain_alpha/annot_crops_combined_360/*.png {WORK_DIR}/tooth_crops/ 2>/dev/null

# All other crops from v2 (train + val)
for split in train val; do
  for cls in caries decolor normal pre-caries; do
    SRC="{BASE}/tooth_crops_final_v2/$split/$cls"
    if [ -d "$SRC/images" ]; then
      cp "$SRC/images"/*.jpg {WORK_DIR}/tooth_crops/ 2>/dev/null
    elif [ -d "$SRC" ]; then
      cp "$SRC"/*.jpg {WORK_DIR}/tooth_crops/ 2>/dev/null
    fi
  done
done

# Test crops
for cls in caries decolor normal pre-caries; do
  cp {BASE}/tooth_crops_test/cls/$cls/*.jpg {WORK_DIR}/tooth_crops/ 2>/dev/null
done

echo "Total crops: $(ls {WORK_DIR}/tooth_crops/ | wc -l)"
""")
    os.chmod(setup_sh, 0o755)
    print(f"\nCreated: {setup_sh}")
    print("  Run: bash setup_crops.sh  (to create tooth_crops/)")
