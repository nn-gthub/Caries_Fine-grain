# Install
# pip install qwen-vl-utils --quiet

# python3 << 'QWEN_ZEROSHOT'
import json, time, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
warnings.filterwarnings('ignore')

WORK_DIR    = Path('/data1/neena/finegrain_alpha_experiments')
CROP_DIR    = WORK_DIR / 'tooth_crops'
ANNOT_CLEAN = WORK_DIR / 'annotation_clean.csv'
CORR_LOG    = WORK_DIR / 'gross_label_corrections.csv'
RESULTS_DIR = WORK_DIR / 'experiment_results' / 'vlm_comparison'
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_CLASSES  = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']

# Load annotated crops (same 355 used in CV)
df = pd.read_csv(ANNOT_CLEAN)
corr = pd.read_csv(CORR_LOG)
lkp  = dict(zip(corr['crop_name'], corr['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(lkp).fillna(df['gross_label'])
df = df[df['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)
df = df[df['split'].isin(['train','val'])].reset_index(drop=True)

print(f"Running Qwen2-VL on {len(df)} annotated crops")

# Load model
print("Loading Qwen2-VL-7B...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-7B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
print("Model loaded")

PROMPT = """You are a dental specialist examining an intraoral tooth photograph.
Classify this tooth into exactly one of these four categories:

- Normal: sound tooth, no pathology, normal color and gloss
- Pre-caries: early demineralisation (white spot or brown fissure staining), NO cavitation
- Caries: structural tooth loss visible (cavitation, cavity, exposed dentin)
- Decolor: discolouration only (stain, fluorosis, wear) — no caries

Also identify which of these features you can see:
chalky (white opacity), brown (brown discolouration), defect (cavity/structural loss),
fill (restoration/filling), stain (surface stain), wear (mechanical wear)

Respond in this exact format only:
LABEL: [Normal or Pre-caries or Caries or Decolor]
FEATURES: [comma-separated list, or none]
REASON: [one sentence]"""

import re

def parse_output(text):
    label_m = re.search(
        r'LABEL:\s*(Normal|Pre-caries|Caries|Decolor)',
        text, re.IGNORECASE)
    feat_m  = re.search(r'FEATURES:\s*(.+)', text)
    reason_m= re.search(r'REASON:\s*(.+)', text)
    return {
        'pred_label':    label_m.group(1).strip() if label_m else 'Unknown',
        'pred_features': feat_m.group(1).strip()  if feat_m  else '',
        'pred_reason':   reason_m.group(1).strip() if reason_m else '',
    }

records = []
failed  = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Qwen2-VL"):
    img_path = CROP_DIR / row['crop name']
    if not img_path.exists():
        failed += 1
        continue

    try:
        img = Image.open(img_path).convert('RGB')

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': img},
                {'type': 'text',  'text':  PROMPT},
            ]
        }]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input], images=[img],
            return_tensors='pt', padding=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens  = 80,
                do_sample       = False,
                temperature     = 1.0,
                repetition_penalty = 1.1,
            )

        input_len = inputs['input_ids'].shape[1]
        new_ids   = output_ids[0][input_len:]
        text_out  = processor.tokenizer.decode(
            new_ids, skip_special_tokens=True).strip()

        pred = parse_output(text_out)

    except Exception as e:
        pred = {'pred_label':'Unknown',
                'pred_features':'', 'pred_reason': str(e)[:50]}
        failed += 1

    records.append({
        'crop name':     row['crop name'],
        'true_label':    row['gross_label'],
        'true_features': ','.join([c for c in FG_CLASSES
                                   if row.get(c,0)==1]),
        **pred
    })

df_results = pd.DataFrame(records)
df_results.to_csv(RESULTS_DIR/'qwen2vl_results.csv', index=False)

# Evaluate
valid = df_results[df_results['pred_label'].isin(CLASS_ORDER)]
print(f"\nValid predictions: {len(valid)}/{len(df_results)}")
print(f"Failed:            {failed}")

print(f"\n=== Qwen2-VL 7B Zero-Shot Results ===")
print(classification_report(
    valid['true_label'],
    valid['pred_label'],
    target_names=CLASS_ORDER,
    zero_division=0))
kappa = cohen_kappa_score(
    valid['true_label'], valid['pred_label'],
    weights='quadratic')
wf1   = f1_score(valid['true_label'], valid['pred_label'],
                  average='weighted', zero_division=0)
print(f"Weighted F1: {wf1:.4f}")
print(f"Kappa:       {kappa:.4f}")

r = {'wf1': float(wf1), 'kappa': float(kappa),
     'n_valid': len(valid), 'n_total': len(df_results)}
with open(RESULTS_DIR/'qwen2vl_metrics.json','w') as f:
    json.dump(r, f, indent=2)
print(f"\nSaved: vlm_comparison/qwen2vl_results.csv")
print(f"Saved: vlm_comparison/qwen2vl_metrics.json")

# Compare against your pipeline
print(f"\n=== VLM COMPARISON ===")
print(f"{'Method':35s}  {'wF1':>7}  {'Kappa':>7}")
print(f"  {'-'*52}")
print(f"  {'Image only (frozen CLIP)':33s}  {'0.504':>7}  {'0.313':>7}  CV")
print(f"  {'Qwen2-VL 7B zero-shot':33s}  {wf1:7.3f}  {kappa:7.3f}  CV")
print(f"  {'Full fusion XGBoost ★':33s}  {'0.958':>7}  {'0.946':>7}  CV")
print(f"\n  Gap (fusion vs Qwen2-VL): {0.958-wf1:+.3f} wF1")
print(f"  This gap = value of structured FG annotation")
print(f"  over best general-purpose VLM")
# QWEN_ZEROSHOT