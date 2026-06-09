"""
LLM TEXT GENERATION FOR UNANNOTATED CROPS
==========================================
Uses LLaVA-Med and LLaVA-1.5 locally to generate clinical notes
for all crops, using predicted FG features as structured context.

Models (free, run locally on A6000):
  - microsoft/llava-med-v1.5-mistral-7b  (biomedical VLM)
  - llava-hf/llava-1.5-7b-hf             (general VLM)

Input:
  predicted_fg_features.csv  (from seg_benchmark.py)
  tooth_crops/               (all crop images)

Output:
  llava_med_notes.csv        (generated clinical notes)
  llava_15_notes.csv         (generated clinical notes)
  text_comparison.csv        (BERTScore vs expert notes)

Run:
  python3 llava_generate.py
"""

import os, json, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          AutoTokenizer)

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR    = Path('/data1/neena/finegrain_alpha_experiments')
CROP_DIR    = WORK_DIR / 'tooth_crops'
PRED_FEAT   = WORK_DIR / 'experiment_results/segmentation/predicted_fg_features.csv'
ANNOT_CLEAN = WORK_DIR / 'annotation_clean.csv'
RESULTS_DIR = WORK_DIR / 'experiment_results' / 'llm_generation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FG_CLASSES  = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

# ── Load predicted features ───────────────────────────────────────────────────
pred_df = pd.read_csv(PRED_FEAT)
print(f"\nLoaded predicted features: {len(pred_df)} crops")

# Load expert notes for comparison (annotated subset only)
df_ann = pd.read_csv(ANNOT_CLEAN)
expert_notes = dict(zip(df_ann['crop name'],
                         df_ann['clinical_note'].fillna('')))


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(row, model_type='llava_med'):
    """
    Build structured prompt using predicted FG features as context.
    The predicted features ground the LLM response in what the
    segmentation model actually detected.
    """
    present = [c for c in FG_CLASSES
               if row.get(f'pred_{c}', 0) > 0]

    conf_info = []
    for c in present:
        conf = row.get(f'pred_{c}_conf', 0)
        if conf > 0:
            conf_info.append(f"{c} (confidence: {conf:.2f})")

    if present:
        feat_str = ', '.join(conf_info if conf_info else present)
        context  = f"Automated analysis detected: {feat_str}."
    else:
        context = "No pathological features detected by automated analysis."

    if model_type == 'llava_med':
        prompt = (
            f"<image>\n"
            f"You are a dental clinician examining an intraoral tooth photograph.\n"
            f"Context from automated feature detection: {context}\n\n"
            f"Based on what you observe in the image, provide a concise clinical "
            f"description (1-2 sentences) covering:\n"
            f"1. Visible surface changes or lesion characteristics\n"
            f"2. Clinical significance and recommended action\n"
            f"Be specific about location, color, and extent of any findings.\n"
            f"Clinical description:"
        )
    else:  # llava-1.5
        prompt = (
            f"USER: <image>\n"
            f"You are examining an intraoral dental photograph. "
            f"Automated analysis found: {context}\n"
            f"Describe the visible dental findings in 1-2 clinical sentences, "
            f"including any discolouration, structural changes, or lesion "
            f"characteristics you observe.\n"
            f"ASSISTANT:"
        )
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 — LLaVA-Med
# ════════════════════════════════════════════════════════════════════════════

def run_llava_inference(model_name, model_type, pred_df,
                         crop_dir, batch_size=1,
                         max_new_tokens=120):
    """
    Run LLaVA inference on all crops.
    Returns DataFrame with generated notes.
    """
    print(f"\nLoading {model_name}...")

    try:
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            load_in_4bit=True,    # 4-bit quantisation — ~8GB VRAM
            bnb_4bit_compute_dtype=torch.float16,
        )
        model.eval()
        print(f"  Model loaded in 4-bit quantisation")
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        print(f"  Trying without quantisation...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
            )
            model.eval()
        except Exception as e2:
            print(f"  Could not load model: {e2}")
            return None

    records = []
    failed  = 0

    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df),
                       desc=f"{model_type}"):
        img_path = Path(crop_dir) / row['crop name']
        note     = ''

        if img_path.exists():
            try:
                img    = Image.open(img_path).convert('RGB')
                prompt = build_prompt(row, model_type)

                inputs = processor(
                    text=prompt,
                    images=img,
                    return_tensors='pt',
                    padding=True,
                ).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens  = max_new_tokens,
                        do_sample       = False,
                        temperature     = 1.0,
                        repetition_penalty = 1.2,
                        pad_token_id    = processor.tokenizer.eos_token_id,
                    )

                # Decode only new tokens
                input_len = inputs['input_ids'].shape[1]
                new_ids   = output_ids[0][input_len:]
                note      = processor.tokenizer.decode(
                    new_ids, skip_special_tokens=True).strip()

                # Clean up common artefacts
                note = note.replace('\n', ' ').strip()
                if len(note) < 10:
                    note = ''

            except Exception as e:
                failed += 1
                note = ''

        records.append({
            'crop name':      row['crop name'],
            'gross_label':    row['gross_label'],
            'split':          row['split'],
            'patient_id':     row.get('patient_id', 'unknown'),
            'predicted_feats': ', '.join([c for c in FG_CLASSES
                                          if row.get(f'pred_{c}',0)>0]),
            'generated_note': note,
            'note_len':       len(note.split()) if note else 0,
        })

    print(f"  Generated: {len(records)-failed}/{len(records)}  "
          f"Failed: {failed}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return pd.DataFrame(records)


# Run LLaVA-Med
results_med = run_llava_inference(
    model_name  = 'microsoft/llava-med-v1.5-mistral-7b',
    model_type  = 'llava_med',
    pred_df     = pred_df,
    crop_dir    = CROP_DIR,
    max_new_tokens = 120,
)

if results_med is not None:
    results_med.to_csv(RESULTS_DIR/'llava_med_notes.csv', index=False)
    print(f"Saved: llava_med_notes.csv")
    usable = (results_med['note_len'] >= 6).sum()
    print(f"Usable notes (≥6 words): {usable}/{len(results_med)}")


# Run LLaVA-1.5 7B
results_15 = run_llava_inference(
    model_name  = 'llava-hf/llava-1.5-7b-hf',
    model_type  = 'llava_15',
    pred_df     = pred_df,
    crop_dir    = CROP_DIR,
    max_new_tokens = 120,
)

if results_15 is not None:
    results_15.to_csv(RESULTS_DIR/'llava_15_notes.csv', index=False)
    print(f"Saved: llava_15_notes.csv")
    usable = (results_15['note_len'] >= 6).sum()
    print(f"Usable notes (≥6 words): {usable}/{len(results_15)}")


# ════════════════════════════════════════════════════════════════════════════
# TEXT QUALITY EVALUATION — BERTScore vs Expert Notes
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEXT QUALITY EVALUATION — BERTScore vs Expert Notes")
print("="*60)

try:
    from bert_score import score as bertscore
    HAS_BERTSCORE = True
except ImportError:
    print("bert_score not installed — pip install bert-score")
    HAS_BERTSCORE = False

def eval_text_quality(generated_df, expert_notes, model_label):
    """
    Compare generated notes against expert notes on annotated crops.
    Uses BERTScore (semantic similarity) and basic stats.
    """
    # Only evaluate on annotated crops with usable expert notes
    ann_mask = generated_df['crop name'].isin(expert_notes.keys())
    ann_sub  = generated_df[ann_mask].copy()
    ann_sub['expert_note'] = ann_sub['crop name'].map(expert_notes)
    usable = ann_sub[
        (ann_sub['note_len'] >= 6) &
        (ann_sub['expert_note'].str.len() > 10)
    ].copy()

    print(f"\n  {model_label}:")
    print(f"    Annotated crops evaluated: {len(usable)}/{len(ann_sub)}")
    print(f"    Mean generated note length: "
          f"{ann_sub['note_len'].mean():.1f} words")

    if len(usable) == 0 or not HAS_BERTSCORE:
        return {}

    # BERTScore
    P, R, F1 = bertscore(
        usable['generated_note'].tolist(),
        usable['expert_note'].tolist(),
        lang='en',
        model_type='microsoft/deberta-xlarge-mnli',
        verbose=False,
    )
    results = {
        'bert_precision': float(P.mean()),
        'bert_recall':    float(R.mean()),
        'bert_f1':        float(F1.mean()),
        'n_evaluated':    len(usable),
    }
    print(f"    BERTScore F1:        {results['bert_f1']:.4f}")
    print(f"    BERTScore Precision: {results['bert_precision']:.4f}")
    print(f"    BERTScore Recall:    {results['bert_recall']:.4f}")
    return results


quality_results = {}

if results_med is not None:
    quality_results['LLaVA-Med'] = eval_text_quality(
        results_med, expert_notes, 'LLaVA-Med 7B')

if results_15 is not None:
    quality_results['LLaVA-1.5'] = eval_text_quality(
        results_15, expert_notes, 'LLaVA-1.5 7B')

with open(RESULTS_DIR/'text_quality_results.json', 'w') as f:
    json.dump(quality_results, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# SAMPLE OUTPUT — Show 5 examples
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SAMPLE GENERATED NOTES")
print("="*60)

for model_label, results_df in [
    ('LLaVA-Med',  results_med),
    ('LLaVA-1.5',  results_15),
]:
    if results_df is None:
        continue
    print(f"\n{model_label} — 5 examples:")
    sample = results_df[results_df['note_len'] >= 6].head(5)
    for _, row in sample.iterrows():
        expert = expert_notes.get(row['crop name'], 'N/A')
        print(f"\n  Crop:      {row['crop name']}")
        print(f"  Label:     {row['gross_label']}")
        print(f"  Pred feats:{row['predicted_feats']}")
        print(f"  Generated: {row['generated_note'][:120]}...")
        if expert and expert != 'N/A':
            print(f"  Expert:    {str(expert)[:120]}...")

print(f"\nAll outputs saved to: {RESULTS_DIR}/")
print("Next: run fusion_with_predictions.py")
print("✓ LLM text generation complete.")
