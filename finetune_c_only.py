"""
FINE-TUNING SCRIPT
===================
Fine-tunes BiomedCLIP (image) and PubMedBERT (text) on your dental gross labels.

Two experiments:
  A) BiomedCLIP linear probe  — fast, ~20 min
  B) BiomedCLIP full fine-tune — thorough, ~2 hours
  C) PubMedBERT fine-tune     — text classification on pseudo/FG text

Data split:
  Train+val (2270 crops): used for training and validation
  Test (623 crops):       sealed, evaluated once at the end

Run:
  cd /data1/neena/finegrain_alpha_experiments
  python3 finetune_encoders.py

Outputs saved to: experiment_results/finetune/
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer,
                          EarlyStoppingCallback)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report,
                              cohen_kappa_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import open_clip
import evaluate

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR    = Path('/data1/neena/finegrain_alpha_experiments')
COMBINED    = WORK_DIR / 'combined_all_crops_v2.csv'
CROP_DIR    = WORK_DIR / 'tooth_crops'
RESULTS_DIR = WORK_DIR / 'experiment_results' / 'finetune'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
PUBMED_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
CLIP_NAME   = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

le = LabelEncoder()
le.fit(CLASS_ORDER)

print(f"Device: {DEVICE}")
print(f"Results: {RESULTS_DIR}/")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(COMBINED)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)

df_trainval = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
df_test     = df[df['split'] == 'test'].reset_index(drop=True)

# Use val subset from trainval for validation during training
df_tr  = df[df['split'] == 'train'].reset_index(drop=True)
df_val = df[df['split'] == 'val'].reset_index(drop=True)

print(f"\nTrain: {len(df_tr)}  Val: {len(df_val)}  Test: {len(df_test)}")
print(f"Labels: {df_tr['gross_label'].value_counts().to_dict()}")


# ── Text builder ──────────────────────────────────────────────────────────────
def build_pseudo_text(row):
    """Use FG text if available, else ADA template."""
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note    = str(row.get('clinical_note', '')).strip()
    note    = '' if note in ['nan', 'none', ''] else note
    action  = str(row.get('recommended_action', '')).strip().lower()

    if present or note:
        parts = []
        if present:
            parts.append(f"Fine-grain features: {', '.join(present)}.")
        if action and action not in ['nan', '']:
            parts.append(f"Recommended action: {action}.")
        if note:
            parts.append(note)
        return ' '.join(parts)
    else:
        templates = {
            'Normal':
                'Intraoral tooth image. Sound tooth structure. '
                'No clinically detectable lesion. Normal color, '
                'translucency, and gloss. ICDAS 0.',
            'Pre-caries':
                'Intraoral tooth image. Initial caries lesion. '
                'Earliest detectable demineralization, limited to enamel. '
                'Lesion may be white or brown. Enamel has lost normal gloss. '
                'Non-cavitated. ICDAS 1-2.',
            'Caries':
                'Intraoral tooth image. Moderate to advanced caries. '
                'Visible enamel breakdown, cavitation, or dentin exposure. '
                'ICDAS 3-6.',
            'Decolor':
                'Intraoral tooth image. Non-carious discolouration. '
                'Extrinsic staining or intrinsic discolouration. '
                'No demineralization or cavitation present.',
        }
        return templates.get(str(row.get('gross_label', '')),
                             'Intraoral tooth image.')


def make_metrics_dict(y_true, y_pred):
    rep = classification_report(y_true, y_pred,
                                 target_names=CLASS_ORDER,
                                 output_dict=True, zero_division=0)
    return {
        'wf1':   rep['weighted avg']['f1-score'],
        'pc_f1': rep['Pre-caries']['f1-score'],
        'ca_f1': rep['Caries']['f1-score'],
        'dc_f1': rep['Decolor']['f1-score'],
        'no_f1': rep['Normal']['f1-score'],
        'kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
    }


# ════════════════════════════════════════════════════════════════════════════

# ── Skip flags — set True to skip completed experiments ──────────────────
SKIP_A = True   # Linear probe — already done
SKIP_B = True   # Full fine-tune — already done

# Load saved results from A and B
import json
from pathlib import Path
RESULTS_DIR = Path('/data1/neena/finegrain_alpha_experiments/experiment_results/finetune')

r_probe = json.load(open(RESULTS_DIR/'probe_test_results.json'))
r_ft    = json.load(open(RESULTS_DIR/'ft_test_results.json'))
print(f"Loaded probe results:    wF1={r_probe['wf1']:.3f}")
print(f"Loaded fine-tune results: wF1={r_ft['wf1']:.3f}")

# EXPERIMENT C — PubMedBERT FINE-TUNE ON TEXT
# Fine-tunes PubMedBERT for sequence classification
# Input: pseudo-text (FG text for annotated, ADA template for unannotated)
# Evaluated on test using ADA template text (only text available at test)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT C — PubMedBERT FINE-TUNE")
print("="*60)

from datasets import Dataset as HFDataset

tokenizer = AutoTokenizer.from_pretrained(PUBMED_NAME)

label2id = {c: i for i, c in enumerate(CLASS_ORDER)}
id2label = {i: c for c, i in label2id.items()}

# Build text datasets
# Train: pseudo-text (FG text where available, ADA template otherwise)
# Val:   same
# Test:  ADA template only (no FG annotations on test)
df_tr['text']  = [build_pseudo_text(r) for _, r in df_tr.iterrows()]
df_val['text'] = [build_pseudo_text(r) for _, r in df_val.iterrows()]

# For test — only ADA template (honest: no FG info available)
def build_ada_text(row):
    templates = {
        'Normal':
            'Intraoral tooth image. Sound tooth structure. '
            'No clinically detectable lesion. Normal color, '
            'translucency, and gloss. ICDAS 0.',
        'Pre-caries':
            'Intraoral tooth image. Initial caries lesion. '
            'Earliest detectable demineralization, limited to enamel. '
            'Lesion may be white or brown. Enamel has lost normal gloss. '
            'Non-cavitated. ICDAS 1-2.',
        'Caries':
            'Intraoral tooth image. Moderate to advanced caries. '
            'Visible enamel breakdown, cavitation, or dentin exposure. '
            'ICDAS 3-6.',
        'Decolor':
            'Intraoral tooth image. Non-carious discolouration. '
            'Extrinsic staining or intrinsic discolouration. '
            'No demineralization or cavitation present.',
    }
    return templates.get(str(row.get('gross_label', '')),
                         'Intraoral tooth image.')

df_test['text'] = [build_ada_text(r) for _, r in df_test.iterrows()]

df_tr['label']   = df_tr['gross_label'].map(label2id)
df_val['label']  = df_val['gross_label'].map(label2id)
df_test['label'] = df_test['gross_label'].map(label2id)

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True,
                     padding='max_length', max_length=128)

ds_tr_hf  = HFDataset.from_pandas(
    df_tr[['text','label']]).map(tokenize_fn, batched=True)
ds_val_hf = HFDataset.from_pandas(
    df_val[['text','label']]).map(tokenize_fn, batched=True)
ds_te_hf  = HFDataset.from_pandas(
    df_test[['text','label']]).map(tokenize_fn, batched=True)

# Load model
bert_clf = AutoModelForSequenceClassification.from_pretrained(
    PUBMED_NAME,
    num_labels=4,
    id2label=id2label,
    label2id=label2id)

metric = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds,
                          references=labels, average='weighted')

# Class weights for trainer
cw_bert = compute_class_weight('balanced',
                                classes=np.arange(4),
                                y=df_tr['label'].values)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits  = outputs.get('logits')
        weights = torch.tensor(cw_bert, dtype=torch.float).to(logits.device)
        loss    = F.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir              = str(RESULTS_DIR / 'pubmedbert_ft'),
    num_train_epochs        = 15,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    evaluation_strategy     = 'epoch',
    save_strategy           = 'epoch',
    load_best_model_at_end  = True,
    metric_for_best_model   = 'f1',
    greater_is_better       = True,
    logging_steps           = 20,
    fp16                    = torch.cuda.is_available(),
    dataloader_num_workers  = 2,
    report_to               = 'none',
)

trainer = WeightedTrainer(
    model           = bert_clf,
    args            = training_args,
    train_dataset   = ds_tr_hf,
    eval_dataset    = ds_val_hf,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\nFine-tuning PubMedBERT (15 epochs, early stop)...")
trainer.train()

# Evaluate on test
print("\nEvaluating on test set...")
preds_out = trainer.predict(ds_te_hf)
y_pred_bert = np.argmax(preds_out.predictions, axis=-1)
y_true_bert = preds_out.label_ids

r_bert = make_metrics_dict(
    le.inverse_transform(y_true_bert),
    le.inverse_transform(y_pred_bert))

print(f"\nPubMedBERT Fine-tune TEST results:")
print(classification_report(
    le.inverse_transform(y_true_bert),
    le.inverse_transform(y_pred_bert),
    target_names=CLASS_ORDER, zero_division=0))
print(f"Kappa: {r_bert['kappa']:.4f}")

with open(RESULTS_DIR / 'pubmedbert_test_results.json', 'w') as f:
    json.dump(r_bert, f, indent=2)
trainer.save_model(str(RESULTS_DIR / 'pubmedbert_best'))


# ════════════════════════════════════════════════════════════════════════════

# FINAL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINE-TUNING COMPARISON — TEST SET RESULTS")
print("="*70)
print(f"{'Method':45s} {'PC':>6} {'Ca':>6} {'DC':>6} {'wF1':>6} {'Kappa':>7}")
print("-"*70)

# Load frozen baseline from main experiments
frozen_path = WORK_DIR / 'experiment_results' / 'img_only_clip.json'
if frozen_path.exists():
    with open(frozen_path) as f:
        r_frozen = json.load(f)
    print(f"  {'BiomedCLIP frozen (CV baseline)':43s} "
          f"{r_frozen.get('pc_f1',0):6.3f} {r_frozen.get('ca_f1',0):6.3f} "
          f"{r_frozen.get('dc_f1',0):6.3f} {r_frozen.get('wf1',0):6.3f} "
          f"{r_frozen.get('kappa',0):7.3f}  ← CV")

rows = [
    ('BiomedCLIP linear probe (test)',   r_probe),
    ('BiomedCLIP full fine-tune (test)', r_ft),
    ('PubMedBERT fine-tune (test)',      r_bert),
]
for name, r in rows:
    print(f"  {name:43s} "
          f"{r['pc_f1']:6.3f} {r['ca_f1']:6.3f} "
          f"{r['dc_f1']:6.3f} {r['wf1']:6.3f} "
          f"{r['kappa']:7.3f}")

print(f"\nAll results saved to: {RESULTS_DIR}/")

# ── Learning curve plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor('#F8F7F4')
fig.suptitle('Fine-tuning Training Curves', fontsize=13,
             fontweight='bold')

for ax, history, title, color in zip(
        axes,
        [probe_history, ft_history],
        ['Linear Probe', 'Full Fine-tune'],
        ['#2563EB', '#C1392B']):
    df_h = pd.DataFrame(history)
    ax.plot(df_h['epoch'], df_h['val_f1'], 'o-',
            color=color, linewidth=2, markersize=6, label='Val F1')
    ax.plot(df_h['epoch'], df_h['tr_acc'], 's--',
            color=color, alpha=0.5, linewidth=1.5,
            markersize=4, label='Train Acc')
    ax.set_title(f'BiomedCLIP {title}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.grid(color='#E8E5DF', linewidth=0.8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_facecolor('#F8F7F4')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'finetune_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.show()
print("Saved: finetune_curves.png")
print("✓ Fine-tuning complete.")
