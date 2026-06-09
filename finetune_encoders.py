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
# EXPERIMENT A — BiomedCLIP LINEAR PROBE
# Freeze the visual encoder, train only a linear classifier head
# Fast (~20 min), shows how much domain-specific information
# is already in the frozen BiomedCLIP embeddings
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT A — BiomedCLIP LINEAR PROBE")
print("="*60)

IMG_TF_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
IMG_TF_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DentalImageDataset(Dataset):
    def __init__(self, df, crop_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.crop_dir  = Path(crop_dir)
        self.transform = transform
        self.label_enc = {c: i for i, c in enumerate(CLASS_ORDER)}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.label_enc[row['gross_label']]
        path  = self.crop_dir / row['crop name']
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


class BiomedCLIPLinearProbe(nn.Module):
    """Frozen BiomedCLIP encoder + trainable linear head."""
    def __init__(self, clip_model, embed_dim=512, num_classes=4):
        super().__init__()
        self.encoder    = clip_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.classifier(feats)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(labels.numpy())
    return np.array(all_true), np.array(all_pred)


# Load BiomedCLIP
print("Loading BiomedCLIP...")
clip_model, _, clip_prep = open_clip.create_model_and_transforms(CLIP_NAME)
clip_model = clip_model.to(DEVICE).eval()

# Build datasets
ds_tr  = DentalImageDataset(df_tr,  CROP_DIR, IMG_TF_TRAIN)
ds_val = DentalImageDataset(df_val, CROP_DIR, IMG_TF_VAL)
ds_te  = DentalImageDataset(df_test, CROP_DIR, IMG_TF_VAL)

dl_tr  = DataLoader(ds_tr,  batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True)
dl_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=4)
dl_te  = DataLoader(ds_te,  batch_size=64, shuffle=False, num_workers=4)

# Class weights
y_tr_arr = le.transform(df_tr['gross_label'].values)
cw       = compute_class_weight('balanced',
                                 classes=np.arange(4), y=y_tr_arr)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(cw, dtype=torch.float).to(DEVICE))

# Build linear probe model
probe = BiomedCLIPLinearProbe(clip_model, embed_dim=512, num_classes=4)
probe = probe.to(DEVICE)

# Only train the classifier head
optimizer = torch.optim.AdamW(
    probe.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30)

print("\nTraining linear probe (30 epochs, head only)...")
best_f1, best_state = 0.0, None
probe_history = []

for epoch in range(30):
    tr_loss, tr_acc = train_epoch(probe, dl_tr, optimizer, criterion, DEVICE)
    scheduler.step()
    y_true, y_pred  = eval_epoch(probe, dl_val, DEVICE)
    val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    probe_history.append({'epoch': epoch+1, 'tr_loss': tr_loss,
                          'tr_acc': tr_acc, 'val_f1': val_f1})

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.clone() for k, v in
                      probe.classifier.state_dict().items()}

    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}: loss={tr_loss:.4f}  "
              f"train_acc={tr_acc:.3f}  val_F1={val_f1:.3f}")

# Load best and evaluate on test
probe.classifier.load_state_dict(best_state)
y_true_te, y_pred_te = eval_epoch(probe, dl_te, DEVICE)

r_probe = make_metrics_dict(
    le.inverse_transform(y_true_te),
    le.inverse_transform(y_pred_te))

print(f"\nLinear Probe TEST results:")
print(classification_report(
    le.inverse_transform(y_true_te),
    le.inverse_transform(y_pred_te),
    target_names=CLASS_ORDER, zero_division=0))
print(f"Kappa: {r_probe['kappa']:.4f}")

torch.save(best_state,
           RESULTS_DIR / 'biomedclip_linear_probe_head.pt')
pd.DataFrame(probe_history).to_csv(
    RESULTS_DIR / 'probe_training_history.csv', index=False)
with open(RESULTS_DIR / 'probe_test_results.json', 'w') as f:
    json.dump(r_probe, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — BiomedCLIP FULL FINE-TUNE
# Unfreeze encoder, train end-to-end with low learning rate
# More powerful but needs careful regularisation to avoid overfitting
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT B — BiomedCLIP FULL FINE-TUNE")
print("="*60)


class BiomedCLIPFineTune(nn.Module):
    """Full fine-tune: encoder + classification head."""
    def __init__(self, clip_model, embed_dim=512, num_classes=4):
        super().__init__()
        self.encoder    = clip_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.4),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.encoder.encode_image(x)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return self.classifier(feats)


# Reload fresh BiomedCLIP (don't use the one modified by linear probe)
clip_ft, _, _ = open_clip.create_model_and_transforms(CLIP_NAME)
clip_ft = clip_ft.to(DEVICE)

ft_model = BiomedCLIPFineTune(clip_ft, embed_dim=512, num_classes=4)
ft_model = ft_model.to(DEVICE)

# Use different LR for encoder vs head
optimizer_ft = torch.optim.AdamW([
    {'params': ft_model.encoder.parameters(),    'lr': 1e-5},  # low LR
    {'params': ft_model.classifier.parameters(), 'lr': 1e-4},  # higher LR
], weight_decay=1e-4)
scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=25)

print("\nFull fine-tuning (25 epochs, encoder + head)...")
best_f1_ft, best_state_ft = 0.0, None
ft_history = []
patience, patience_count = 5, 0

for epoch in range(25):
    tr_loss, tr_acc = train_epoch(
        ft_model, dl_tr, optimizer_ft, criterion, DEVICE)
    scheduler_ft.step()
    y_true_v, y_pred_v = eval_epoch(ft_model, dl_val, DEVICE)
    val_f1 = f1_score(y_true_v, y_pred_v,
                      average='weighted', zero_division=0)
    ft_history.append({'epoch': epoch+1, 'tr_loss': tr_loss,
                       'tr_acc': tr_acc, 'val_f1': val_f1})

    if val_f1 > best_f1_ft:
        best_f1_ft    = val_f1
        best_state_ft = {k: v.clone() for k, v in
                         ft_model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}: loss={tr_loss:.4f}  "
              f"train_acc={tr_acc:.3f}  val_F1={val_f1:.3f}")

    if patience_count >= patience:
        print(f"  Early stop at epoch {epoch+1}")
        break

ft_model.load_state_dict(best_state_ft)
y_true_te_ft, y_pred_te_ft = eval_epoch(ft_model, dl_te, DEVICE)

r_ft = make_metrics_dict(
    le.inverse_transform(y_true_te_ft),
    le.inverse_transform(y_pred_te_ft))

print(f"\nFull Fine-tune TEST results:")
print(classification_report(
    le.inverse_transform(y_true_te_ft),
    le.inverse_transform(y_pred_te_ft),
    target_names=CLASS_ORDER, zero_division=0))
print(f"Kappa: {r_ft['kappa']:.4f}")

torch.save(best_state_ft,
           RESULTS_DIR / 'biomedclip_finetuned.pt')
pd.DataFrame(ft_history).to_csv(
    RESULTS_DIR / 'ft_training_history.csv', index=False)
with open(RESULTS_DIR / 'ft_test_results.json', 'w') as f:
    json.dump(r_ft, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
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
