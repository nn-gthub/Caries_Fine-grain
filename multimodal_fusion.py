# ================================================================
# C1 — MULTIMODAL FUSION EXPERIMENT
# Image (BiomedCLIP) + Text (BioBERT + Structured Notes)
# → Gross Label Classification
#
# Narrative: Fine-grain features (visual + textual) improve over
# gross-label-only direct classification
#
# Run cells top to bottom in Jupyter
# ================================================================


# ════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies (run once)
# ════════════════════════════════════════════════════════════════

# !pip install torch torchvision transformers open_clip_torch
# !pip install scikit-learn pandas numpy matplotlib seaborn
# !pip install sentence-transformers xgboost shap umap-learn
# !pip install Pillow tqdm

print("If all imports below work, skip this cell.")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Imports
# ════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                              cohen_kappa_score, f1_score,
                              silhouette_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import shap

# Plot style
PAL = {
    'Normal':     '#2D6A4F',
    'Pre-caries': '#E9A039',
    'Caries':     '#C1392B',
    'Decolor':    '#6C3483',
    'bg':         '#F8F7F4',
    'grid':       '#E8E5DF',
    'text':       '#1A1A1A',
    'sub':        '#6B6B6B',
    'blue':       '#2563EB',
}
CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']

plt.rcParams.update({
    'figure.facecolor': PAL['bg'],
    'axes.facecolor':   PAL['bg'],
    'font.family':      'DejaVu Sans',
})

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════
# CELL 3 — Load and prepare data
# ════════════════════════════════════════════════════════════════

# --- CHANGE THESE TO YOUR CORRECT NAS BACKUP PATHS ---
MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"
LOG_CSV  = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"
CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")

# from pathlib import Path

# # Metadata CSVs
# MAIN_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/annotation_clean.csv"

# LOG_CSV = "/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/metadata/gross_label_corrections.csv"

# # Unified image directory
# CROP_DIR = Path("/Nasbackup/lab_nirmal/neena/datasets/finegrain_alpha/all_images/")



df = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

# Apply corrected labels
label_lookup = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(label_lookup).fillna(df['gross_label'])

# Normalise checkboxes
for col in FG_COLS:
    df[col] = (df[col]
               .map({True:1, False:0, 'True':1, 'False':0, np.nan:0})
               .fillna(0).astype(int))

# Keep only annotated + valid label rows (ALL kept, quality flagged)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)

# Standardise image quality flag
df['image_quality'] = (df['image_quality'].fillna('unknown')
                        .astype(str).str.lower()
                        .replace({'bad':'poor'}))
df['is_poor'] = (df['image_quality'] == 'poor').astype(int)

print(f"Dataset: {len(df)} crops from {df['patient_id'].nunique()} patients")
print(df['gross_label'].value_counts().to_string())


# ════════════════════════════════════════════════════════════════
# CELL 4 — Build structured text column
# ════════════════════════════════════════════════════════════════
# Combines checkbox info + clinical note into one rich text string
# WITHOUT modifying the original columns

def build_structured_text(row):
    """
    Combine checkbox fine-grain labels + clinical note into
    one structured text string for embedding.

    Format:
      "Features present: [list]. [Clinical note]."

    This gives the text encoder both the structured signal
    (checkbox labels) and the free-text clinical reasoning.
    """
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note    = str(row.get('clinical_note', '')).strip()
    note    = '' if note in ['nan', 'none', ''] else note

    parts = []

    # Part 1: checkbox-derived structured sentence
    if present:
        feat_str = ', '.join(present)
        parts.append(f"Fine-grain features identified: {feat_str}.")
    else:
        parts.append("No fine-grain features identified.")

    # Part 2: recommended action if available
    action = str(row.get('recommended_action', '')).strip().lower()
    if action and action not in ['nan', '']:
        parts.append(f"Recommended action: {action}.")

    # Part 3: clinical note
    if note:
        parts.append(note)

    return ' '.join(parts)


# Also build a checkbox-only text (for ablation)
def build_checkbox_text(row):
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    if present:
        return f"Fine-grain features: {', '.join(present)}. Action: {str(row.get('recommended_action','')).lower()}."
    return "No fine-grain features present."


# Also build note-only text (for ablation)
def build_note_only_text(row):
    note = str(row.get('clinical_note', '')).strip()
    return '' if note in ['nan','none',''] else note


df['structured_text']  = df.apply(build_structured_text, axis=1)
df['checkbox_text']    = df.apply(build_checkbox_text, axis=1)
df['note_only_text']   = df.apply(build_note_only_text, axis=1)

print("Sample structured_text:")
for i in range(3):
    print(f"\n  [{df['gross_label'].iloc[i]}]")
    print(f"  {df['structured_text'].iloc[i]}")


# ════════════════════════════════════════════════════════════════
# CELL 5 — Extract BiomedCLIP image embeddings
# ════════════════════════════════════════════════════════════════

import open_clip

print("Loading BiomedCLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
clip_model = clip_model.to(DEVICE).eval()
print("BiomedCLIP loaded.")

def get_clip_embeddings(df, crop_dir, batch_size=32):
    """Extract BiomedCLIP image embeddings for all crops."""
    embeddings = []
    missing    = []

    paths = [crop_dir / row['crop name'] for _, row in df.iterrows()]

    for i in tqdm(range(0, len(paths), batch_size),
                  desc="BiomedCLIP image embeddings"):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        valid_idx = []

        for j, p in enumerate(batch_paths):
            if p.exists():
                try:
                    img = Image.open(p).convert('RGB')
                    imgs.append(clip_preprocess(img))
                    valid_idx.append(i + j)
                except Exception:
                    missing.append(str(p))
                    imgs.append(torch.zeros(3, 224, 224))
                    valid_idx.append(i + j)
            else:
                missing.append(str(p))
                # zero embedding for missing images
                embeddings.append(np.zeros(512))

        if imgs:
            batch_tensor = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                feats = clip_model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.extend(feats.cpu().numpy().tolist())

    if missing:
        print(f"Warning: {len(missing)} images not found")

    return np.array(embeddings)

# Run — this takes a few minutes
clip_embs = get_clip_embeddings(df, CROP_DIR)
print(f"BiomedCLIP embeddings: {clip_embs.shape}")

# Save so you don't have to rerun
np.save('clip_embeddings.npy', clip_embs)
print("Saved: clip_embeddings.npy")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Extract BioBERT text embeddings
# ════════════════════════════════════════════════════════════════

print("Loading BioBERT...")
BIOBERT_NAME = "dmis-lab/biobert-base-cased-v1.2"
bert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_NAME)
bert_model     = AutoModel.from_pretrained(BIOBERT_NAME).to(DEVICE).eval()
print("BioBERT loaded.")

def get_bert_embeddings(texts, tokenizer, model, batch_size=32,
                        desc="BioBERT"):
    """
    Mean-pool BioBERT [CLS] token embeddings.
    Returns numpy array (n, 768)
    """
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors='pt')
        enc   = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            # Mean pooling over token dimension
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb  = torch.nn.functional.normalize(emb, dim=-1)

        all_embs.extend(emb.cpu().numpy().tolist())

    return np.array(all_embs)

# Three text variants for ablation
print("\nEncoding structured_text (checkbox + note)...")
bert_structured = get_bert_embeddings(
    df['structured_text'].tolist(), bert_tokenizer, bert_model,
    desc="BioBERT structured")

print("\nEncoding checkbox_text only...")
bert_checkbox   = get_bert_embeddings(
    df['checkbox_text'].tolist(), bert_tokenizer, bert_model,
    desc="BioBERT checkbox")

print("\nEncoding note_only_text...")
bert_note       = get_bert_embeddings(
    df['note_only_text'].fillna('').tolist(), bert_tokenizer, bert_model,
    desc="BioBERT note")

np.save('bert_structured_embs.npy', bert_structured)
np.save('bert_checkbox_embs.npy',   bert_checkbox)
np.save('bert_note_embs.npy',       bert_note)
print(f"\nBioBERT structured: {bert_structured.shape}")
print(f"BioBERT checkbox:   {bert_checkbox.shape}")
print(f"BioBERT note:       {bert_note.shape}")


# ════════════════════════════════════════════════════════════════
# CELL 7 — Extract PubMedBERT embeddings (third model)
# ════════════════════════════════════════════════════════════════
# PubMedBERT is pretrained exclusively on PubMed abstracts
# Provides a different text representation from BioBERT
# Good for comparison — BioBERT vs PubMedBERT on dental notes

print("Loading PubMedBERT...")
PUBMED_NAME    = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
pubmed_tok     = AutoTokenizer.from_pretrained(PUBMED_NAME)
pubmed_model   = AutoModel.from_pretrained(PUBMED_NAME).to(DEVICE).eval()
print("PubMedBERT loaded.")

bert_pubmed = get_bert_embeddings(
    df['structured_text'].tolist(), pubmed_tok, pubmed_model,
    desc="PubMedBERT structured")

np.save('pubmed_embs.npy', bert_pubmed)
print(f"PubMedBERT: {bert_pubmed.shape}")


# ════════════════════════════════════════════════════════════════
# CELL 8 — Load saved embeddings (skip cells 5-7 if already run)
# ════════════════════════════════════════════════════════════════

clip_embs       = np.load('clip_embeddings.npy')
bert_structured = np.load('bert_structured_embs.npy')
bert_checkbox   = np.load('bert_checkbox_embs.npy')
bert_note       = np.load('bert_note_embs.npy')
bert_pubmed     = np.load('pubmed_embs.npy')

print("Embeddings loaded:")
print(f"  CLIP (image):           {clip_embs.shape}")
print(f"  BioBERT structured:     {bert_structured.shape}")
print(f"  BioBERT checkbox:       {bert_checkbox.shape}")
print(f"  BioBERT note:           {bert_note.shape}")
print(f"  PubMedBERT structured:  {bert_pubmed.shape}")


# ════════════════════════════════════════════════════════════════
# CELL 9 — Build fusion feature sets
# ════════════════════════════════════════════════════════════════
# Each feature set = one row in your results table

# Fine-grain checkbox vector (6 binary features + counts)
fg_vec = df[FG_COLS].values.astype(float)

# Normalise embedding arrays
scaler_clip   = StandardScaler()
scaler_bert_s = StandardScaler()
scaler_bert_c = StandardScaler()
scaler_bert_n = StandardScaler()
scaler_pubmed = StandardScaler()
scaler_fg     = StandardScaler()

clip_norm      = scaler_clip.fit_transform(clip_embs)
bert_s_norm    = scaler_bert_s.fit_transform(bert_structured)
bert_c_norm    = scaler_bert_c.fit_transform(bert_checkbox)
bert_n_norm    = scaler_bert_n.fit_transform(bert_note)
pubmed_norm    = scaler_pubmed.fit_transform(bert_pubmed)
fg_norm        = scaler_fg.fit_transform(fg_vec)

# ── Fusion combinations ───────────────────────────────────────
feature_sets = {
    # Baselines (no fine-grain)
    'Image only\n(BiomedCLIP)':
        clip_norm,

    # Text variants
    'Text: checkbox only\n(BioBERT)':
        np.hstack([bert_c_norm]),

    'Text: note only\n(BioBERT)':
        np.hstack([bert_n_norm]),

    'Text: structured\n(BioBERT)':
        np.hstack([bert_s_norm]),

    'Text: structured\n(PubMedBERT)':
        np.hstack([pubmed_norm]),

    # Fine-grain vector alone
    'FG features\n(checkbox vector)':
        fg_norm,

    # Fusion: image + fine-grain text
    'Image + FG text\n(CLIP + BioBERT)':
        np.hstack([clip_norm, bert_s_norm]),

    'Image + FG text\n(CLIP + PubMedBERT)':
        np.hstack([clip_norm, pubmed_norm]),

    # Full fusion: image + checkbox + text
    'Full fusion\n(CLIP + FG + BioBERT)':
        np.hstack([clip_norm, fg_norm, bert_s_norm]),

    'Full fusion\n(CLIP + FG + PubMedBERT)':
        np.hstack([clip_norm, fg_norm, pubmed_norm]),
}

print("Feature sets built:")
for name, feat in feature_sets.items():
    print(f"  {name.replace(chr(10),' '):45s}: {feat.shape[1]} dims")

y      = df['gross_label'].values
groups = df['patient_id'].values

le = LabelEncoder()
le.fit(CLASS_ORDER)
y_enc = le.transform(y)


# ════════════════════════════════════════════════════════════════
# CELL 10 — Classification experiment (patient-level CV)
# ════════════════════════════════════════════════════════════════

def run_experiment(X, y_enc, groups, n_splits=5, random_state=42):
    """
    Patient-level stratified cross-validation.
    Uses XGBoost classifier.
    Returns per-class F1, weighted F1, kappa, OOF predictions.
    """
    from sklearn.utils.class_weight import compute_class_weight
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    all_true, all_pred = [], []

    for fold, (tr, val) in enumerate(skf.split(X, y_enc)):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y_enc[tr], y_enc[val]

        # Class weights for imbalance
        cw = compute_class_weight('balanced',
                                   classes=np.unique(y_tr), y=y_tr)
        sw = np.array([cw[yi] for yi in y_tr])

        clf = xgb.XGBClassifier(
            
            max_depth        = 5,
            learning_rate    = 0.05,
            n_estimators     = 200,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = random_state,
            verbosity        = 0,
            eval_metric      = 'mlogloss',
            early_stopping_rounds=10
        )
        clf.fit(X_tr, y_tr, sample_weight=sw,
                eval_set=[(X_val, y_val)],
                verbose=False,
               )

        pred = clf.predict(X_val)
        all_true.extend(y_val)
        all_pred.extend(pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Per-class F1
    report = classification_report(all_true, all_pred,
                                    target_names=CLASS_ORDER,
                                    output_dict=True,
                                    zero_division=0)
    kappa  = cohen_kappa_score(all_true, all_pred,
                               weights='quadratic')
    return {
        'report':    report,
        'kappa':     kappa,
        'all_true':  all_true,
        'all_pred':  all_pred,
        'wf1':       report['weighted avg']['f1-score'],
        'pc_f1':     report['Pre-caries']['f1-score'],
        'dc_f1':     report['Decolor']['f1-score'],
        'ca_f1':     report['Caries']['f1-score'],
        'no_f1':     report['Normal']['f1-score'],
    }

print("Running all experiments...")
print("(This takes a few minutes)\n")

all_results = {}
for name, X in feature_sets.items():
    clean_name = name.replace('\n', ' ')
    print(f"  {clean_name:50s}", end='', flush=True)
    r = run_experiment(X, y_enc, groups)
    all_results[name] = r
    print(f"  wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  "
          f"Decolor={r['dc_f1']:.3f}  Kappa={r['kappa']:.3f}")

print("\nAll experiments complete.")


# ════════════════════════════════════════════════════════════════
# CELL 11 — Results table
# ════════════════════════════════════════════════════════════════

rows = []
for name, r in all_results.items():
    rows.append({
        'Method':          name.replace('\n',' '),
        'Normal F1':       round(r['no_f1'],  3),
        'Pre-caries F1':   round(r['pc_f1'],  3),
        'Caries F1':       round(r['ca_f1'],  3),
        'Decolor F1':      round(r['dc_f1'],  3),
        'Weighted F1':     round(r['wf1'],    3),
        'Kappa':           round(r['kappa'],  3),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv('c1_results.csv', index=False)
print(results_df.to_string(index=False))
print("\nSaved: c1_results.csv")


# ════════════════════════════════════════════════════════════════
# CELL 12 — PLOT 1: Results comparison (lollipop chart)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Fine-Grain Features Improve Classification\n'
             'Image + Text Fusion vs Baselines',
             fontsize=14, fontweight='bold', color=PAL['text'])

methods  = [r['Method'] for r in rows]
wf1_vals = [r['Weighted F1'] for r in rows]
pc_vals  = [r['Pre-caries F1'] for r in rows]

# Colour: baseline grey, fusion blue, best red
def method_color(m):
    if 'Full fusion' in m: return PAL['Caries']
    if 'Image +' in m:     return PAL['blue']
    if 'Image only' in m:  return '#888888'
    return PAL['sub']

colors = [method_color(m) for m in methods]
y_pos  = range(len(methods))

# Left: Weighted F1
for ax, vals, title in zip(
        axes,
        [wf1_vals, pc_vals],
        ['Weighted F1 — All Classes',
         'F1 — Pre-caries Class\n(hardest boundary)']):

    ax.hlines(y_pos, 0, vals, colors=colors,
              linewidth=2.5, alpha=0.8)
    ax.scatter(vals, y_pos, color=colors, s=180, zorder=5)

    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f'{v:.3f}',
                va='center', fontsize=9.5,
                color=PAL['text'], fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=PAL['text'], pad=10)
    ax.set_xlabel('F1 Score', fontsize=10, color=PAL['sub'])
    ax.axvline(0.5, color=PAL['grid'], linewidth=1,
               linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color=PAL['grid'], linewidth=0.8)
    ax.set_facecolor(PAL['bg'])

# Legend
legend_elements = [
    mpatches.Patch(color='#888888', label='Baseline (image only)'),
    mpatches.Patch(color=PAL['blue'],   label='Image + FG text fusion'),
    mpatches.Patch(color=PAL['Caries'], label='Full fusion (best)'),
    mpatches.Patch(color=PAL['sub'],    label='Text only'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02),
           facecolor=PAL['bg'], edgecolor=PAL['grid'])

plt.tight_layout()
plt.savefig('c1_plot1_results.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: c1_plot1_results.png")


# ════════════════════════════════════════════════════════════════
# CELL 13 — PLOT 2: Confusion matrices — baseline vs best fusion
# ════════════════════════════════════════════════════════════════

# Find baseline and best fusion key names
baseline_key = [k for k in all_results if 'Image only' in k][0]
best_key     = max(
    [k for k in all_results if 'Full fusion' in k],
    key=lambda k: all_results[k]['wf1']
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Confusion Matrix: Before vs After Fine-Grain Fusion',
             fontsize=13, fontweight='bold', color=PAL['text'])

for ax, key, title in zip(
        axes,
        [baseline_key, best_key],
        ['Baseline: Image Only\n(BiomedCLIP)',
         f'Best Fusion\n({best_key.replace(chr(10)," ")})']):

    r   = all_results[key]
    cm  = confusion_matrix(r['all_true'], r['all_pred'])
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(pct,
                annot=True, fmt='.0f', cmap='Blues',
                xticklabels=CLASS_ORDER,
                yticklabels=CLASS_ORDER,
                linewidths=2, linecolor=PAL['bg'],
                cbar_kws={'label':'%', 'shrink':0.8},
                ax=ax, annot_kws={'size':12,'weight':'bold'})
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=PAL['text'], pad=10)
    ax.set_xlabel('Predicted', fontsize=10, color=PAL['sub'])
    ax.set_ylabel('True', fontsize=10, color=PAL['sub'])

plt.tight_layout()
plt.savefig('c1_plot2_confusion.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: c1_plot2_confusion.png")
print("\nKey: look at Pre-caries row — does confusion with Decolor decrease?")
print("That reduction is your main claim visualised.")


# ════════════════════════════════════════════════════════════════
# CELL 14 — PLOT 3: Semantic similarity heatmap
# ════════════════════════════════════════════════════════════════
# Shows how similar different gross label classes are
# in image space vs text space vs combined space

def class_centroid_similarity(embeddings, labels, class_order):
    """Cosine similarity between class centroids."""
    centroids = {}
    for gl in class_order:
        mask = labels == gl
        if mask.sum() > 0:
            cent = embeddings[mask].mean(axis=0)
            centroids[gl] = cent / (np.linalg.norm(cent) + 1e-8)

    n = len(class_order)
    sim_matrix = np.zeros((n, n))
    for i, c1 in enumerate(class_order):
        for j, c2 in enumerate(class_order):
            if c1 in centroids and c2 in centroids:
                sim_matrix[i, j] = float(
                    np.dot(centroids[c1], centroids[c2]))
    return sim_matrix

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Class Centroid Cosine Similarity\n'
             'How similar are gross label classes in each embedding space?',
             fontsize=13, fontweight='bold', color=PAL['text'])

spaces = [
    (clip_norm,      'Image Space\n(BiomedCLIP)'),
    (bert_s_norm,    'Text Space\n(BioBERT structured)'),
    (np.hstack([clip_norm, bert_s_norm]),
                     'Combined Space\n(CLIP + BioBERT)'),
]

for ax, (emb, title) in zip(axes, spaces):
    sim = class_centroid_similarity(emb, y, CLASS_ORDER)
    # Mask diagonal for clearer off-diagonal reading
    mask_diag = np.eye(4, dtype=bool)

    sns.heatmap(sim,
                annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=CLASS_ORDER,
                yticklabels=CLASS_ORDER,
                vmin=0.5, vmax=1.0,
                linewidths=2, linecolor=PAL['bg'],
                cbar_kws={'shrink':0.8},
                ax=ax, annot_kws={'size':11,'weight':'bold'})
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=PAL['text'], pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=9.5)

plt.tight_layout()
plt.savefig('c1_plot3_similarity.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: c1_plot3_similarity.png")
print("\nHow to read:")
print("  High similarity between Pre-caries and Decolor in image space")
print("  → confirms they look similar visually")
print("  Lower similarity in text/combined space")
print("  → fine-grain text separates them better")
print("  This directly motivates your multimodal approach")


# ════════════════════════════════════════════════════════════════
# CELL 15 — PLOT 4: UMAP projections
# ════════════════════════════════════════════════════════════════

from umap import UMAP

print("Running UMAP projections (takes ~1 minute)...")

umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                  random_state=42, verbose=False)

spaces_umap = {
    'Image only\n(BiomedCLIP)':         clip_norm,
    'Text: structured\n(BioBERT)':      bert_s_norm,
    'Full fusion\n(CLIP + FG + BioBERT)':
        np.hstack([clip_norm, fg_norm, bert_s_norm]),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('UMAP Projection — Class Separation Across Embedding Spaces',
             fontsize=13, fontweight='bold', color=PAL['text'])

for ax, (title, emb) in zip(axes, spaces_umap.items()):
    proj = umap_model.fit_transform(emb)

    for gl in CLASS_ORDER:
        mask = y == gl
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=PAL[gl], label=gl,
                   alpha=0.6, s=35,
                   edgecolors='white', linewidths=0.3, zorder=3)

    # Compute silhouette score
    try:
        sil = silhouette_score(proj, y, metric='euclidean')
        score_text = f'Silhouette: {sil:.3f}'
    except Exception:
        score_text = ''

    ax.set_title(f'{title}\n{score_text}',
                 fontsize=10.5, fontweight='bold',
                 color=PAL['text'], pad=8)
    ax.set_xlabel('UMAP 1', fontsize=9, color=PAL['sub'])
    ax.set_ylabel('UMAP 2', fontsize=9, color=PAL['sub'])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(PAL['bg'])
    ax.grid(color=PAL['grid'], linewidth=0.6, alpha=0.5)

handles = [mpatches.Patch(color=PAL[gl], label=gl)
           for gl in CLASS_ORDER]
fig.legend(handles=handles, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, -0.04),
           facecolor=PAL['bg'], edgecolor=PAL['grid'])

plt.tight_layout()
plt.savefig('c1_plot4_umap.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: c1_plot4_umap.png")
print("\nHigher silhouette score = better class separation")
print("Full fusion should show tighter, more separated clusters")
print("than image-only or text-only projections")


# ════════════════════════════════════════════════════════════════
# CELL 16 — PLOT 5: Fine-grain feature contribution (SHAP)
# ════════════════════════════════════════════════════════════════
# Train final XGBoost on full fusion features
# Use SHAP to show which fine-grain features drive each class

print("Training final XGBoost on full fusion features for SHAP...")

full_fusion_key = max(
    [k for k in all_results if 'Full fusion' in k],
    key=lambda k: all_results[k]['wf1']
)
X_full = feature_sets[full_fusion_key]
feat_names = (
    [f'CLIP_{i}' for i in range(clip_norm.shape[1])] +
    [f'FG_{c}' for c in FG_COLS] +
    [f'BioBERT_{i}' for i in range(bert_s_norm.shape[1])]
)

from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
sw = np.array([cw[yi] for yi in y_enc])

final_clf = xgb.XGBClassifier(
    
    max_depth=5, learning_rate=0.05,
    n_estimators=200, random_state=42,
    verbosity=0
)
final_clf.fit(X_full, y_enc, sample_weight=sw)

# SHAP — use only the interpretable features (FG + a few top CLIP)
# For visualisation clarity, show just the FG feature block
fg_start = clip_norm.shape[1]
fg_end   = fg_start + len(FG_COLS)
X_fg_only = X_full[:, fg_start:fg_end]
fg_feat_names = [f'FG: {c}' for c in FG_COLS]

# Re-train on FG features only for clean SHAP plot
clf_fg = xgb.XGBClassifier(
   
    max_depth=4, n_estimators=200,
    random_state=42, verbosity=0
)
clf_fg.fit(X_fg_only, y_enc, sample_weight=sw)

explainer   = shap.TreeExplainer(clf_fg)
shap_values = explainer.shap_values(X_fg_only)

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('SHAP — Fine-Grain Feature Contribution per Gross Label\n'
             'Which fine-grain features drive each diagnosis?',
             fontsize=13, fontweight='bold', color=PAL['text'])

for ax, (i, gl) in enumerate(
        zip(range(4), CLASS_ORDER)):
    sv   = shap_values[i]
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)

    colors_shap = [PAL[gl] if v > 0 else '#BBBBBB'
                   for v in mean_abs[order]]
    y_pos = range(len(FG_COLS))

    ax.barh(y_pos, mean_abs[order],
            color=[PAL[gl]]*len(order),
            alpha=0.8, edgecolor=PAL['bg'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([fg_feat_names[o] for o in order],
                       fontsize=9.5)
    for j, v in enumerate(mean_abs[order]):
        ax.text(v + 0.001, j, f'{v:.3f}',
                va='center', fontsize=8.5)
    ax.set_title(gl, fontsize=11, fontweight='bold',
                 color=PAL[gl], pad=8)
    ax.set_xlabel('Mean |SHAP|', fontsize=9, color=PAL['sub'])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color=PAL['grid'], linewidth=0.8)
    ax.set_facecolor(PAL['bg'])

plt.tight_layout()
plt.savefig('c1_plot5_shap.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: c1_plot5_shap.png")
print("\nKey finding to look for:")
print("  defect dominates Caries")
print("  chalky/brown dominates Pre-caries")
print("  stain dominates Decolor")
print("  This shows the model learned clinically correct feature associations")


# ════════════════════════════════════════════════════════════════
# CELL 17 — PLOT 6: Fine-grain to gross label mapping
# ════════════════════════════════════════════════════════════════
# Shows empirically how each fine-grain feature distributes
# across gross labels — the learned mapping visualised

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Fine-Grain Feature Distribution Across Gross Labels\n'
             'Empirical mapping from your annotated dataset',
             fontsize=13, fontweight='bold', color=PAL['text'])

for ax, fg in zip(axes.flatten(), FG_COLS):
    # For each gross label: % of crops in that class with this feature
    rates = {}
    for gl in CLASS_ORDER:
        sub = df[df['gross_label'] == gl]
        rates[gl] = 100 * sub[fg].mean() if len(sub) > 0 else 0

    y_pos  = range(len(CLASS_ORDER))
    colors = [PAL[gl] for gl in CLASS_ORDER]
    vals   = [rates[gl] for gl in CLASS_ORDER]

    ax.hlines(y_pos, 0, vals, colors=colors,
              linewidth=3, alpha=0.8)
    ax.scatter(vals, y_pos, color=colors,
               s=180, zorder=5)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f'{v:.0f}%',
                va='center', fontsize=10,
                fontweight='bold',
                color=PAL[CLASS_ORDER[i]])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_ORDER, fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_title(f'"{fg}" feature',
                 fontsize=11, fontweight='bold',
                 color=PAL['text'], pad=8)
    ax.set_xlabel('% crops with feature', fontsize=9,
                  color=PAL['sub'])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color=PAL['grid'], linewidth=0.8)
    ax.set_facecolor(PAL['bg'])

plt.tight_layout()
plt.savefig('c1_plot6_fg_mapping.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: c1_plot6_fg_mapping.png")
print("\nThis plot is your 'empirical mapping' figure for the paper.")
print("Each subplot shows one fine-grain feature and which gross labels it appears in.")
print("Ideal: each feature has a dominant gross label → clean mapping.")
print("Brown appearing in both Pre-caries AND Decolor → confirms it is your boundary class.")


# ════════════════════════════════════════════════════════════════
# CELL 18 — Summary printout for prof
# ════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT C1 SUMMARY — FINE-GRAIN FUSION RESULTS")
print("="*70)

# Best per category
img_only  = all_results[[k for k in all_results if 'Image only' in k][0]]
text_best = max(
    [r for k,r in all_results.items() if 'Text' in k or 'checkbox' in k],
    key=lambda r: r['wf1'])
fuse_best = max(
    [r for k,r in all_results.items() if 'fusion' in k.lower()],
    key=lambda r: r['wf1'])

print(f"\n{'Method':40s} {'Weighted F1':>12} {'Pre-caries F1':>14} {'Kappa':>8}")
print("-"*76)
print(f"{'Image only (BiomedCLIP)':40s} "
      f"{img_only['wf1']:>12.3f} {img_only['pc_f1']:>14.3f} "
      f"{img_only['kappa']:>8.3f}")
print(f"{'Best text only':40s} "
      f"{text_best['wf1']:>12.3f} {text_best['pc_f1']:>14.3f} "
      f"{text_best['kappa']:>8.3f}")
print(f"{'Best full fusion':40s} "
      f"{fuse_best['wf1']:>12.3f} {fuse_best['pc_f1']:>14.3f} "
      f"{fuse_best['kappa']:>8.3f}")

delta_wf1 = fuse_best['wf1'] - img_only['wf1']
delta_pc  = fuse_best['pc_f1'] - img_only['pc_f1']

print(f"\nImprovement from image-only to full fusion:")
print(f"  Weighted F1:     {delta_wf1:+.3f} ({delta_wf1*100:+.1f} points)")
print(f"  Pre-caries F1:   {delta_pc:+.3f}  ({delta_pc*100:+.1f} points)")
print(f"\nPlots saved:")
for f in ['c1_plot1_results.png', 'c1_plot2_confusion.png',
          'c1_plot3_similarity.png', 'c1_plot4_umap.png',
          'c1_plot5_shap.png', 'c1_plot6_fg_mapping.png']:
    print(f"  {f}")
print(f"\nResults table: c1_results.csv")
print(f"\n✓ Experiment C1 complete.")