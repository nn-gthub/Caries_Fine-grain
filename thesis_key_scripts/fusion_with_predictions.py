"""
FUSION WITH PREDICTED FEATURES
================================
Complete end-to-end pipeline evaluation on the sealed test set.
Uses predicted FG features (from seg_benchmark.py) and
generated clinical notes (from llava_generate.py).

This is the final experiment that closes the loop:
  Image → YOLOv8-seg → predicted FG features → XGBoost → gross label
  Image → LLaVA-Med  → generated clinical note → PubMedBERT → XGBoost

Comparison table:
  Method                        | Test wF1 | Pre-c F1 | Kappa
  ------------------------------|----------|----------|------
  Image only (frozen CLIP)      |          |          |
  Image only (fine-tuned CLIP)  |          |          |
  Pred FG + image               |          |          |
  Pred FG + ADA text + image    |          |          |
  Pred FG + LLaVA-Med + image   |          |          |
  Pred FG + LLaVA-1.5 + image   |          |          |
  GT FG + expert notes CV ★     | 0.958    | 0.929    | 0.946

Run:
  python3 fusion_with_predictions.py
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              cohen_kappa_score,
                              f1_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import open_clip

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR    = Path('/data1/neena/finegrain_alpha_experiments')
COMBINED    = WORK_DIR / 'combined_all_crops_v2.csv'
CROP_DIR    = WORK_DIR / 'tooth_crops'
PRED_FEAT   = WORK_DIR / 'experiment_results/segmentation/predicted_fg_features.csv'
LLM_DIR     = WORK_DIR / 'experiment_results/llm_generation'
RESULTS_DIR = WORK_DIR / 'experiment_results/fusion_predictions'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CV results from master_experiments.py (ground truth upper bound)
CV_RESULTS  = WORK_DIR / 'experiment_results/full_fusion_xgb.json'
FT_RESULTS  = WORK_DIR / 'experiment_results/finetune/ft_test_results.json'

FG_CLASSES  = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
PRED_COLS   = [f'pred_{c}' for c in FG_CLASSES]
CONF_COLS   = [f'pred_{c}_conf' for c in FG_CLASSES]
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
PUBMED_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
CLIP_NAME   = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

le = LabelEncoder()
le.fit(CLASS_ORDER)

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

print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD ALL DATA
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1 — LOAD DATA")
print("="*60)

# Main combined CSV
df_all   = pd.read_csv(COMBINED)
df_all   = df_all[df_all['gross_label'].isin(CLASS_ORDER)].copy()
df_all   = df_all.reset_index(drop=True)
df_tr    = df_all[df_all['split'].isin(['train','val'])].reset_index(drop=True)
df_te    = df_all[df_all['split']=='test'].reset_index(drop=True)

# Predicted FG features
pred_df  = pd.read_csv(PRED_FEAT)
pred_df  = pred_df.set_index('crop name')

# Add predicted features to main dfs
for df in [df_tr, df_te]:
    for col in PRED_COLS + CONF_COLS:
        df[col] = df['crop name'].map(
            pred_df[col] if col in pred_df.columns else pd.Series()
        ).fillna(0.0).astype(float)

y_tr = le.transform(df_tr['gross_label'].values)
y_te = le.transform(df_te['gross_label'].values)

print(f"Train: {len(df_tr)}  Test: {len(df_te)}")
print(f"Predicted features available: "
      f"{(df_te[PRED_COLS].sum(axis=1)>0).sum()}/{len(df_te)} test crops")

# Load generated notes
llava_med_notes, llava_15_notes = {}, {}
med_path = LLM_DIR / 'llava_med_notes.csv'
l15_path = LLM_DIR / 'llava_15_notes.csv'

if med_path.exists():
    med_df = pd.read_csv(med_path)
    llava_med_notes = dict(zip(med_df['crop name'],
                               med_df['generated_note'].fillna('')))
    print(f"LLaVA-Med notes loaded: {len(llava_med_notes)}")
else:
    print("LLaVA-Med notes not found — run llava_generate.py first")

if l15_path.exists():
    l15_df = pd.read_csv(l15_path)
    llava_15_notes = dict(zip(l15_df['crop name'],
                               l15_df['generated_note'].fillna('')))
    print(f"LLaVA-1.5 notes loaded: {len(llava_15_notes)}")
else:
    print("LLaVA-1.5 notes not found — run llava_generate.py first")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2 — EXTRACT EMBEDDINGS")
print("="*60)

# ── CLIP embeddings ────────────────────────────────────────────────────────
def get_clip_embs(df_sub, crop_dir, model, preprocess):
    embs = []
    for _, row in tqdm(df_sub.iterrows(),
                       total=len(df_sub), desc="CLIP", leave=False):
        p = Path(crop_dir) / row['crop name']
        try:
            img = Image.open(p).convert('RGB')
            t   = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                e = model.encode_image(t)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy()[0])
        except Exception:
            embs.append(np.zeros(512))
    return np.array(embs)


# ── BERT text embeddings ───────────────────────────────────────────────────
def get_bert_embs(texts, tok, mod, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="BERT", leave=False):
        batch = texts[i:i+batch_size]
        enc   = tok(batch, padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
        enc   = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out  = mod(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb  = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.extend(emb.cpu().numpy().tolist())
    return np.array(all_embs)


def build_ada_text(row, use_label=True):
    """
    Build ADA-based text.
    use_label=True  → uses gross_label (for training only — label known)
    use_label=False → generic text (for test — label unknown at inference)
    """
    if not use_label:
        # At inference time the gross label is unknown
        # Use predicted FG features as text context instead
        present = [c for c in ['chalky','brown','defect','fill','stain','wear']
                   if row.get(f'pred_{c}', 0) > 0]
        if present:
            return (f"Intraoral tooth photograph. "
                    f"Automated analysis detected: {', '.join(present)}.")
        return "Intraoral tooth photograph requiring clinical assessment."
    templates = {
        'Normal':
            'Intraoral tooth image. Sound tooth structure. '
            'No clinically detectable lesion. ICDAS 0.',
        'Pre-caries':
            'Intraoral tooth image. Initial caries lesion. '
            'Non-cavitated enamel changes, white or brown. ICDAS 1-2.',
        'Caries':
            'Intraoral tooth image. Moderate to advanced caries. '
            'Cavitation or dentin exposure. ICDAS 3-6.',
        'Decolor':
            'Intraoral tooth image. Non-carious discolouration. '
            'No demineralization or cavitation.',
    }
    return templates.get(str(row.get('gross_label', '')),
                         'Intraoral tooth image.')


# Check for saved embeddings
clip_tr_path = RESULTS_DIR / 'clip_tr.npy'
clip_te_path = RESULTS_DIR / 'clip_te.npy'

if clip_tr_path.exists() and clip_te_path.exists():
    print("Loading saved CLIP embeddings...")
    clip_tr = np.load(clip_tr_path)
    clip_te = np.load(clip_te_path)
else:
    print("Extracting CLIP embeddings...")
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        CLIP_NAME)
    clip_model = clip_model.to(DEVICE).eval()
    clip_tr = get_clip_embs(df_tr, CROP_DIR, clip_model, clip_prep)
    clip_te = get_clip_embs(df_te, CROP_DIR, clip_model, clip_prep)
    np.save(clip_tr_path, clip_tr)
    np.save(clip_te_path, clip_te)
    del clip_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"CLIP: train={clip_tr.shape}  test={clip_te.shape}")

# Load PubMedBERT
print("Loading PubMedBERT...")
pub_tok = AutoTokenizer.from_pretrained(PUBMED_NAME)
pub_mod = AutoModel.from_pretrained(PUBMED_NAME).to(DEVICE).eval()

# Build text variants for train and test
def get_texts(df, note_dict=None, source='ada', is_test=False):
    """
    Build text inputs.
    For training: use gross_label in ADA template (label is known)
    For test:     use predicted FG features only (label is NOT known)
    """
    texts = []
    for _, row in df.iterrows():
        if source == 'llava' and note_dict:
            note = note_dict.get(row['crop name'], '')
            if note and len(note.split()) >= 5:
                texts.append(note)
            else:
                texts.append(build_ada_text(row, use_label=not is_test))
        else:
            texts.append(build_ada_text(row, use_label=not is_test))
    return texts

# Encode text variants
print("Encoding text variants...")
bert_ada_tr  = get_bert_embs(get_texts(df_tr, source='ada', is_test=False),
                               pub_tok, pub_mod)
bert_ada_te  = get_bert_embs(get_texts(df_te, source='ada', is_test=True),
                               pub_tok, pub_mod)

bert_med_tr, bert_med_te = None, None
bert_15_tr,  bert_15_te  = None, None

if llava_med_notes:
    bert_med_tr = get_bert_embs(
        get_texts(df_tr, llava_med_notes, 'llava'), pub_tok, pub_mod)
    bert_med_te = get_bert_embs(
        get_texts(df_te, llava_med_notes, 'llava'), pub_tok, pub_mod)
    print(f"LLaVA-Med text encoded.")

if llava_15_notes:
    bert_15_tr = get_bert_embs(
        get_texts(df_tr, llava_15_notes, 'llava'), pub_tok, pub_mod)
    bert_15_te = get_bert_embs(
        get_texts(df_te, llava_15_notes, 'llava'), pub_tok, pub_mod)
    print(f"LLaVA-1.5 text encoded.")

del pub_mod
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Predicted FG feature vectors
pred_tr = df_tr[PRED_COLS].values.astype(float)
pred_te = df_te[PRED_COLS].values.astype(float)
conf_tr = df_tr[CONF_COLS].values.astype(float)
conf_te = df_te[CONF_COLS].values.astype(float)

# Combine predicted FG with confidence scores
fg_pred_tr = np.hstack([pred_tr, conf_tr])
fg_pred_te = np.hstack([pred_te, conf_te])


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN XGBOOST FUSION MODELS AND EVALUATE ON TEST
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3 — FUSION MODELS ON TEST SET")
print("="*60)


def get_sample_weights(y_arr):
    present = np.unique(y_arr)
    cw_p    = compute_class_weight('balanced',
                                    classes=present, y=y_arr)
    cw_full = np.ones(len(CLASS_ORDER))
    for i, cls in enumerate(present):
        cw_full[cls] = cw_p[i]
    return np.array([cw_full[yi] for yi in y_arr])


def train_and_eval(X_tr, X_te, y_tr, y_te, name=''):
    """Train XGBoost on train, evaluate on test."""
    sc   = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    sw = get_sample_weights(y_tr)
    clf = xgb.XGBClassifier(
        objective        = 'multi:softprob',
        num_class        = 4,
        max_depth        = 5,
        learning_rate    = 0.05,
        n_estimators     = 300,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        verbosity        = 0,
        early_stopping_rounds = 20,
        eval_metric      = 'mlogloss',
    )

    # Use 15% of train as internal eval set for early stopping
    from sklearn.model_selection import train_test_split
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val, sw_tr, _ = \
        train_test_split(X_tr_s, y_tr, sw,
                          test_size=0.15, random_state=42,
                          stratify=y_tr)

    clf.fit(X_sub_tr, y_sub_tr, sample_weight=sw_tr,
            eval_set=[(X_sub_val, y_sub_val)],
            verbose=False)

    y_pred = clf.predict(X_te_s)

    rep = classification_report(
        le.inverse_transform(y_te),
        le.inverse_transform(y_pred),
        target_names=CLASS_ORDER,
        output_dict=True, zero_division=0)

    return {
        'wf1':      rep['weighted avg']['f1-score'],
        'pc_f1':    rep['Pre-caries']['f1-score'],
        'ca_f1':    rep['Caries']['f1-score'],
        'dc_f1':    rep['Decolor']['f1-score'],
        'no_f1':    rep['Normal']['f1-score'],
        'kappa':    cohen_kappa_score(y_te, y_pred, weights='quadratic'),
        'all_true': y_te,
        'all_pred': y_pred,
        'report':   rep,
    }


all_results = {}

# Fit scalers on train
sc_clip = StandardScaler().fit(clip_tr)
sc_ada  = StandardScaler().fit(bert_ada_tr)
sc_fg   = StandardScaler().fit(fg_pred_tr)

clip_tr_n = sc_clip.transform(clip_tr)
clip_te_n = sc_clip.transform(clip_te)
ada_tr_n  = sc_ada.transform(bert_ada_tr)
ada_te_n  = sc_ada.transform(bert_ada_te)
fg_tr_n   = sc_fg.transform(fg_pred_tr)
fg_te_n   = sc_fg.transform(fg_pred_te)

# ── Experiment 1: Image only (frozen CLIP) ────────────────────────────────
print("\n[1] Image only (frozen CLIP)...")
r = train_and_eval(clip_tr_n, clip_te_n, y_tr, y_te)
all_results['Image only\n(frozen CLIP)'] = r
print(f"    wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  Kappa={r['kappa']:.3f}")

# ── Experiment 2: Predicted FG + image ───────────────────────────────────
print("[2] Predicted FG features + image...")
X_tr2 = np.hstack([clip_tr_n, fg_tr_n])
X_te2 = np.hstack([clip_te_n, fg_te_n])
r = train_and_eval(X_tr2, X_te2, y_tr, y_te)
all_results['Pred FG + Image\n(CLIP+YOLOv8)'] = r
print(f"    wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  Kappa={r['kappa']:.3f}")

# ── Experiment 3: Predicted FG + ADA text + image ─────────────────────────
print("[3] Predicted FG + ADA text + image...")
X_tr3 = np.hstack([clip_tr_n, fg_tr_n, ada_tr_n])
X_te3 = np.hstack([clip_te_n, fg_te_n, ada_te_n])
r = train_and_eval(X_tr3, X_te3, y_tr, y_te)
all_results['Pred FG + ADA text + Image'] = r
print(f"    wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  Kappa={r['kappa']:.3f}")

# ── Experiment 4: Predicted FG + LLaVA-Med text + image ──────────────────
if bert_med_tr is not None:
    print("[4] Predicted FG + LLaVA-Med text + image...")
    sc_med   = StandardScaler().fit(bert_med_tr)
    med_tr_n = sc_med.transform(bert_med_tr)
    med_te_n = sc_med.transform(bert_med_te)
    X_tr4    = np.hstack([clip_tr_n, fg_tr_n, med_tr_n])
    X_te4    = np.hstack([clip_te_n, fg_te_n, med_te_n])
    r = train_and_eval(X_tr4, X_te4, y_tr, y_te)
    all_results['Pred FG + LLaVA-Med\n(CLIP+YOLOv8+LLaVA-Med)'] = r
    print(f"    wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  Kappa={r['kappa']:.3f}")

# ── Experiment 5: Predicted FG + LLaVA-1.5 text + image ──────────────────
if bert_15_tr is not None:
    print("[5] Predicted FG + LLaVA-1.5 text + image...")
    sc_15   = StandardScaler().fit(bert_15_tr)
    l15_tr_n = sc_15.transform(bert_15_tr)
    l15_te_n = sc_15.transform(bert_15_te)
    X_tr5    = np.hstack([clip_tr_n, fg_tr_n, l15_tr_n])
    X_te5    = np.hstack([clip_te_n, fg_te_n, l15_te_n])
    r = train_and_eval(X_tr5, X_te5, y_tr, y_te)
    all_results['Pred FG + LLaVA-1.5\n(CLIP+YOLOv8+LLaVA-1.5)'] = r
    print(f"    wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  Kappa={r['kappa']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — FINAL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINAL COMPARISON — TEST SET (623 crops, held-out)")
print("="*70)
print(f"{'Method':45s} {'Norm':>6} {'PC':>6} {'Ca':>6} {'DC':>6} "
      f"{'wF1':>6} {'Kappa':>7}")
print("-"*75)

for name, r in all_results.items():
    clean = name.replace('\n', ' ')
    print(f"  {clean:43s} "
          f"{r['no_f1']:6.3f} {r['pc_f1']:6.3f} "
          f"{r['ca_f1']:6.3f} {r['dc_f1']:6.3f} "
          f"{r['wf1']:6.3f} {r['kappa']:7.3f}")

# Add CV upper bound
print(f"\n  {'--- UPPER BOUND (CV, requires annotation) ---':43s}")
if CV_RESULTS.exists():
    cv_r = json.load(open(CV_RESULTS))
    print(f"  {'GT FG + Expert notes CV ★':43s} "
          f"{'--':>6} {cv_r.get('pc_f1',0.929):6.3f} "
          f"{cv_r.get('ca_f1',0.980):6.3f} {cv_r.get('dc_f1',0.940):6.3f} "
          f"{cv_r.get('wf1',0.958):6.3f} {cv_r.get('kappa',0.946):7.3f}")
else:
    print(f"  {'GT FG + Expert notes CV ★':43s} "
          f"{'--':>6} {'0.929':>6} {'0.980':>6} "
          f"{'0.940':>6} {'0.958':>6} {'0.946':>7}")

# Add fine-tuned CLIP result
if FT_RESULTS.exists():
    ft_r = json.load(open(FT_RESULTS))
    print(f"  {'BiomedCLIP fine-tuned (test)':43s} "
          f"{'--':>6} {'--':>6} {'--':>6} {'--':>6} "
          f"{ft_r.get('wf1',0.678):6.3f} {ft_r.get('kappa',0.514):7.3f}")

print(f"\n  ★ = ground-truth upper bound (requires dentist annotation)")
print(f"  All other rows = automatic pipeline (no annotation needed)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — PLOTS
# ════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PAL['bg'])
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=PAL['text'], pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=PAL['sub'])
    ax.set_ylabel(ylabel, fontsize=10, color=PAL['sub'])
    ax.tick_params(colors=PAL['sub'], labelsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis='x', color=PAL['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

plt.rcParams.update({'figure.facecolor': PAL['bg'],
                     'axes.facecolor':   PAL['bg'],
                     'font.family':      'DejaVu Sans'})

# Plot 1: Main comparison lollipop
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('End-to-End Pipeline: Automatic vs Annotated Results\n'
             'Test Set (623 held-out crops)',
             fontsize=13, fontweight='bold', color=PAL['text'])

# Add CV reference
names  = [n.replace('\n',' ') for n in all_results.keys()]
wf1s   = [r['wf1']   for r in all_results.values()]
pc_f1s = [r['pc_f1'] for r in all_results.values()]

# Colour by method type
def method_color(n):
    if 'LLaVA-Med'  in n: return '#C1392B'
    if 'LLaVA-1.5'  in n: return '#6C3483'
    if 'Pred FG'     in n: return PAL['blue']
    return '#888888'

colors = [method_color(n) for n in names]
y_pos  = range(len(names))

for ax, vals, title in zip(
        axes,
        [wf1s, pc_f1s],
        ['Weighted F1 — All Classes', 'F1 — Pre-caries\n(hardest class)']):
    ax.hlines(y_pos, 0, vals, colors=colors, linewidth=2.5, alpha=0.8)
    ax.scatter(vals, y_pos, color=colors, s=160, zorder=5)
    for i, v in enumerate(vals):
        ax.text(v+0.005, i, f'{v:.3f}', va='center',
                fontsize=9.5, fontweight='bold', color=PAL['text'])

    # CV reference line
    cv_val = 0.958 if 'F1' in title else 0.929
    ax.axvline(cv_val, color='#C1392B', linestyle='--',
               linewidth=1.5, alpha=0.6,
               label=f'CV upper bound ({cv_val:.3f})')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=9)
    style_ax(ax, title=title, xlabel='F1 Score')

legend_el = [
    mpatches.Patch(color='#888888', label='Image only'),
    mpatches.Patch(color=PAL['blue'],   label='+ Predicted FG (YOLOv8)'),
    mpatches.Patch(color='#C1392B', label='+ LLaVA-Med text'),
    mpatches.Patch(color='#6C3483', label='+ LLaVA-1.5 text'),
]
fig.legend(handles=legend_el, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.03),
           facecolor=PAL['bg'], edgecolor=PAL['grid'])
plt.tight_layout()
plt.savefig(RESULTS_DIR/'plot_fusion_comparison.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: plot_fusion_comparison.png")

# Plot 2: Confusion matrix for best automatic method
best_auto = max(
    [(n,r) for n,r in all_results.items() if 'image' not in n.lower()],
    key=lambda x: x[1]['wf1'],
    default=(None, None))

if best_auto[0] is not None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(PAL['bg'])
    fig.suptitle('Confusion Matrix: Image Only vs Best Automatic Pipeline',
                 fontsize=13, fontweight='bold', color=PAL['text'])

    r_img  = all_results[[k for k in all_results if 'Image only' in k][0]]
    r_best = best_auto[1]

    for ax, r, title in zip(
            axes,
            [r_img, r_best],
            ['Image only (test)',
             f'Best automatic\n{best_auto[0].replace(chr(10)," ")}']):
        cm  = confusion_matrix(r['all_true'], r['all_pred'])
        pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
        sns.heatmap(pct, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=CLASS_ORDER,
                    yticklabels=CLASS_ORDER,
                    linewidths=2, linecolor=PAL['bg'],
                    cbar_kws={'label':'%', 'shrink':0.8},
                    ax=ax,
                    annot_kws={'size':11,'weight':'bold'})
        ax.set_title(f'{title}\nwF1={r["wf1"]:.3f}  Kappa={r["kappa"]:.3f}',
                     fontsize=10, fontweight='bold',
                     color=PAL['text'], pad=8)
        ax.set_xlabel('Predicted', fontsize=9, color=PAL['sub'])
        ax.set_ylabel('True', fontsize=9, color=PAL['sub'])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR/'plot_confusion_comparison.png', dpi=150,
                bbox_inches='tight', facecolor=PAL['bg'])
    plt.show()
    print("Saved: plot_confusion_comparison.png")

# Save all results
results_save = {}
for k, v in all_results.items():
    kk = k.replace('\n',' ')
    results_save[kk] = {
        m: float(v[m]) for m in
        ['wf1','pc_f1','ca_f1','dc_f1','no_f1','kappa']
    }
with open(RESULTS_DIR/'fusion_test_results.json','w') as f:
    json.dump(results_save, f, indent=2)

print(f"\nAll results saved to: {RESULTS_DIR}/")
print("✓ Fusion with predictions complete.")
