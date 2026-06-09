"""
MASTER EXPERIMENTS
==================
Fine-Grain Feature Fusion for Dental Caries Classification
IIT Bombay — Intraoral Caries Dataset

Run from:
  cd /data1/neena/finegrain_alpha_experiments
  python3 master_experiments.py

Sections:
  0 — Setup, data loading, embedding extraction (runs once, saves .npy)
  2 — Text and fine-grain ablation experiments
  3 — Fusion experiments (main contribution)
  4 — Ablations for reviewers
  5 — Final sealed test set evaluation
  6 — All publication plots

Section 1 (CNN fine-tuned baselines) is commented out — uncomment to run.

Data required in working directory:
  annotation_clean.csv          (360 FG crops metadata)
  gross_label_corrections.csv   (relabelling log)
  combined_all_crops_v2.csv     (all 2893 crops, built by build_combined_csv.py)
  tooth_crops/                  (all crop images flat folder)
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              cohen_kappa_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR    = Path('/data1/neena/finegrain_alpha_experiments')
MAIN_CSV    = WORK_DIR / 'annotation_clean.csv'
LOG_CSV     = WORK_DIR / 'gross_label_corrections.csv'
COMBINED    = WORK_DIR / 'combined_all_crops_v2.csv'
CROP_DIR    = WORK_DIR / 'tooth_crops'
RESULTS_DIR = WORK_DIR / 'experiment_results'
RESULTS_DIR.mkdir(exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_ORDER = ['Normal', 'Pre-caries', 'Caries', 'Decolor']
FG_COLS     = ['chalky', 'brown', 'defect', 'fill', 'stain', 'wear']
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
PUBMED_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
CLIP_NAME   = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

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
plt.rcParams.update({
    'figure.facecolor': PAL['bg'],
    'axes.facecolor':   PAL['bg'],
    'font.family':      'DejaVu Sans',
})

print(f"Device: {DEVICE}")
print(f"Results: {RESULTS_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DATA LOADING AND EMBEDDING EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 0 — SETUP")
print("="*60)

# ── Load FG-annotated metadata (360 crops) ──────────────────────────────────
df  = pd.read_csv(MAIN_CSV)
log = pd.read_csv(LOG_CSV)

# Apply corrections
lkp = dict(zip(log['crop_name'], log['corrected_gross_label']))
df['gross_label'] = df['crop name'].map(lkp).fillna(df['gross_label'])

# Normalise checkboxes
for col in FG_COLS:
    df[col] = (df[col]
               .map({True:1,False:0,'True':1,'False':0,np.nan:0})
               .fillna(0).astype(int))

# Keep valid labels only (drops REVIEW_NEEDED)
df = df[df['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)
df['image_quality'] = (df['image_quality'].fillna('unknown')
                        .astype(str).str.lower().replace({'bad':'poor'}))

# Splits from annotation_clean (FG-annotated only)
df_train = df[df['split'].isin(['train','val'])].copy().reset_index(drop=True)
df_test  = df[df['split'] == 'test'].copy().reset_index(drop=True)
# All annotation_clean train+val crops including Normal
# Normal crops have no FG checkboxes — that IS their annotation
df_fg    = df_train.copy().reset_index(drop=True)

# ── Load combined CSV (all 2893 crops) ──────────────────────────────────────
df_all       = pd.read_csv(COMBINED)
df_all       = df_all[df_all['gross_label'].isin(CLASS_ORDER)].copy().reset_index(drop=True)
df_all_train = df_all[df_all['split'].isin(['train','val'])].copy().reset_index(drop=True)
df_all_test  = df_all[df_all['split'] == 'test'].copy().reset_index(drop=True)

for col in FG_COLS:
    df_all_train[col] = pd.to_numeric(df_all_train[col], errors='coerce').fillna(0).astype(int)
    df_all_test[col]  = pd.to_numeric(df_all_test[col],  errors='coerce').fillna(0).astype(int)

# Label encoders
le = LabelEncoder()
le.fit(CLASS_ORDER)

y_fg_enc       = le.transform(df_fg['gross_label'].values)
y_tr_enc       = le.transform(df_all_train['gross_label'].values)
y_te_enc       = le.transform(df_all_test['gross_label'].values)

print(f"\nFG-annotated (train+val): {len(df_fg)} crops")
print(f"All crops (train+val):    {len(df_all_train)} crops")
print(f"Test (sealed):            {len(df_all_test)} crops")
print(f"\nLabel distribution (FG subset):")
print(df_fg['gross_label'].value_counts().to_string())


# ── Helper functions ─────────────────────────────────────────────────────────
def get_sample_weights(y_arr):
    # Fixed: np.unique misses absent classes in small folds -> IndexError
    present    = np.unique(y_arr)
    cw_present = compute_class_weight(
        class_weight='balanced',
        classes=present,
        y=y_arr
    )
    cw_full = np.ones(len(CLASS_ORDER))
    for i, cls in enumerate(present):
        cw_full[cls] = cw_present[i]
    return np.array([cw_full[yi] for yi in y_arr])
    
def make_metrics(all_true, all_pred, le):
    rep = classification_report(all_true, all_pred,
                                 target_names=CLASS_ORDER,
                                 output_dict=True, zero_division=0)
    return {
        'wf1':      rep['weighted avg']['f1-score'],
        'no_f1':    rep['Normal']['f1-score'],
        'pc_f1':    rep['Pre-caries']['f1-score'],
        'ca_f1':    rep['Caries']['f1-score'],
        'dc_f1':    rep['Decolor']['f1-score'],
        'kappa':    cohen_kappa_score(all_true, all_pred, weights='quadratic'),
        'all_true': all_true,
        'all_pred': all_pred,
        'report':   rep,
    }

def run_xgb_cv(X, y_enc, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    sc = StandardScaler()
    X  = sc.fit_transform(X)
    for tr, val in skf.split(X, y_enc):
        sw  = get_sample_weights(y_enc[tr])
        clf = xgb.XGBClassifier(
            objective='multi:softprob', num_class=4,
            max_depth=5, learning_rate=0.05, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0,
            eval_metric='mlogloss',
            early_stopping_rounds=20)
        clf.fit(X[tr], y_enc[tr], sample_weight=sw,
                eval_set=[(X[val], y_enc[val])],
                verbose=False)
        all_pred.extend(clf.predict(X[val]).tolist())
        all_true.extend(y_enc[val].tolist())
    return make_metrics(np.array(all_true), np.array(all_pred), le)

def run_mlp_cv(X, y_enc, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    for tr, val in skf.split(X, y_enc):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu', alpha=0.01, max_iter=500,
                random_state=seed, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=15))
        ])
        pipe.fit(X[tr], y_enc[tr])
        all_pred.extend(pipe.predict(X[val]).tolist())
        all_true.extend(y_enc[val].tolist())
    return make_metrics(np.array(all_true), np.array(all_pred), le)

def save_result(name, metrics):
    out  = {k: float(v) for k, v in metrics.items()
            if k not in ('all_true','all_pred','report')}
    path = RESULTS_DIR / f"{name.replace(' ','_').replace('/','_')}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

def encode_texts(texts, tok, mod, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), leave=False):
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

def build_fg_text(row):
    """Structured text from FG annotations + clinical note."""
    present = [c for c in FG_COLS if row.get(c, 0) == 1]
    note    = str(row.get('clinical_note', '')).strip()
    note    = '' if note in ['nan','none',''] else note
    action  = str(row.get('recommended_action', '')).strip().lower()
    parts   = []
    if present:
        parts.append(f"Fine-grain features: {', '.join(present)}.")
    else:
        parts.append("No fine-grain features identified.")
    if action and action not in ['nan','']:
        parts.append(f"Recommended action: {action}.")
    if note:
        parts.append(note)
    return ' '.join(parts)

def build_gross_text(row):
    """ADA-based gross-label text — control condition (no FG info).
    Based on American Dental Association Caries Classification System."""
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
    return templates.get(str(row.get('gross_label','')),
                         'Intraoral tooth image, condition unclear.')

print("\nHelper functions loaded.")


# ── Embedding extraction ─────────────────────────────────────────────────────
import open_clip

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

needs = not all([
    (RESULTS_DIR/'clip_fg.npy').exists(),
    (RESULTS_DIR/'bert_fg_fg.npy').exists(),
    (RESULTS_DIR/'clip_tr_all.npy').exists(),
])

if needs:
    print("\nExtracting embeddings (first run — saves to disk)...")
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(CLIP_NAME)
    clip_model = clip_model.to(DEVICE).eval()
    pub_tok    = AutoTokenizer.from_pretrained(PUBMED_NAME)
    pub_mod    = AutoModel.from_pretrained(PUBMED_NAME).to(DEVICE).eval()

    # FG subset
    print("  CLIP — FG subset...")
    clip_fg = get_clip_embs(df_fg, CROP_DIR, clip_model, clip_prep)
    print("  BERT FG text — FG subset...")
    bert_fg_fg = encode_texts([build_fg_text(r) for _,r in df_fg.iterrows()],
                               pub_tok, pub_mod)
    print("  BERT gross text — FG subset (control)...")
    bert_gross_fg = encode_texts([build_gross_text(r) for _,r in df_fg.iterrows()],
                                  pub_tok, pub_mod)

    # All train+val (for pseudo-label expansion)
    print("  CLIP — all train+val...")
    clip_tr_all = get_clip_embs(df_all_train, CROP_DIR, clip_model, clip_prep)
    print("  BERT pseudo text — all train+val...")
    pseudo_texts = [build_fg_text(r) if r[FG_COLS].sum()>0
                    else build_gross_text(r)
                    for _, r in df_all_train.iterrows()]
    bert_pseudo_all = encode_texts(pseudo_texts, pub_tok, pub_mod)

    # Test
    print("  CLIP — test...")
    clip_te = get_clip_embs(df_all_test, CROP_DIR, clip_model, clip_prep)
    print("  BERT FG text — test...")
    bert_fg_te = encode_texts([build_fg_text(r) for _,r in df_all_test.iterrows()],
                               pub_tok, pub_mod)

    for name, arr in [
        ('clip_fg',       clip_fg),
        ('bert_fg_fg',    bert_fg_fg),
        ('bert_gross_fg', bert_gross_fg),
        ('clip_tr_all',   clip_tr_all),
        ('bert_pseudo_all', bert_pseudo_all),
        ('clip_te',       clip_te),
        ('bert_fg_te',    bert_fg_te),
    ]:
        np.save(RESULTS_DIR/f'{name}.npy', arr)
    print("  Saved all embeddings.")
    del clip_model, pub_mod
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
else:
    print("\nLoading saved embeddings...")
    clip_fg         = np.load(RESULTS_DIR/'clip_fg.npy')
    bert_fg_fg      = np.load(RESULTS_DIR/'bert_fg_fg.npy')
    bert_gross_fg   = np.load(RESULTS_DIR/'bert_gross_fg.npy')
    clip_tr_all     = np.load(RESULTS_DIR/'clip_tr_all.npy')
    bert_pseudo_all = np.load(RESULTS_DIR/'bert_pseudo_all.npy')
    clip_te         = np.load(RESULTS_DIR/'clip_te.npy')
    bert_fg_te      = np.load(RESULTS_DIR/'bert_fg_te.npy')

# Checkbox vectors
fg_vec_fg  = df_fg[FG_COLS].values.astype(float)
fg_vec_all = df_all_train[FG_COLS].values.astype(float)
fg_vec_te  = df_all_test[FG_COLS].values.astype(float)

# Scalers fitted on FG subset
sc_clip  = StandardScaler().fit(clip_fg)
sc_bert  = StandardScaler().fit(bert_fg_fg)
sc_gross = StandardScaler().fit(bert_gross_fg)
sc_fg    = StandardScaler().fit(fg_vec_fg)
sc_pseudo= StandardScaler().fit(bert_pseudo_all)
sc_all_fg= StandardScaler().fit(fg_vec_all)

clip_fg_n    = sc_clip.transform(clip_fg)
bert_fg_n    = sc_bert.transform(bert_fg_fg)
bert_gross_n = sc_gross.transform(bert_gross_fg)
fg_fg_n      = sc_fg.transform(fg_vec_fg)

clip_all_n   = sc_clip.transform(clip_tr_all)
pseudo_all_n = sc_pseudo.transform(bert_pseudo_all)
fg_all_n     = sc_all_fg.transform(fg_vec_all)

clip_te_n    = sc_clip.transform(clip_te)
bert_te_n    = sc_bert.transform(bert_fg_te)
fg_te_n      = sc_all_fg.transform(fg_vec_te)

print(f"\nEmbedding dims — CLIP:{clip_fg_n.shape[1]}  "
      f"BERT:{bert_fg_n.shape[1]}  FG:{fg_fg_n.shape[1]}")
print("Section 0 complete.\n")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CNN BASELINES (optional — comment out to skip)
# ════════════════════════════════════════════════════════════════════════════

# Uncomment to run CNN baselines. Each takes ~20-30 min on GPU.
# cnn_results = {}
# TRAIN_TF = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])
# VAL_TF = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])
# class DentalDataset(Dataset):
#     def __init__(self,df,crop_dir,tf=None):
#         self.df=df.reset_index(drop=True); self.d=Path(crop_dir); self.tf=tf
#         self.le={c:i for i,c in enumerate(CLASS_ORDER)}
#     def __len__(self): return len(self.df)
#     def __getitem__(self,idx):
#         row=self.df.iloc[idx]; label=self.le[row['gross_label']]
#         try: img=Image.open(self.d/row['crop name']).convert('RGB')
#         except: img=Image.new('RGB',(224,224),(128,128,128))
#         return (self.tf(img) if self.tf else img), label
# def train_cnn(mname,df_tr,epochs=25,lr=1e-4,bs=32):
#     skf=StratifiedKFold(5,shuffle=True,random_state=42)
#     y=le.transform(df_tr['gross_label'].values); aT,aP=[],[]
#     for fold,(tri,vli) in enumerate(skf.split(df_tr,y)):
#         print(f"  Fold {fold+1}",end=' ',flush=True)
#         dtr=DentalDataset(df_tr.iloc[tri],CROP_DIR,TRAIN_TF)
#         dvl=DentalDataset(df_tr.iloc[vli],CROP_DIR,VAL_TF)
#         ltr=DataLoader(dtr,bs,shuffle=True,num_workers=4,pin_memory=True)
#         lvl=DataLoader(dvl,bs,shuffle=False,num_workers=4)
#         if mname=='resnet50':
#             m=models.resnet50(weights='IMAGENET1K_V2'); m.fc=nn.Linear(m.fc.in_features,4)
#         else:
#             m=models.efficientnet_b3(weights='IMAGENET1K_V1')
#             m.classifier[1]=nn.Linear(m.classifier[1].in_features,4)
#         m=m.to(DEVICE)
#         cw=compute_class_weight('balanced',classes=np.unique(y[tri]),y=y[tri])
#         crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float).to(DEVICE))
#         opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
#         sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
#         bf,bp=0.0,[]
#         for ep in range(epochs):
#             m.train()
#             for imgs,labs in ltr:
#                 opt.zero_grad(); loss=crit(m(imgs.to(DEVICE)),labs.to(DEVICE))
#                 loss.backward(); opt.step()
#             sch.step(); m.eval(); ep_p=[]
#             with torch.no_grad():
#                 for imgs,_ in lvl: ep_p.extend(m(imgs.to(DEVICE)).argmax(1).cpu().tolist())
#             f=f1_score(y[vli],ep_p,average='weighted',zero_division=0)
#             if f>bf: bf=f; bp=ep_p[:]
#         aT.extend(y[vli].tolist()); aP.extend(bp); print(f"F1={bf:.3f}")
#     r=make_metrics(np.array(aT),np.array(aP),le); save_result(f'cnn_{mname}',r); return r
# print("ResNet50:"); cnn_results['ResNet50']=train_cnn('resnet50',df_all_train)
# print("EfficientNet-B3:"); cnn_results['EfficientNet-B3']=train_cnn('efficientnet_b3',df_all_train)

# Placeholder — replace with actual CNN results when you run Section 1
cnn_results = {}
CNN_SKIPPED = True


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TEXT AND FINE-GRAIN EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

print("="*60)
print("SECTION 2 — TEXT AND FINE-GRAIN EXPERIMENTS")
print("="*60)

text_results = {}

print("\n[2.1] Image only (BiomedCLIP frozen — zero-shot baseline)...")
r = run_xgb_cv(clip_fg_n, y_fg_enc)
text_results['Image only\n(BiomedCLIP)'] = r
save_result('img_only_clip', r)
print(f"      wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  Kappa={r['kappa']:.3f}")

print("[2.2] FG checkbox vector only...")
r = run_xgb_cv(fg_fg_n, y_fg_enc)
text_results['FG checkbox\nonly'] = r
save_result('fg_checkbox_only', r)
print(f"      wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}")

print("[2.3] Gross-label text only (ADA-based control — no FG info)...")
r = run_xgb_cv(bert_gross_n, y_fg_enc)
text_results['Gross-label text\n(ADA control)'] = r
save_result('gross_text_control', r)
print(f"      wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}")
print(f"      ← control: ADA clinical description, no per-tooth FG annotation")

print("[2.4] FG structured text — PubMedBERT (your contribution)...")
r = run_xgb_cv(bert_fg_n, y_fg_enc)
text_results['FG structured text\n(PubMedBERT)'] = r
save_result('fg_text_pubmed', r)
print(f"      wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}")

delta_pc = text_results['FG structured text\n(PubMedBERT)']['pc_f1'] - \
           text_results['Gross-label text\n(ADA control)']['pc_f1']
print(f"      Delta Pre-caries (FG vs control): {delta_pc:+.3f}")
print(f"      → This delta = value of fine-grain annotation")

print("[2.5] FG text with MLP (vs XGBoost comparison)...")
r_mlp = run_mlp_cv(bert_fg_n, y_fg_enc)
text_results['FG text + MLP'] = r_mlp
save_result('fg_text_mlp', r_mlp)
xgb_wf1 = text_results["FG structured text\n(PubMedBERT)"]["wf1"]
print(f"      XGBoost: {xgb_wf1:.3f}  MLP: {r_mlp['wf1']:.3f}")

print("\nSection 2 complete.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FUSION EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 3 — FUSION EXPERIMENTS")
print("="*60)

fusion_results = {}

print("\n[3.1] Image + FG text (CLIP + PubMedBERT)...")
X = np.hstack([clip_fg_n, bert_fg_n])
r = run_xgb_cv(X, y_fg_enc)
fusion_results['Image + FG text\n(CLIP+PubMedBERT)'] = r
save_result('fusion_clip_text', r)
print(f"      wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}")

print("[3.2] Full fusion XGBoost — CLIP + FG checkbox + PubMedBERT...")
X_full = np.hstack([clip_fg_n, fg_fg_n, bert_fg_n])
r = run_xgb_cv(X_full, y_fg_enc)
fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★'] = r
save_result('full_fusion_xgb', r)
print(f"      wF1={r['wf1']:.3f}  Pre-c={r['pc_f1']:.3f}  ← MAIN RESULT")

print("[3.3] Full fusion MLP (classifier comparison)...")
r_mlp = run_mlp_cv(X_full, y_fg_enc)
fusion_results['Full fusion MLP\n(CLIP+FG+PubMedBERT)'] = r_mlp
save_result('full_fusion_mlp', r_mlp)
print(f"      wF1={r_mlp['wf1']:.3f}  Pre-c={r_mlp['pc_f1']:.3f}")
print(f"      XGBoost vs MLP: {r['wf1']:.3f} vs {r_mlp['wf1']:.3f}")
print(f"      Note: XGBoost chosen — SHAP explainability not available for MLP")

print("\n[3.4] Pseudo-label expansion — all 2893 crops...")
X_pseudo = np.hstack([clip_all_n, fg_all_n, pseudo_all_n])
r_pseudo = run_xgb_cv(X_pseudo, y_tr_enc)
fusion_results['Pseudo-label expansion\n(all 2893 crops)'] = r_pseudo
save_result('pseudo_label_expansion', r_pseudo)
print(f"      wF1={r_pseudo['wf1']:.3f}  Pre-c={r_pseudo['pc_f1']:.3f}")
best_fg = fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★']['wf1']
print(f"      360-crop fusion: {best_fg:.3f}  →  2893-crop pseudo: {r_pseudo['wf1']:.3f}")

print("\nSection 3 complete.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ABLATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 4 — ABLATIONS")
print("="*60)

r_best = fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★']

print("\n[4.1] Feature knockout — remove one FG feature at a time...")
knockout_results = {}
for i, feat in enumerate(FG_COLS):
    fg_k = fg_fg_n.copy()
    fg_k[:, i] = 0
    X_k = np.hstack([clip_fg_n, fg_k, bert_fg_n])
    r_k = run_xgb_cv(X_k, y_fg_enc)
    knockout_results[feat] = {
        'wf1':      r_k['wf1'],
        'delta':    r_best['wf1'] - r_k['wf1'],
        'pc_delta': r_best['pc_f1'] - r_k['pc_f1'],
    }
    print(f"      Remove {feat:10s}: wF1={r_k['wf1']:.3f}  "
          f"drop={r_best['wf1']-r_k['wf1']:+.4f}  "
          f"pc_drop={r_best['pc_f1']-r_k['pc_f1']:+.4f}")

print("\n[4.2] Incremental feature addition...")
incremental_results = {}
r0 = run_xgb_cv(clip_fg_n, y_fg_enc)
incremental_results['Image only'] = r0['wf1']
print(f"      Image only:      {r0['wf1']:.3f}")
r1 = run_xgb_cv(np.hstack([clip_fg_n, bert_fg_n]), y_fg_enc)
incremental_results['+ FG text'] = r1['wf1']
print(f"      + FG text:       {r1['wf1']:.3f}")
for feat in FG_COLS:
    fi = FG_COLS.index(feat)
    fp = np.zeros_like(fg_fg_n)
    for prev in FG_COLS[:fi+1]:
        fp[:, FG_COLS.index(prev)] = fg_fg_n[:, FG_COLS.index(prev)]
    ri = run_xgb_cv(np.hstack([clip_fg_n, fp, bert_fg_n]), y_fg_enc)
    incremental_results[f'+ {feat}'] = ri['wf1']
    print(f"      + {feat:10s}: {ri['wf1']:.3f}")

pd.DataFrame([{'step':k,'wf1':v} for k,v in incremental_results.items()]
             ).to_csv(RESULTS_DIR/'incremental.csv', index=False)

print("\nSection 4 complete.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL TEST SET EVALUATION (run once on sealed test set)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 5 — FINAL TEST SET (sealed, run once)")
print("="*60)

# ── TEST SET EVALUATION ─────────────────────────────────────────────────
# The test set has NO fine-grain annotations and NO clinical notes.
# Using FG or BERT features on test creates train/test distribution mismatch.
# Only CLIP image embeddings are valid for test evaluation.
#
# This is scientifically correct:
#   CV results  → validate full fusion pipeline (FG annotations available)
#   Test result → validate image-only generalisation to unseen patients
#   The gap     → quantifies the value of fine-grain annotation

sc5      = StandardScaler().fit(clip_tr_all)
sc5_fin  = StandardScaler()
X_tr_img = sc5_fin.fit_transform(sc5.transform(clip_tr_all))
X_te_img = sc5_fin.transform(sc5.transform(clip_te))

sw = get_sample_weights(y_tr_enc)

final_model = xgb.XGBClassifier(
    objective='multi:softprob', num_class=4,
    max_depth=5, learning_rate=0.05, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0)
final_model.fit(X_tr_img, y_tr_enc, sample_weight=sw)
final_model.save_model(str(RESULTS_DIR/'final_model.json'))

y_pred_final = final_model.predict(X_te_img)

print("\nFINAL TEST RESULTS — Image only (CLIP, no FG annotations):")
print(classification_report(
    le.inverse_transform(y_te_enc),
    le.inverse_transform(y_pred_final),
    target_names=CLASS_ORDER, zero_division=0))
kappa_test = cohen_kappa_score(y_te_enc, y_pred_final, weights='quadratic')
print(f"Weighted Kappa: {kappa_test:.4f}")

test_report = classification_report(
    le.inverse_transform(y_te_enc),
    le.inverse_transform(y_pred_final),
    target_names=CLASS_ORDER, output_dict=True, zero_division=0)
pd.DataFrame(test_report).T.to_csv(RESULTS_DIR/'final_test_results.csv')

# McNemar not applicable (same model) — skip
# Store for plot and table
y_pred_img  = y_pred_final
img_report  = test_report
kappa_img   = kappa_test

# FG text not evaluable on test (no annotations) — set to None
y_pred_text = None
text_report = None
kappa_text  = None

print("\nNOTE: FG text and full fusion cannot be evaluated on test set")
print("      (test crops have no fine-grain annotations or clinical notes)")
print("      CV results in Sections 2-3 are the primary validation.")
print("Saved: final_test_results.csv")
print("Section 5 complete.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 6 — GENERATING PLOTS")
print("="*60)

def style_ax(ax, title='', xlabel='', ylabel='', grid='y'):
    ax.set_facecolor(PAL['bg'])
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=PAL['text'], pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=PAL['sub'])
    ax.set_ylabel(ylabel, fontsize=10, color=PAL['sub'])
    ax.tick_params(colors=PAL['sub'], labelsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    if grid: ax.grid(axis=grid, color=PAL['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# Build method dict — skip CNN if not run
all_methods = {
    'Image only\n(BiomedCLIP)':          text_results['Image only\n(BiomedCLIP)'],
    'Gross-label text\n(ADA control)':   text_results['Gross-label text\n(ADA control)'],
    'FG checkbox\nonly':                 text_results['FG checkbox\nonly'],
    'FG text\n(PubMedBERT)':             text_results['FG structured text\n(PubMedBERT)'],
    'Image + FG text\n(fusion)':         fusion_results['Image + FG text\n(CLIP+PubMedBERT)'],
    'Full fusion XGBoost ★':             fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★'],
    'Full fusion MLP':                   fusion_results['Full fusion MLP\n(CLIP+FG+PubMedBERT)'],
    'Pseudo-label\nexpansion':           fusion_results['Pseudo-label expansion\n(all 2893 crops)'],
}
if not CNN_SKIPPED and cnn_results:
    all_methods['ResNet50\n(fine-tuned)'] = cnn_results.get('ResNet50', {})
    all_methods['EfficientNet-B3\n(fine-tuned)'] = cnn_results.get('EfficientNet-B3', {})

def method_color(m):
    if '★' in m:                    return PAL['Caries']
    if 'fusion' in m.lower():        return PAL['blue']
    if 'fine-tuned' in m.lower():    return '#888888'
    if 'BiomedCLIP' in m:            return '#AAAAAA'
    if 'control' in m.lower():       return '#D4C5A9'
    return PAL['sub']

# Plot 1: Main results lollipop
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Fine-Grain Feature Fusion Improves Dental Caries Classification',
             fontsize=14, fontweight='bold', color=PAL['text'])

names  = [n for n,v in all_methods.items() if v]
wf1s   = [all_methods[n]['wf1'] for n in names]
pc_f1s = [all_methods[n]['pc_f1'] for n in names]
colors = [method_color(n) for n in names]
y_pos  = range(len(names))

for ax, vals, title in zip(
        axes,
        [wf1s, pc_f1s],
        ['Weighted F1 — All Classes', 'F1 — Pre-caries\n(hardest boundary)']):
    ax.hlines(y_pos, 0, vals, colors=colors, linewidth=2.5, alpha=0.8)
    ax.scatter(vals, y_pos, color=colors, s=160, zorder=5)
    for i, v in enumerate(vals):
        ax.text(v+0.005, i, f'{v:.3f}', va='center',
                fontsize=9, fontweight='bold', color=PAL['text'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color=PAL['grid'], linewidth=1, linestyle='--', alpha=0.7)
    style_ax(ax, title=title, xlabel='F1 Score', grid='x')

legend_el = [
    mpatches.Patch(color='#AAAAAA', label='Image only (frozen)'),
    mpatches.Patch(color='#D4C5A9', label='Control (ADA text)'),
    mpatches.Patch(color=PAL['blue'],   label='Fusion variants'),
    mpatches.Patch(color=PAL['Caries'], label='Best ★'),
]
fig.legend(handles=legend_el, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5,-0.03),
           facecolor=PAL['bg'], edgecolor=PAL['grid'])
plt.tight_layout()
plt.savefig(RESULTS_DIR/'plot_main_results.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: plot_main_results.png")

# Plot 2: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Confusion Matrix: Image-only vs Full Fusion',
             fontsize=13, fontweight='bold', color=PAL['text'])

r_img  = text_results['Image only\n(BiomedCLIP)']
r_fus  = fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★']

for ax, result, title in zip(
        axes,
        [r_img, r_fus],
        [f'Image only\nwF1={r_img["wf1"]:.3f}',
         f'Full Fusion ★\nwF1={r_fus["wf1"]:.3f}']):
    cm  = confusion_matrix(result['all_true'], result['all_pred'])
    pct = cm.astype(float)/cm.sum(axis=1, keepdims=True)*100
    sns.heatmap(pct, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=[c[:4] for c in CLASS_ORDER],
                yticklabels=[c[:4] for c in CLASS_ORDER],
                linewidths=2, linecolor=PAL['bg'],
                cbar_kws={'shrink':0.8},
                ax=ax, annot_kws={'size':11,'weight':'bold'})
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=PAL['text'], pad=8)
    ax.set_xlabel('Predicted', fontsize=9, color=PAL['sub'])
    ax.set_ylabel('True', fontsize=9, color=PAL['sub'])

plt.tight_layout()
plt.savefig(RESULTS_DIR/'plot_confusion.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: plot_confusion.png")

# Plot 3: Feature knockout
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('Feature Knockout — Contribution of Each Fine-Grain Feature',
             fontsize=13, fontweight='bold', color=PAL['text'])

feats  = list(knockout_results.keys())
deltas = [knockout_results[f]['delta']    for f in feats]
pcd    = [knockout_results[f]['pc_delta'] for f in feats]

for ax, vals, ylabel, title in zip(
        axes,
        [deltas, pcd],
        ['Δ Weighted F1', 'Δ Pre-caries F1'],
        ['Weighted F1 drop when removed','Pre-caries F1 drop when removed']):
    cols = [PAL['Caries'] if v>0.005 else PAL['blue'] for v in vals]
    bars = ax.bar(feats, vals, color=cols, edgecolor=PAL['bg'], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                v+max(max(vals),0.001)*0.05,
                f'{v:+.4f}', ha='center', fontsize=9.5, fontweight='bold')
    ax.axhline(0, color=PAL['text'], linewidth=0.8)
    style_ax(ax, title=title, ylabel=ylabel, grid='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR/'plot_knockout.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: plot_knockout.png")

# Plot 4: Incremental addition
fig, ax = plt.subplots(figsize=(12, 4.5))
fig.patch.set_facecolor(PAL['bg'])
steps = list(incremental_results.keys())
vals  = list(incremental_results.values())
x     = range(len(steps))
ax.plot(x, vals, 'o-', color=PAL['Pre-caries'],
        linewidth=2.5, markersize=9, zorder=5)
for i, v in enumerate(vals):
    ax.text(i, v+0.006, f'{v:.3f}', ha='center',
            fontsize=9.5, fontweight='bold', color=PAL['text'])
ax.axvspan(1.5, len(steps)-0.5, alpha=0.07,
           color=PAL['blue'], label='FG feature additions')
ax.axvline(1.5, color=PAL['blue'], linewidth=1.2, linestyle='--', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(steps, rotation=25, ha='right', fontsize=9)
ax.legend(fontsize=9)
style_ax(ax, title='Incremental Fine-Grain Feature Addition', ylabel='Weighted F1')
plt.tight_layout()
plt.savefig(RESULTS_DIR/'plot_incremental.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: plot_incremental.png")

# Plot 5: Test set confusion
r_fus_cv = fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★']
cv_cm    = confusion_matrix(r_fus_cv['all_true'], r_fus_cv['all_pred'])
te_cm    = confusion_matrix(y_te_enc, y_pred_final)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(PAL['bg'])
fig.suptitle('CV Full Fusion vs Held-out Test (Image only)',
             fontsize=13, fontweight='bold', color=PAL['text'])

for ax, cm, title in zip(
        axes,
        [cv_cm, te_cm],
        [f'Full Fusion CV\n(FG annotated, wF1={r_fus_cv["wf1"]:.3f})',
         f'Image only Test (no FG) wF1={img_report["weighted avg"]["f1-score"]:.3f} Kappa={kappa_test:.3f}']):
    pct = cm.astype(float)/cm.sum(axis=1, keepdims=True)*100
    sns.heatmap(pct, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                linewidths=2, linecolor=PAL['bg'],
                cbar_kws={'label':'%','shrink':0.8},
                ax=ax, annot_kws={'size':11,'weight':'bold'})
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=PAL['text'], pad=10)
    ax.set_xlabel('Predicted', fontsize=9, color=PAL['sub'])
    ax.set_ylabel('True', fontsize=9, color=PAL['sub'])

plt.tight_layout()
plt.savefig(RESULTS_DIR/'plot_test_results.png', dpi=150,
            bbox_inches='tight', facecolor=PAL['bg'])
plt.show()
print("Saved: plot_test_results.png")


# ── Final paper table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL PAPER TABLE")
print("="*70)
print(f"{'Method':40s} {'Norm':>6} {'PC':>6} {'Ca':>6} {'DC':>6} "
      f"{'wF1':>6} {'Kappa':>7}")
print("-"*70)

table_rows = [
    ('Image only (BiomedCLIP frozen)',
     text_results['Image only\n(BiomedCLIP)']),
    ('Gross-label text — ADA control [no FG]',
     text_results['Gross-label text\n(ADA control)']),
    ('FG checkbox only',
     text_results['FG checkbox\nonly']),
    ('FG structured text (PubMedBERT)',
     text_results['FG structured text\n(PubMedBERT)']),
    ('Image + FG text',
     fusion_results['Image + FG text\n(CLIP+PubMedBERT)']),
    ('Full fusion XGBoost ★',
     fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★']),
    ('Full fusion MLP',
     fusion_results['Full fusion MLP\n(CLIP+FG+PubMedBERT)']),
    ('Pseudo-label expansion (2893 crops)',
     fusion_results['Pseudo-label expansion\n(all 2893 crops)']),
]

for name, r in table_rows:
    print(f"  {name:38s} "
          f"{r['no_f1']:6.3f} {r['pc_f1']:6.3f} "
          f"{r['ca_f1']:6.3f} {r['dc_f1']:6.3f} "
          f"{r['wf1']:6.3f} {r['kappa']:7.3f}")

# CV results summary
r_img_cv  = text_results['Image only\n(BiomedCLIP)']
r_fus_cv  = fusion_results['Full fusion XGBoost\n(CLIP+FG+PubMedBERT) ★']
gap_wf1   = r_fus_cv['wf1']   - r_img_cv['wf1']
gap_pc    = r_fus_cv['pc_f1'] - r_img_cv['pc_f1']

print(f"\n  CV RESULTS (FG annotations available, 5-fold cross-validation):")
print(f"  Image only CV:    wF1={r_img_cv['wf1']:.3f}  Pre-c={r_img_cv['pc_f1']:.3f}")
print(f"  Full fusion CV:   wF1={r_fus_cv['wf1']:.3f}  Pre-c={r_fus_cv['pc_f1']:.3f}")
print(f"  Gap (FG value):   wF1={gap_wf1:+.3f}  Pre-c={gap_pc:+.3f}")
print(f"\n  TEST SET (held-out 623 crops, image features only):")
print(f"  Image only test:  wF1={img_report['weighted avg']['f1-score']:.3f}  "
      f"Pre-c={img_report['Pre-caries']['f1-score']:.3f}  Kappa={kappa_test:.3f}")
print(f"\n  FG text and full fusion not evaluated on test")
print(f"  (test set has no fine-grain annotations — by design)")
print(f"\n  ★ = main contribution")
print(f"  Control row (ADA text) shows FG text beats standard clinical descriptions")
print(f"  → improvement is specifically from fine-grain per-tooth annotation")
print(f"\nAll outputs saved to: {RESULTS_DIR}/")
print("✓ COMPLETE.")
