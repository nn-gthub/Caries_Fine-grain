# Fine-Grain Multimodal Fusion for Dental Caries Classification

MTech Biomedical Engineering, IIT Bombay — Stage 2 Major Technical Project

## Overview
Two-stage pipeline combining BiomedCLIP image embeddings, PubMedBERT clinical
text embeddings, and clinician-annotated fine-grain morphological features for
four-class dental caries classification (Normal, Pre-caries, Caries, Decolor).

## Key Results
- Full fusion: wF1 = 0.958 (Cohen's kappa = 0.946, Almost Perfect)
- Image-only baseline: wF1 = 0.504
- Annotation value: +0.454 wF1
- vs Qwen2-VL 7B zero-shot (0.375): +0.583 wF1

## Main Scripts
- master_experiments.py — cross-validation experiments
- finetune_encoders.py — BiomedCLIP fine-tuning
- seg_benchmark.py — YOLOv8 segmentation
- fusion_with_predictions.py — test-set end-to-end pipeline
- statistical_tests.py — significance testing
- clinical_analysis.py — caries spectrum, UMAP, PCA

Note: Model weights, embeddings, and the image dataset are excluded (see .gitignore).
