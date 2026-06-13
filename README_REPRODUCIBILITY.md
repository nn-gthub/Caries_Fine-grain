# Reproducibility Guide
Fine-Grained Dental Lesion Classification

## Environment

Original environment:

- Python 3.10
- CUDA 12.x
- NVIDIA RTX A6000 (48 GB)
- Conda/venv environment: fusion_env_cuda12

Activate:

    source fusion_env_cuda12/bin/activate

------------------------------------------------------------

## Package Contents

scripts/
    Core experiment scripts

data/
    Annotation and metadata CSV files

weights/
    Trained model weights

results/
    Saved JSON outputs and benchmark results

------------------------------------------------------------

## Experiment Order

### 1. Main Cross-Validation Experiments

    python3 master_experiments.py

Outputs:
- img_only_clip.json
- fg_checkbox_only.json
- fg_text_pubmed.json
- fusion_clip_text.json
- full_fusion_xgb.json
- full_fusion_mlp.json
- pseudo_label_expansion.json

This produces the primary CV results reported in the thesis.

------------------------------------------------------------

### 2. BiomedCLIP Fine-Tuning

    python3 finetune_encoders.py

Outputs:
- probe_test_results.json
- ft_test_results.json

------------------------------------------------------------

### 3. Fine-Grain Feature Segmentation

    python3 seg_benchmark.py

Outputs:
- YOLOv8n / YOLOv8s models
- predicted feature files
- segmentation benchmark metrics

------------------------------------------------------------

### 4. LLaVA Clinical Note Generation

    python3 llava_generate.py

Output:
- llava_15_notes.csv

------------------------------------------------------------

### 5. Qwen2-VL Baseline

    python3 qwen_zeroshot.py

Output:
- qwen2vl_metrics.json

------------------------------------------------------------

### 6. End-to-End Fusion Pipeline

    python3 fusion_with_predictions.py

Outputs:
- fusion_test_results.json
- confusion matrices
- comparison plots

------------------------------------------------------------

### 7. Statistical Analysis

    python3 statistical_tests.py

Outputs:
- McNemar tests
- confidence intervals
- statistical_tests_report.txt

------------------------------------------------------------

## Quick Verification

To verify the main thesis results:

    python3 master_experiments.py
    python3 statistical_tests.py

Compare:

    img_only_clip.json
    full_fusion_xgb.json

Expected observation:

    Full fusion > Image-only baseline

------------------------------------------------------------

## Notes

- Ground-truth fine-grain experiments represent an upper-bound performance estimate.
- Fully automatic experiments use predicted fine-grain features and generated text.
- All major outputs required for verification are provided in the results folder.

Project Author:
Neena S Nair
M.Tech Biomedical Engineering, IIT Bombay
