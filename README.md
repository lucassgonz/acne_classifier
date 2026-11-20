# Region-Aware ResNet-18 for Facial Acne Severity Classification

Automatic acne severity grading (Hayashi scale, levels 0–3) from facial images, combining landmark-based anatomical segmentation with a ResNet-18 that incorporates learnable region embeddings.

**Regions**: forehead, nose, chin, left cheek, right cheek
**Classes**: 0 = mild, 1 = moderate, 2 = severe, 3 = very severe

---

## Background and related work

One of the references for this project was *"AcneNet: A Deep CNN Based Classification Approach for Acne Classes"* ([IEEE Xplore](https://ieeexplore.ieee.org/document/8850935)), which reports accuracy above 94% for acne class classification. That figure served as an initial performance expectation, but it is not directly comparable to the problem addressed here: grading scales, evaluation protocols and datasets differ. "AcneNet" is the name of that reference work, not of this project.

The results reported here were obtained under a deliberately strict protocol (splits verified for image leakage between train/val/test, a controlled ablation, and cross-dataset generalization), which explains why the numbers are more conservative than those of works that do not report such checks.

---

## Architecture

```
Facial region crop (224×224) + region index
     │
     ▼
ResNet-18 backbone (ImageNet → SCIN → fine-tuning)  →  feature vector (512-d)
     │
     ▼
Learnable region embedding (5×64)  →  concatenated with the feature vector
     │
     ▼
MLP classifier (Dropout + BatchNorm)  →  logits (4 classes)
```

The final model combines Stochastic Weight Averaging (SWA), an ensemble of 5 independent seeds and test-time augmentation (7 views). The `ResNet18WithEmbed` and `ResNet18NoEmbed` (used in the ablation) classes are defined in [`training/run_ablation.py`](training/run_ablation.py).

Two complementary architectures were also evaluated:
- **Whole-image baseline** ([`training/run_wholeimage.py`](training/run_wholeimage.py)): classifies the entire image, without region crops.
- **Multi-region fusion** ([`training/run_multiregion_fusion.py`](training/run_multiregion_fusion.py)): fuses all available regions of the same image into a single prediction, keeping the anatomical embedding.

---

## Repository structure

```
training/
  run_ablation.py                       Main model (WithEmbed/NoEmbed) + training + ablation
  run_wholeimage.py                     Whole-image baseline
  run_multiregion_fusion.py             Multi-region fusion
  pretrain_backbone.py                  Backbone pre-training (ImageNet → SCIN)
  evaluate_ensemble_tta.py              Ensemble + TTA evaluation (region model)
  evaluate_wholeimage_ensemble_tta.py   Ensemble + TTA evaluation (whole image)
  evaluate_multiregion_ensemble_tta.py  Ensemble + TTA evaluation (multi-region fusion)
  evaluate_by_region.py                 Per-region performance
  predict.py                            Command-line inference (one checkpoint at a time)
  model.py                              Legacy architecture (not used by the current pipeline; kept for history)
  models/                               Backbones and final checkpoints (not versioned; see below)
  results/                              Results of each experiment (JSON)
  utils/
    segment_faces_mediapipe.py          Facial segmentation (MediaPipe Face Mesh, 478 landmarks)
    prepare_acne04_split.py             ACNE04 train/val/test split (70/15/15)
    merge_manual_data.py                Merges the annotated collection with ACNE04 (leakage-safe protocol)
    consolidate_new_sources.py          Consolidation of additional public sources
    prepare_scin_pretrain.py            SCIN preparation for backbone pre-training
    prepare_wholeimage_data.py          Whole-image dataset preparation
segment/
  yolo_train_and_segment_v2.ipynb       Legacy YOLOv8 segmentation (historical; replaced by MediaPipe)
acne_classifier_app/                    Flutter app (mobile prototype)
acne_classifier_android/                Native Android app in Kotlin (Jetpack Compose)
mobile_models/                          ONNX ensemble (5 seeds) used by both apps
```

> The dataset (ACNE04 + the collection annotated by the authors) is not included in the repository because of its size and, for the authors' collection, because it is not a redistributable public dataset. See "Data" below.

---

## Data

The model is trained on the combination of two image sources:

| Source | Description | Link |
|---|---|---|
| ACNE04 | Public dataset, original labels on the Hayashi scale | [Kaggle](https://www.kaggle.com/datasets/manuelhettich/acne04) |
| Collection annotated by the authors | Images from multiple public sources, manually annotated and reviewed by the authors on the Hayashi scale | not directly redistributable; see the note below |

Each image is segmented into up to 5 anatomical crops with MediaPipe Face Mesh (`training/utils/segment_faces_mediapipe.py`). The train/val/test split is made **before** segmentation, at the level of the original image, so all crops of the same photo stay in the same split.

**Processed dataset and trained checkpoints**: distributed separately (outside the repository, because of their size) as `dataset.zip` (~620 MB) and `models.zip` (~1.1 GB). To use them:

```
dataset.zip → extract into data/            (creates data/final, data/wholeimage, data/manual_annotations, data/manual_crops, data/acne_1024)
models.zip  → extract into training/models/ (creates the final checkpoints and the backbone)
```

---

## Hardware requirements

- **NVIDIA GPU** with CUDA 12.1, or
- **Apple Silicon** (M1 or later) through the PyTorch MPS backend
- Python 3.9+

The code selects the device automatically (`cuda` → `mps` → `cpu`).

---

## Environment setup

**NVIDIA GPU (CUDA):**
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r training/requirements_training.txt
```

**Apple Silicon (MPS):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r training/requirements_mps.txt
```
> `mediapipe` forces numpy ≥2.0, which breaks torchvision's `ColorJitter` on Apple Silicon. Install mediapipe separately, run the segmentation, and reinstall `numpy<2.0,>=1.26.0` before training (see the comment in `requirements_mps.txt`).

---

## Training pipeline (from scratch)

> Steps 1–3 document how `data/final` and `data/wholeimage` were originally built, but the shared **`dataset.zip` already contains that output**. To retrain or reproduce results, go straight to step 4 (or to "Reproducing the results without retraining" below, if you also have `models.zip`).

1. **ACNE04 split (70/15/15, per image)**
   ```bash
   python training/utils/prepare_acne04_split.py --acne04 acne_1024 --output data/acne04_images
   ```

2. **Facial segmentation (MediaPipe)**
   ```bash
   cd training
   python utils/segment_faces_mediapipe.py --input-dir ../data/manual_annotations --output-dir ../data/manual_crops
   ```

3. **Merge the annotated collection with ACNE04** (no leakage between splits)
   ```bash
   python training/utils/merge_manual_data.py --annotations-dir data/manual_annotations --crops-dir data/manual_crops --final-dir data/final
   ```

4. **Pre-train the backbone (ImageNet → SCIN)**
   ```bash
   python training/utils/prepare_scin_pretrain.py --output data/scin_pretrain --min-samples 30
   python training/pretrain_backbone.py --data-dir data/scin_pretrain --epochs 15
   ```

5. **Train the main model (5 seeds, with SWA)**
   ```bash
   cd training
   for seed in 42 43 44 45 46; do
     python run_ablation.py --seed $seed --data-dir ../data/final --only-config with_embedding --swa \
       --pretrained-backbone models/backbone_scin_pretrained_v2.pth \
       --save-models-dir models/ablation_ckpt_clean_swa \
       --output results/ablation_clean_swa.json
   done
   ```

6. **Evaluate ensemble + TTA**
   ```bash
   python evaluate_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding --tta-views 7
   ```

---

## Reproducing the results without retraining

Full training (5 seeds × 4 configurations) takes several GPU hours. With `dataset.zip` and `models.zip` extracted (see "Data"), the held-out test split results can be reproduced directly:

```bash
cd training

# Main model (Region-Aware ResNet-18): 64.18% acc / QWK 0.718
python evaluate_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding --tta-views 7

# Ablation (conventional ResNet-18, no embedding): 64.18% acc / QWK 0.698
python evaluate_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_noembed_swa --tag no_embedding --tta-views 7

# Whole-image baseline: 76.26% acc / QWK 0.829
python evaluate_wholeimage_ensemble_tta.py --data-dir ../data/wholeimage --checkpoints-dir models/wholeimage_clean_swa --tta-views 7

# Multi-region fusion: 69.06% acc / QWK 0.770
python evaluate_multiregion_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/multiregion_ckpt_v3 --tta-views 7

# Per-region performance
python evaluate_by_region.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding --tta-views 7
```

To classify a single image that is already cropped to a region:
```bash
python predict.py path/to/image.jpg --region forehead --weights models/ablation_ckpt_clean_swa/with_embedding_seed42.pth
```
(this uses a single seed, not the full ensemble; for the reported results, use the `evaluate_*` scripts above.)

---

## Results

Grouped 5-fold cross-validation on the combined ACNE04 + AcneLevel set (2,863 images, never the same photo in train and test); metrics pooled over the 5 test folds, last epoch, no selection on the test set:

| Model | Evaluation unit | Accuracy | Macro-F1 | QWK |
|---|---|---|---|---|
| Region-Aware ResNet-18 (WithEmbed), regions fused per photo | 2,450 images | 70.45% | 0.663 | 0.706 |
| Conventional ResNet-18 (NoEmbed), regions fused per photo | 2,450 images | 66.94% | 0.630 | 0.674 |
| Region-Aware ResNet-18 (WithEmbed), per crop | 8,295 crops | 58.36% | 0.543 | 0.556 |
| Whole image: CBAM branch (SCIN + mixup + SWA + TTA, 5 seeds) | 2,863 images | 79.04% | 0.772 | 0.859 |
| Whole image: lesion-count supervised branch (7 seeds) | 2,863 images | 77.26% | 0.745 | 0.853 |
| **Whole image: 50/50 average of both branches (final reference model)** | 2,863 images | **80.20%** | **0.779** | **0.874** |

Official ACNE04 5-fold split (1,460 test images), comparable to JAGC (84.11%):

| Model | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| CBAM branch (1 seed) | 76.64% | 0.719 | 0.847 |
| Lesion-count supervised branch (5 seeds) | 81.64% | 0.788 | 0.885 |
| **50/50 average (same configuration as on the combined set)** | **80.55%** | **0.776** | **0.879** |

All models use a ResNet-18 backbone. The lesion-count branch follows JAGC's grading + counting formulation (`training/run_jagc_original_replica.py --arch resnet18`); the final numbers come from `training/evaluate_final_whole_ensemble.py` (which writes `training/results/final_whole_ensemble.json` and the confusion matrix). The moderate class remains the main bottleneck (57.6% sensitivity on the combined set, confused mostly with mild).

---

## Model (input/output)

| Input | Format |
|---------|---------|
| Region crop | tensor `[batch, 3, 224, 224]` |
| Region index | tensor `[batch]` (int, 0–4) |

| Output | Format |
|-------|---------|
| Class logits | tensor `[batch, 4]` |

Region indices: `0=forehead`, `1=chin`, `2=nose`, `3=left_cheek`, `4=right_cheek`

---

## Mobile app

A Flutter prototype (`acne_classifier_app/`) integrates the trained model and guides the user through image capture, preprocessing, per-region severity prediction and an assessment history.

There is also a native Android version in Kotlin and Jetpack Compose (`acne_classifier_android/`), with the same flow and the same ONNX ensemble, using ML Kit for face detection and ONNX Runtime for inference. Build instructions are in [`acne_classifier_android/README.md`](acne_classifier_android/README.md).

---

## Authors

Daniel C. Anselmo, Lucas G. Andrade, Enzo G. R. B. Moreira — Instituto Federal do Ceará (IFCE)
Advisors: Prof. Dr. Pedro P. Rebouças Filho, Prof. Dr. Roger M. Sarmento — PPGCC/IFCE
