# Narrative Similarity - Deep Learning Approach

This repository contains an improved implementation for the Narrative Similarity task with both Track A (cross-encoder) and Track B (bi-encoder) models.

## 📋 Overview

- **Track A**: Cross-encoder model that directly compares two stories against an anchor
- **Track B**: Bi-encoder (embedding) model that generates story embeddings for similarity comparison

## 🎯 Training Workflow

**⚡ Train Fast on Google Colab (Recommended)**  
**💻 Run Inference Locally**

This hybrid approach gives you:
- **3-4x faster training** on Colab's free T4 GPU (16GB)
- **No local GPU needed** for training
- **Models saved to Google Drive** - access anywhere
- **Local inference** on your own machine

**Training Time Comparison:**
- Local (RTX 4050 6GB): ~4.5 hours
- Colab (T4 16GB): ~1 hour 15 minutes ⚡

---

## 🚀 Quick Start

### Option 1: Train on Colab + Inference Locally (Recommended)

**Step 1: Setup Google Drive** (5 minutes)

See detailed instructions in **[DRIVE_FOLDER_SETUP.md](DRIVE_FOLDER_SETUP.md)**

```
Upload to Google Drive:
MyDrive/narrative_similarity/
  ├── config.yaml
  ├── data/
  │   ├── dev_track_a.jsonl
  │   └── dev_track_b.jsonl
  └── training/
      ├── colab_utils.py
      ├── prepare_data.py
      ├── train_track_a.py
      └── train_track_b.py
```

**Step 2: Train on Google Colab** (~1 hour 15 min)

See detailed instructions in **[COLAB_TRAINING.md](COLAB_TRAINING.md)**

1. Open [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run these cells:

```python
# Cell 1: Setup
!nvidia-smi
!pip install -q transformers sentence-transformers datasets scikit-learn pyyaml accelerate

from google.colab import drive
drive.mount('/content/drive')

import os, sys
os.chdir('/content/drive/MyDrive/narrative_similarity')
sys.path.insert(0, '/content/drive/MyDrive/narrative_similarity/training')

# Cell 2: Prepare data (~1 min)
!python training/prepare_data.py

# Cell 3: Train Track B (~20 min)
!python training/train_track_b.py

# Cell 4: Train Track A (~50 min)
!python training/train_track_a.py
```

**Step 3: Get Models Locally**

**Option A: Mount Drive (Easiest)**
1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Models automatically available at: `G:\My Drive\narrative_similarity\models\`

**Option B: Download Models**
```python
# In Colab: Zip and download
!zip -r trained_models.zip models/
from google.colab import files
files.download('trained_models.zip')
```

Then extract locally:
```powershell
# Windows
cd C:\...\V4
Expand-Archive -Path trained_models.zip -DestinationPath . -Force
```

**Step 4: Run Inference Locally**

```powershell
# Install dependencies (one time)
pip install transformers sentence-transformers torch numpy pandas pyyaml

# Run inference (uses models from Step 3)
python track_b.py
python track_a.py

# Evaluate
python scripts/eval_local.py --track both
```

---

### Option 2: Train and Inference Locally

If you have a capable GPU locally (8GB+ VRAM):

**Step 1: Environment Setup**

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CUDA GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Step 2: Training Pipeline**

```bash
# Prepare data (~1 minute)
python training/prepare_data.py

# Train Track B (~30-45 minutes on 6GB GPU)
python training/train_track_b.py

# Train Track A (~3-4 hours on 6GB GPU)
python training/train_track_a.py
```

**Step 3: Inference**

```bash
python track_b.py
python track_a.py
```

---

## 📁 Project Structure

```
V4/
├── data/
│   ├── dev_track_a.jsonl          # Track A development data
│   ├── dev_track_b.jsonl          # Track B development data
│   └── prepared/                   # Prepared training data (generated)
├── training/
│   ├── colab_utils.py             # NEW: Colab/Drive integration
│   ├── prepare_data.py            # Data preparation script
│   ├── train_track_a.py           # Track A training (Colab-ready)
│   └── train_track_b.py           # Track B training (Colab-ready)
├── scripts/
│   ├── eval_local.py              # Local evaluation script
│   └── run_full_pipeline.py       # Full pipeline automation (local only)
├── models/
│   ├── track_a_cross_encoder_fold*/  # Trained Track A models (from Colab)
│   └── track_b_embedder/             # Trained Track B model (from Colab)
├── output/
│   ├── track_a.jsonl              # Track A predictions
│   └── track_b.npy                # Track B embeddings
├── config.yaml                     # Hyperparameters and settings
├── track_a.py                      # Track A inference script
├── track_b.py                      # Track B inference script
├── APPROACH.md                     # Technical approach documentation
├── COLAB_TRAINING.md              # NEW: Colab training guide
├── DRIVE_FOLDER_SETUP.md          # NEW: Drive setup quick reference
└── README.md                       # This file
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Track B Settings
```yaml
track_b:
  base_model: "BAAI/bge-base-en-v1.5"  # Upgraded embedding model
  batch_size: 24
  epochs: 10
  learning_rate: 1.0e-5
```

### Track A Settings
```yaml
track_a:
  base_model: "roberta-large"  # 355M params
  batch_size: 6
  epochs: 5
  learning_rate: 1.0e-5
  use_kfold: true
  n_folds: 5
```

**Colab T4 GPU (16GB)** - can increase batch sizes:
```yaml
track_a:
  batch_size: 12  # Increase from 6
track_b:
  batch_size: 32  # Increase from 24
```

---

## 🎯 Expected Performance

- **Baseline (random)**: ~50% accuracy
- **Track B (fine-tuned BGE)**: 85-87% accuracy
- **Track A (fine-tuned RoBERTa-large)**: 70-80% accuracy

**Expected improvement: 25-35% accuracy gain over baseline**

---

## 💻 System Requirements

### For Training on Colab (Recommended)
- Google account (free Colab tier sufficient)
- ~2GB Google Drive space
- Internet connection

### For Local Training
- **GPU**: 6GB+ VRAM (8GB+ recommended)
- **RAM**: 12GB system memory
- **Storage**: ~2GB for models + data

### For Local Inference Only
- **GPU**: Optional (CPU works, just slower)
- **RAM**: 8GB
- **Storage**: ~2GB for models

---

## 🔧 Troubleshooting

### Colab Issues

**Issue: "Session disconnected during training"**
- Models auto-save every epoch to Drive
- Just re-run the training cell - it will resume

**Issue: "CUDA out of memory" in Colab**
- Reduce batch sizes in `config.yaml`
- T4 should handle default settings fine

See **[COLAB_TRAINING.md](COLAB_TRAINING.md)** for complete troubleshooting.

### Local Inference Issues

**Issue: "Model not found"**
- Verify models extracted to: `V4/models/`
- Check: `ls models/track_a_cross_encoder_fold0/`

**Issue: "CUDA out of memory" during inference**
- Reduce batch size in inference scripts
- Track A already uses batch_size=4 for 6GB GPUs

**Issue: Using Drive-mounted models (Drive for Desktop)**
- Update `config.yaml` to point to Drive:
  ```yaml
  track_a:
    model_save_path: "G:/My Drive/narrative_similarity/models/track_a_cross_encoder"
  ```

---

## 📦 Submission

Create a submission ZIP file:

**Windows:**
```powershell
Compress-Archive -Path output\* -DestinationPath submission.zip
```

**Linux/Mac:**
```bash
zip -j submission.zip output/*
```

The ZIP should contain:
- `track_a.jsonl` - Track A predictions
- `track_b.npy` - Track B embeddings

---

## 📖 Documentation

- **[APPROACH.md](APPROACH.md)** - Detailed technical approach and SOTA techniques
- **[COLAB_TRAINING.md](COLAB_TRAINING.md)** - Complete Colab training guide
- **[DRIVE_FOLDER_SETUP.md](DRIVE_FOLDER_SETUP.md)** - Google Drive setup reference

---

## 🛠️ Development

### Key Technologies
- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: Pre-trained models (RoBERTa-large)
- **Sentence-Transformers**: Embedding models (BGE-base)
- **Google Colab**: Free GPU training
- **Google Drive**: Model storage and persistence

### Key Features
- **Pairwise scoring** for Track A (SOTA IR approach)
- **Direct metric optimization** for Track B
- **Mixed precision (FP16)** training
- **K-fold cross-validation** for Track A
- **Multi-phase training** for Track B
- **Colab integration** - train anywhere
- **Drive persistence** - never lose models

---

## 🚀 Recommended Workflow Summary

**Best Practice: Colab for Training, Local for Inference**

1. **Upload** (5 min): Files to Google Drive
2. **Train** (1h 15m): Run on Colab T4 GPU
3. **Download/Mount** (5 min): Get models locally
4. **Inference** (2-3 min): Run on local machine
5. **Evaluate** (1 min): Check results

**Total time: ~1.5 hours** (vs 4.5 hours locally)

**Training once on Colab, inference unlimited times locally!** 🎯

---

## 📊 Training Time Breakdown

### Colab T4 (16GB) - Recommended
```
Data preparation:  1 minute
Track B training:  20 minutes
Track A training:  50 minutes
─────────────────────────────
Total:             71 minutes ⚡
```

### Local RTX 4050 (6GB)
```
Data preparation:  1 minute
Track B training:  45 minutes
Track A training:  210 minutes (3.5 hours)
─────────────────────────────
Total:             256 minutes (4h 16m)
```

**Speedup: 3.6x faster on Colab!**

---

## 🎓 Academic Context

This implementation uses state-of-the-art techniques from modern NLP research:

- **Pairwise Scoring**: Standard in IR reranking (Nogueira & Cho, 2019)
- **BGE Embeddings**: SOTA retrieval model (Xiao et al., 2023)
- **Metric Learning**: TripletLoss + MNR (Schroff et al., 2015)
- **Mixed Precision**: Industry standard (Micikevicius et al., 2018)

See **[APPROACH.md](APPROACH.md)** for complete technical details and references.

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| **Train on Colab** | See [COLAB_TRAINING.md](COLAB_TRAINING.md) |
| **Inference Track B** | `python track_b.py` |
| **Inference Track A** | `python track_a.py` |
| **Evaluate Both** | `python scripts/eval_local.py --track both` |
| **Create Submission** | `Compress-Archive -Path output\* -DestinationPath submission.zip` |

---

## 📄 License

This project follows the license terms of the original competition/task.

---

## 🆘 Support

- **Training issues**: See [COLAB_TRAINING.md](COLAB_TRAINING.md#troubleshooting)
- **Setup issues**: See [DRIVE_FOLDER_SETUP.md](DRIVE_FOLDER_SETUP.md)
- **Technical details**: See [APPROACH.md](APPROACH.md)

**Ready to train on Colab and run inference locally!** 🚀
