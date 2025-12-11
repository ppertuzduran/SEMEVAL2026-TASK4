# Narrative Similarity - Deep Learning Approach

Deep learning solution for narrative similarity using cross-encoder (Track A) and bi-encoder (Track B) models.

## 🎯 Training Workflow

**⚡ Train on Google Colab (GPU) → 💻 Inference Locally (CPU/GPU)**

- Training scripts: Colab-only (optimized for T4 GPU)
- Inference scripts: Local execution
- Code: GitHub (version control)
- Data & Models: Google Drive (persistence)

**Training Time on Colab T4**: ~1h 15min total

---

## 🚀 Quick Start

### 1. Push Code to GitHub (2 min)

```powershell
cd C:\Users\pertu\OneDrive\Documentos\DEV\master\V4
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/narrative-similarity.git
git push -u origin main
```

### 2. Upload Data to Google Drive (2 min)

Create folder structure and upload data:

```
MyDrive/
└── narrative_similarity/      ← Create this folder
    └── data/                  ← Create this subfolder
        ├── dev_track_a.jsonl  ← Upload
        └── dev_track_b.jsonl  ← Upload
```

**Only 2 files needed!** Everything else comes from GitHub.

### 3. Train on Google Colab (~1h 15min)

1. Open [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU** ✅
3. Copy-paste these cells:

```python
# === CELL 1: Install Dependencies ===
!nvidia-smi
!pip install -q transformers sentence-transformers datasets scikit-learn pyyaml accelerate

# === CELL 2: Clone Repository ===
REPO_URL = "https://github.com/YOUR_USERNAME/narrative-similarity.git"  # ← CHANGE THIS
!git clone {REPO_URL} /content/project
%cd /content/project

# === CELL 3: Mount Drive & Setup ===
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/project/training')
from colab_utils import setup_colab_environment
paths = setup_colab_environment(project_name="narrative_similarity")

# === CELL 4: Prepare Data (~1 min) ===
!python training/prepare_data.py

# === CELL 5: Train Track B (~20 min) ===
!python training/train_track_b.py

# === CELL 6: Train Track A (~50 min) ===
!python training/train_track_a.py

# === CELL 7: Download Models (Optional) ===
!cd /content/drive/MyDrive/narrative_similarity && zip -r /content/trained_models.zip models/
from google.colab import files
files.download('/content/trained_models.zip')
```

### 4. Get Models Locally

**Option A: Download from Colab** (Cell 7 above)
```powershell
cd C:\Users\pertu\OneDrive\Documentos\DEV\master\V4
Expand-Archive -Path trained_models.zip -DestinationPath . -Force
```

**Option B: Mount Drive** (Recommended)
1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Models auto-sync to: `G:\My Drive\narrative_similarity\models\`

### 5. Run Inference Locally

```powershell
# Install dependencies (one time)
pip install transformers sentence-transformers torch numpy pandas pyyaml

# Run inference
python track_b.py
python track_a.py

# Evaluate
python scripts/eval_local.py --track both
```

---

## 🔄 Updating Code

Modified training code? Just push and pull:

```powershell
# Local
git add .
git commit -m "Updated training"
git push

# Colab
%cd /content/project
!git pull
!python training/train_track_a.py  # Re-run training
```

---

## 📁 Project Structure

```
V4/
├── data/
│   ├── dev_track_a.jsonl          # Input data
│   └── dev_track_b.jsonl          # Input data
├── training/
│   ├── colab_utils.py             # Colab setup utilities
│   ├── prepare_data.py            # Data preparation
│   ├── train_track_a.py           # Track A training
│   └── train_track_b.py           # Track B training
├── models/                         # Trained models (from Colab)
├── output/                         # Predictions & embeddings
├── config.yaml                     # Hyperparameters
├── track_a.py                      # Track A inference
├── track_b.py                      # Track B inference
└── APPROACH.md                     # Technical details
```

---

## ⚙️ Configuration

Edit `config.yaml` for hyperparameters:

```yaml
track_b:
  base_model: "BAAI/bge-base-en-v1.5"
  batch_size: 24
  epochs: 10

track_a:
  base_model: "roberta-large"
  batch_size: 6
  epochs: 5
  use_kfold: true
  n_folds: 5
```

**Colab T4 (16GB)** - can increase batch sizes:
```yaml
track_a:
  batch_size: 12  # vs 6 default
track_b:
  batch_size: 32  # vs 24 default
```

---

## 🎯 Expected Performance

- **Baseline**: ~50% accuracy
- **Track B (BGE)**: 85-87%
- **Track A (RoBERTa-large)**: 70-80%

**Improvement: +25-35% over baseline**

---

## 💻 Requirements

### For Colab Training (Recommended)
- Google account (free tier OK)
- ~2GB Google Drive space
- Internet connection

### For Local Inference
- GPU: Optional (CPU works, slower)
- RAM: 8GB+
- Storage: ~2GB

---

## 🔧 Troubleshooting

### Colab Issues

**"Session disconnected"**
- Models auto-save to Drive every epoch
- Re-run training cell to resume

**"CUDA out of memory"**
- Reduce batch sizes in `config.yaml`
- Default settings should work on T4

### Local Issues

**"Model not found"**
- Check models in: `V4/models/`
- Or update paths in `config.yaml` to Drive location

**"CUDA out of memory" (inference)**
- Track A uses `batch_size=4` for 6GB GPUs
- Reduce further if needed in `track_a.py`

**Using Drive-mounted models**
```yaml
track_a:
  model_save_path: "G:/My Drive/narrative_similarity/models/track_a_cross_encoder"
```

---

## 📦 Submission

```powershell
# Windows
Compress-Archive -Path output\* -DestinationPath submission.zip

# Linux/Mac
zip -j submission.zip output/*
```

Should contain:
- `track_a.jsonl` - Predictions
- `track_b.npy` - Embeddings

---

## 📖 Documentation

- **[APPROACH.md](APPROACH.md)** - Technical approach and SOTA techniques

---

## 🛠️ Key Technologies

- **PyTorch** - Deep learning framework
- **Transformers** - RoBERTa-large model
- **Sentence-Transformers** - BGE embeddings
- **Google Colab** - Free GPU training
- **Google Drive** - Model persistence

### Key Features
- Pairwise scoring (Track A)
- Direct metric optimization (Track B)
- Mixed precision (FP16) training
- K-fold cross-validation
- Multi-phase training
- Colab/Drive integration

---

## 📊 Training Time Comparison

| Environment | Total Time |
|-------------|------------|
| **Colab T4 (16GB)** | **71 min** ⚡ |
| Local RTX 4050 (6GB) | 256 min |

**Speedup: 3.6x faster on Colab**

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| **Inference Track B** | `python track_b.py` |
| **Inference Track A** | `python track_a.py` |
| **Evaluate Both** | `python scripts/eval_local.py --track both` |
| **Create Submission** | `Compress-Archive -Path output\* -DestinationPath submission.zip` |

---

**Ready to train on Colab and run inference locally!** 🚀
