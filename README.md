# Narrative Similarity - Deep Learning Approach

Deep learning solution for narrative similarity using cross-encoder (Track A) and bi-encoder (Track B) models.

**⚠️ Python 3.11 Required** for local GPU training/inference with CUDA support.

## 🎯 Recommended Workflow

**⚡ Complete Pipeline on Google Colab + Google Drive**

```
GitHub (Code)
    ↓ clone
Google Colab (Training) → Google Drive (Models)
    ↓                          ↓
Google Colab (Inference) → Google Drive (Results)
    ↓
Local Access via Drive for Desktop
```

**Timing:**
- **Training:** ~1h 15min (Track A + Track B)
- **Inference:** ~7-9 min (both tracks with ensemble)
- **Total:** ~1.5 hours on free Colab T4

**Storage:**
- **Data:** Google Drive (`narrative_similarity/data/`)
- **Models:** Google Drive (`narrative_similarity/models/`)
- **Results:** Google Drive (`narrative_similarity/output/`)
- **Code:** GitHub (version control)

**Alternative:** Local inference on RTX 4050 (~22 min, or 2 min without ensemble)

---

## 🚀 Quick Start

### 1. Setup Local Environment with GPU Support (5 min)

**Requirements:**
- Python 3.11 (required for CUDA compatibility)
- NVIDIA GPU with CUDA support (e.g., RTX 4050)
- ~5GB disk space

**Create Python 3.11 Virtual Environment:**

```powershell
# Navigate to project directory
cd C:\Users\pertu\OneDrive\Documentos\DEV\master\SEMEVAL2026-TASK4\v4

# Create Python 3.11 virtual environment
python3.11 -m venv venv311

# Activate the environment
.\venv311\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1 support (for RTX 4050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

**Note:** Always activate `venv311` before running training or inference scripts to ensure GPU acceleration works properly.

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
REPO_URL = "https://github.com/ppertuzduran/SEMEVAL2026-TASK4.git"
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

# === CELL 7: Verify Data and Update Config ===
# First, verify data files exist in Drive
import os

data_dir = '/content/drive/MyDrive/narrative_similarity/data'
print("Checking for data files...")
print(f"📁 {data_dir}")

track_a_exists = os.path.exists(f'{data_dir}/dev_track_a.jsonl')
track_b_exists = os.path.exists(f'{data_dir}/dev_track_b.jsonl')

print(f"  {'✓' if track_a_exists else '❌'} dev_track_a.jsonl")
print(f"  {'✓' if track_b_exists else '❌'} dev_track_b.jsonl")

if not (track_a_exists and track_b_exists):
    print("\n❌ ERROR: Data files not found!")
    print("Please upload data files to Google Drive:")
    print("  MyDrive/narrative_similarity/data/dev_track_a.jsonl")
    print("  MyDrive/narrative_similarity/data/dev_track_b.jsonl")
else:
    print("\n✓ All data files found!")
    
    # Update config.yaml to use Drive paths
    import yaml
    
    with open('/content/project/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['data']['dev_track_a'] = f'{data_dir}/dev_track_a.jsonl'
    config['data']['dev_track_b'] = f'{data_dir}/dev_track_b.jsonl'
    config['data']['output_dir'] = '/content/project/output'
    config['track_a']['model_save_path'] = '/content/drive/MyDrive/narrative_similarity/models/track_a_cross_encoder'
    config['track_b']['model_save_path'] = '/content/drive/MyDrive/narrative_similarity/models/track_b_embedder'
    
    with open('/content/project/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✓ Config updated for inference!")
    print(f"  Data: {config['data']['dev_track_a']}")
    print(f"  Models: {config['track_a']['model_save_path']}")

# === CELL 8: Run Inference on Colab (RECOMMENDED - Faster than Local) ===
# ⚠️ IMPORTANT: Only run this AFTER Cell 7 shows "✓ All data files found!"

# Track B Inference (~30 seconds)
!python track_b.py

# Track A Inference (~6-8 minutes for ensemble)
!python track_a.py

# Evaluate results
!python scripts/eval_local.py --track both

# === CELL 9: Save Results to Google Drive ===
# Create output directory in Drive
!mkdir -p /content/drive/MyDrive/narrative_similarity/output

# Copy inference results to Drive
!cp -r /content/project/output/* /content/drive/MyDrive/narrative_similarity/output/

# Verify files were saved
!ls -lh /content/drive/MyDrive/narrative_similarity/output/

print("✓ Results saved to Google Drive:")
print("  - track_a.jsonl (predictions)")
print("  - track_b.npy (embeddings)")
print("  - Access at: MyDrive/narrative_similarity/output/")

# === CELL 10 (Optional): Download Results as ZIP ===
# Only needed if you DON'T have Google Drive for Desktop
!cd /content/drive/MyDrive/narrative_similarity/output && zip /content/inference_results.zip *
from google.colab import files
files.download('/content/inference_results.zip')
```

**⚡ Colab Workflow Timeline:**
- **Cells 1-3:** Setup (~2 min)
- **Cells 4-6:** Training (~1h 15min)
- **Cell 7:** Update config paths (~5 sec) ⚠️ **REQUIRED**
- **Cell 8:** Inference (~7-9 min)
- **Cell 9:** Save to Drive (~5 sec)

**⚠️ Important:** 
- Cell 7 verifies data files exist and updates config.yaml
- Only proceed to Cell 8 if Cell 7 shows "✓ All data files found!"
- If Cell 7 shows ❌, upload data files to `MyDrive/narrative_similarity/data/` first

**Quick Diagnostic:**
```python
# Run this to check your setup before inference
!ls -lh /content/drive/MyDrive/narrative_similarity/data/
!ls -lh /content/drive/MyDrive/narrative_similarity/models/
```

**⚡ Inference Speed:**
- **Track B:** ~30 seconds (vs. ~2 min local)
- **Track A Ensemble:** ~6-8 minutes (vs. ~20+ min local on RTX 4050)
- **Total:** ~7-9 minutes on Colab T4 GPU

**💾 Results Storage:**
- Results automatically saved to: `MyDrive/narrative_similarity/output/`
- Access via Google Drive for Desktop or web interface
- No manual downloads needed!

### 4. Access Results Locally (from Google Drive)

**Recommended: Use Google Drive for Desktop**

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Results auto-sync to your local machine:
   - **Models:** `G:\My Drive\narrative_similarity\models\`
   - **Inference results:** `G:\My Drive\narrative_similarity\output\`
     - `track_a.jsonl` - Track A predictions
     - `track_b.npy` - Track B embeddings

**No manual downloads needed!** Everything syncs automatically.

### 5. Run Inference Locally (Alternative to Colab)

⚠️ **Note:** Inference is **much faster on Colab** (see Cell 7 above). Local inference on RTX 4050 with ensemble can take 20+ minutes.

**If you prefer local inference:**

```powershell
# Activate the Python 3.11 virtual environment (if not already activated)
.\venv311\Scripts\Activate.ps1

# Run inference
python track_b.py
python track_a.py

# Evaluate
python scripts/eval_local.py --track both
```

**Speed Comparison (200 samples):**
| Environment | Track A Ensemble | Track B | Total |
|------------|------------------|---------|-------|
| **Colab T4 (16GB)** | 6-8 min | 30 sec | **~7-9 min** ⚡ |
| Local RTX 4050 (6GB) | 20+ min | 2 min | ~22 min |

**To speed up local inference:** Set `use_ensemble: false` in `config.yaml` (5x faster, ~1-2% lower accuracy)

---

## 🔄 Git Workflow & Repository Setup

### First-Time Setup (One-Time)

**1. Configure Git (if not already done):**
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**2. Initialize Repository and Connect to GitHub:**
```powershell
# Navigate to your project directory
cd C:\Users\pertu\OneDrive\Documentos\DEV\master\semeval2026-task4\v5.1

# Initialize git (if not already initialized)
git init

# Add remote repository
git remote add origin https://github.com/ppertuzduran/SEMEVAL2026-TASK4.git

# Check remote is configured correctly
git remote -v
```

**3. First Push (if starting fresh):**
```powershell
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Deep learning approach for narrative similarity"

# Push to GitHub (set upstream)
git push -u origin main
```

### Daily Workflow - Pushing Changes

**After making code changes:**

```powershell
# Check what files changed
git status

# Add specific files
git add track_a.py track_b.py config.yaml
# OR add all changes
git add .

# Commit with descriptive message
git commit -m "Fixed indentation and logic errors in track_a.py"

# Push to GitHub
git push origin main
```

### Pulling Updates (Colab or Another Machine)

**In Google Colab:**
```python
# Clone repository (first time)
!git clone https://github.com/ppertuzduran/SEMEVAL2026-TASK4.git /content/project
%cd /content/project

# Pull latest changes (subsequent times)
%cd /content/project
!git pull origin main

# Re-run training with updated code
!python training/train_track_a.py
```

**On Local Machine:**
```powershell
# Pull latest changes from GitHub
git pull origin main

# If there are conflicts, resolve them and then:
git add .
git commit -m "Resolved merge conflicts"
git push origin main
```

### Common Git Commands

```powershell
# View commit history
git log --oneline

# Discard local changes (careful!)
git checkout -- filename.py

# Create a new branch for experiments
git checkout -b experimental-feature

# Switch back to main branch
git checkout main

# View differences before committing
git diff

# Remove file from git (but keep locally)
git rm --cached filename
```

### .gitignore (Already Configured)

The following are automatically ignored:
- `venv311/` - Python virtual environment
- `models/` - Trained models (too large for git)
- `output/` - Generated predictions
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files

**Note:** Models should be stored in Google Drive, not Git!

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

### For Local Inference/Training
- **Python 3.11** (required for CUDA support)
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 4050)
  - CUDA 12.1 or 11.8 compatible
  - 6GB+ VRAM recommended
  - CPU-only mode also supported (slower)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: ~5GB (models + dependencies)

---

## 🔧 Troubleshooting

### Colab Issues

**"ValueError: Expected object or value" during inference**

This means data files aren't found. Check:

```python
# Run this in Colab to verify data location
!ls -lh /content/drive/MyDrive/narrative_similarity/data/
```

Should show:
```
dev_track_a.jsonl
dev_track_b.jsonl
```

**Fix:**
1. Ensure data files are uploaded to `MyDrive/narrative_similarity/data/`
2. Run Cell 7 (config update) BEFORE Cell 8 (inference)
3. Cell 7 will verify files exist and show ✓ or ❌

**"Session disconnected"**
- Models auto-save to Drive every epoch
- Re-run training cell to resume

**"CUDA out of memory"**
- Reduce batch sizes in `config.yaml`
- Default settings should work on T4

### Local Issues

**"CUDA not available" or GPU not detected**
```powershell
# Check Python version
python --version  # Should be 3.11.x

# Verify correct virtual environment
.\venv311\Scripts\Activate.ps1

# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# Reinstall PyTorch with CUDA if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

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

### From Google Drive (after Colab inference)

**Option 1: Use Drive for Desktop (Recommended)**
```powershell
# Results are already synced locally at:
cd "G:\My Drive\narrative_similarity\output"

# Create submission zip
Compress-Archive -Path track_a.jsonl,track_b.npy -DestinationPath submission.zip
```

**Option 2: From Drive Web Interface**
1. Go to Google Drive → `narrative_similarity/output/`
2. Select `track_a.jsonl` and `track_b.npy`
3. Right-click → Download → Files download as zip

### From Local Inference

```powershell
# Windows
Compress-Archive -Path output\* -DestinationPath submission.zip

# Linux/Mac
zip -j submission.zip output/*
```

### Submission Contents

Should contain:
- `track_a.jsonl` - Predictions (200 rows with text_a_is_closer predictions)
- `track_b.npy` - Embeddings (200 x 768 numpy array)

---

## 📖 Documentation

- **[APPROACH.md](APPROACH.md)** - Technical approach and SOTA techniques
- **[COLAB_INFERENCE.md](COLAB_INFERENCE.md)** - Complete guide for running inference on Colab (recommended)

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

## 📊 Time Comparison: Colab vs Local

### Training Time
| Environment | Training Time |
|-------------|--------------|
| **Colab T4 (16GB)** | **71 min** ⚡ |
| Local RTX 4050 (6GB) | 256 min |

**Training Speedup: 3.6x faster on Colab**

### Inference Time (200 samples, with ensemble)
| Environment | Track A | Track B | Total |
|-------------|---------|---------|-------|
| **Colab T4 (16GB)** | **6-8 min** | **30 sec** | **~7-9 min** ⚡ |
| Local RTX 4050 (6GB) | 20+ min | 2 min | ~22 min |

**Inference Speedup: ~3x faster on Colab**

### Complete Pipeline (Training + Inference)
- **Colab T4:** ~82 min (~1.4 hours)
- **Local RTX 4050:** ~278 min (~4.6 hours)

**💡 Recommendation:** Use Colab for both training and inference for best speed!

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
