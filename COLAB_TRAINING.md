# Google Colab Training Guide

Complete guide for training models on Google Colab with Google Drive integration.

---

## 📁 Google Drive Folder Structure

### Required Structure

Create this exact structure in your Google Drive:

```
MyDrive/
└── narrative_similarity/          ← Root project folder
    ├── config.yaml                 ← Upload from local
    ├── data/
    │   ├── dev_track_a.jsonl      ← Upload from local
    │   └── dev_track_b.jsonl      ← Upload from local
    ├── training/                   ← Create empty folder
    │   ├── colab_utils.py         ← Upload from local/training/
    │   ├── prepare_data.py        ← Upload from local/training/
    │   ├── train_track_a.py       ← Upload from local/training/
    │   └── train_track_b.py       ← Upload from local/training/
    ├── models/                     ← Auto-created, models saved here
    │   ├── track_a_cross_encoder_fold0/  ← Created during training
    │   ├── track_a_cross_encoder_fold1/
    │   ├── ...
    │   └── track_b_embedder/
    ├── data/prepared/              ← Auto-created during data prep
    │   ├── triplets.jsonl
    │   ├── pairs.jsonl
    │   └── ...
    └── output/                     ← Auto-created for logs
```

### Important Notes

- **Root folder name**: Must be `narrative_similarity` (or change in colab_utils.py)
- **All trained models** automatically saved to `models/` in Drive
- **No local storage** - everything persists in Drive
- **Resume training** - models in Drive allow continuation after disconnect

---

## 🚀 Step-by-Step Setup

### Step 1: Prepare Local Files

On your local machine:

```powershell
# Ensure you have these files ready
config.yaml
data/dev_track_a.jsonl
data/dev_track_b.jsonl
training/colab_utils.py
training/prepare_data.py
training/train_track_a.py
training/train_track_b.py
```

### Step 2: Create Drive Folder

1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `narrative_similarity`
3. Inside it, create subfolder: `data` and `training`

### Step 3: Upload Files to Drive

Upload to `narrative_similarity/`:
- `config.yaml`

Upload to `narrative_similarity/data/`:
- `dev_track_a.jsonl`
- `dev_track_b.jsonl`

Upload to `narrative_similarity/training/`:
- `colab_utils.py`
- `prepare_data.py`
- `train_track_a.py`
- `train_track_b.py`

### Step 4: Create Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → New Notebook**
3. **IMPORTANT**: Enable GPU
   - Click **Runtime → Change runtime type**
   - Select **T4 GPU** (or V100/A100 if Pro)
   - Click **Save**

---

## 📝 Colab Notebook Cells

Create these cells in your Colab notebook:

### Cell 1: Setup & Mount Drive

```python
# Check GPU
!nvidia-smi

# Install dependencies (required only once per session)
!pip install -q transformers sentence-transformers datasets scikit-learn pyyaml accelerate

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/narrative_similarity')
print(f"Current directory: {os.getcwd()}")

# Add training folder to path
import sys
sys.path.insert(0, '/content/drive/MyDrive/narrative_similarity/training')

print("\n✓ Setup complete!")
```

### Cell 2: Verify Setup

```python
# Verify all files are present
import os
from pathlib import Path

required_files = [
    'config.yaml',
    'data/dev_track_a.jsonl',
    'data/dev_track_b.jsonl',
    'training/colab_utils.py',
    'training/prepare_data.py',
    'training/train_track_a.py',
    'training/train_track_b.py'
]

print("Checking files...")
all_present = True
for file in required_files:
    exists = Path(file).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {file}")
    if not exists:
        all_present = False

if all_present:
    print("\n✓ All files present - ready to train!")
else:
    print("\n✗ Missing files - please upload them to Drive")
```

### Cell 3: Prepare Data

```python
# Run data preparation
%cd /content/drive/MyDrive/narrative_similarity
!python training/prepare_data.py
```

**Expected output:**
```
RUNNING IN GOOGLE COLAB
Mounting Google Drive...
✓ Drive mounted successfully!
✓ Project directory: /content/drive/MyDrive/narrative_similarity
✓ All required files found
GPU: Tesla T4
VRAM: 15.89 GB
Loading Track A data...
Created triplets, pairs, cross-encoder data
✓ Data preparation complete!
```

### Cell 4: Train Track B (~15-20 minutes)

```python
# Train Track B (bi-encoder)
%cd /content/drive/MyDrive/narrative_similarity
!python training/train_track_b.py
```

**Training progress:**
- Phase 1: Pairwise Softmax (~5-7 min)
- Phase 2: TripletLoss (~5-7 min)
- Phase 3: MNR (~5-7 min)
- Model auto-saved to `models/track_b_embedder/`

### Cell 5: Train Track A (~45-60 minutes)

```python
# Train Track A (cross-encoder with 5-fold CV)
%cd /content/drive/MyDrive/narrative_similarity
!python training/train_track_a.py
```

**Training progress:**
- Fold 1/5 (~10-12 min)
- Fold 2/5 (~10-12 min)
- Fold 3/5 (~10-12 min)
- Fold 4/5 (~10-12 min)
- Fold 5/5 (~10-12 min)
- Models saved to `models/track_a_cross_encoder_fold{0-4}/`

### Cell 6: Check Trained Models

```python
# List all trained models
import os
from pathlib import Path

models_dir = Path('models')
if models_dir.exists():
    print("✓ Trained models:")
    for model_path in sorted(models_dir.iterdir()):
        if model_path.is_dir() and not model_path.name.startswith('.'):
            size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            print(f"  {model_path.name}: {size / 1024**2:.1f} MB")
else:
    print("✗ No models directory found")
```

---

## 💻 Local Inference Using Drive Models

### Option 1: Mount Drive Locally (Windows)

Install **Google Drive for Desktop**:
1. Download from [Google Drive Desktop](https://www.google.com/drive/download/)
2. Install and sign in
3. Your Drive appears as `G:\My Drive\` (or similar)

Then in your local project:

```powershell
# Create symbolic link to Drive models
cmd /c mklink /D "C:\...\V4\models" "G:\My Drive\narrative_similarity\models"

# Run inference (uses Drive-mounted models)
python track_b.py
python track_a.py
```

### Option 2: Download Models from Drive

In Colab, run:

```python
# Zip models for download
!zip -r trained_models.zip models/

# Download
from google.colab import files
files.download('trained_models.zip')
```

Then locally:

```powershell
# Extract to your project
Expand-Archive -Path trained_models.zip -DestinationPath .

# Run inference
python track_b.py
python track_a.py
```

### Option 3: Update config.yaml to Point to Drive

Edit `config.yaml` locally:

```yaml
track_a:
  model_save_path: "G:/My Drive/narrative_similarity/models/track_a_cross_encoder"

track_b:
  model_save_path: "G:/My Drive/narrative_similarity/models/track_b_embedder"
```

Then:

```powershell
python track_b.py
python track_a.py
```

---

## ⚡ Colab Pro Tips

### 1. Keep Session Alive

Add this cell to prevent disconnection:

```python
# Keep session alive (run in background)
import time
from IPython.display import clear_output

while True:
    time.sleep(60)
    clear_output(wait=True)
    print("Session alive...")
```

### 2. Monitor Training in Real-Time

```python
# Watch training logs
!tail -f /content/drive/MyDrive/narrative_similarity/output/training.log
```

### 3. Increase Batch Size for T4

For T4 GPU (16GB), edit `config.yaml` in Drive:

```yaml
track_a:
  batch_size: 12  # Increase from 6

track_b:
  batch_size: 32  # Increase from 24
```

### 4. Resume Interrupted Training

Training automatically resumes from best checkpoint if interrupted. Just re-run the training cell.

---

## 🔧 Troubleshooting

### Issue: "Missing required files"

**Solution**: Verify folder structure in Drive matches exactly:
```
narrative_similarity/
  ├── config.yaml
  ├── data/
  └── training/
```

### Issue: "Drive not mounted"

**Solution**: Re-run Cell 1 and approve Drive access when prompted.

### Issue: "CUDA out of memory"

**Solution**: Reduce batch sizes in `config.yaml`:
```yaml
track_a:
  batch_size: 4  # Reduce from 6
track_b:
  batch_size: 16  # Reduce from 24
```

### Issue: "No module named 'colab_utils'"

**Solution**: Ensure `colab_utils.py` is uploaded to `training/` folder in Drive.

### Issue: "Session disconnected during training"

**Solution**: 
- Models are saved automatically every epoch
- Just re-run the training cell - it will resume
- Consider using Colab Pro for longer sessions

---

## 📊 Expected Performance on Colab T4

| Task | Time | GPU Usage | VRAM | Output |
|------|------|-----------|------|--------|
| Data Prep | ~1 min | Minimal | <1GB | `data/prepared/` |
| Track B | ~15-20 min | High | ~8GB | `models/track_b_embedder/` |
| Track A | ~45-60 min | High | ~10GB | `models/track_a_cross_encoder_fold*/` |

**Total training time: ~1 hour 10 minutes**

---

## ✅ Complete Workflow Summary

1. **Setup** (once):
   - Create `narrative_similarity/` in Google Drive
   - Upload config and data files
   - Upload training scripts

2. **Train in Colab** (per run):
   - Open Colab notebook
   - Run setup cells
   - Run data preparation
   - Train Track B (~20 min)
   - Train Track A (~50 min)
   - Models saved to Drive automatically

3. **Inference Locally** (anytime):
   - Mount Drive or download models
   - Run `track_b.py` and `track_a.py`
   - Use trained models from Drive

**Benefit**: Train on powerful Colab GPUs, inference on your local machine! 🚀

---

## 📁 Files You Need

Minimum files to upload to Drive:

```
✓ config.yaml                    (19 lines)
✓ data/dev_track_a.jsonl        (~200 samples)
✓ data/dev_track_b.jsonl        (~200 samples)
✓ training/colab_utils.py       (New file created)
✓ training/prepare_data.py      (Modified)
✓ training/train_track_a.py     (Modified)
✓ training/train_track_b.py     (Modified)
```

Everything else is auto-generated during training!

