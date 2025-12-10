# Narrative Similarity - Deep Learning Approach

This repository contains an improved implementation for the Narrative Similarity task with both Track A (cross-encoder) and Track B (bi-encoder) models.

## 📋 Overview

- **Track A**: Cross-encoder model that directly compares two stories against an anchor
- **Track B**: Bi-encoder (embedding) model that generates story embeddings for similarity comparison

## 🚀 Quick Start

### 1. Environment Setup

Create a virtual environment and install dependencies:

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**For CUDA Support (GPU):**
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Training Pipeline

**Step 1: Prepare Data**
```bash
python training/prepare_data.py
```

This creates triplets, pairs, and cross-encoder datasets from the dev data.

**Step 2: Train Track B (Bi-encoder) - ~15-30 minutes**
```bash
python training/train_track_b.py
```

Trains an embedding model using TripletLoss and MultipleNegativesRankingLoss.

**Step 3: Train Track A (Cross-encoder) - ~45-90 minutes**
```bash
python training/train_track_a.py
```

Trains a cross-encoder with k-fold cross-validation (5 folds by default).

### 3. Inference

**Run Track B Inference:**
```bash
python track_b.py
```
Output: `output/track_b.npy` (embeddings matrix)

**Run Track A Inference:**
```bash
python track_a.py
```
Output: `output/track_a.jsonl` (predictions)

### 4. Evaluation

```bash
# Evaluate both tracks
python scripts/eval_local.py --track both

# Evaluate Track A only
python scripts/eval_local.py --track a

# Evaluate Track B only
python scripts/eval_local.py --track b
```

### 5. Alternative: Run Full Pipeline

```bash
python scripts/run_full_pipeline.py --stage all
```

This runs all steps automatically: data preparation → training → inference → evaluation.

## 📁 Project Structure

```
V4/
├── data/
│   ├── dev_track_a.jsonl          # Track A development data (not in repo)
│   ├── dev_track_b.jsonl          # Track B development data (not in repo)
│   └── prepared/                   # Prepared training data (generated)
├── training/
│   ├── prepare_data.py            # Data preparation script
│   ├── train_track_a.py           # Track A training script
│   └── train_track_b.py           # Track B training script
├── scripts/
│   ├── eval_local.py              # Local evaluation script
│   └── run_full_pipeline.py       # Full pipeline automation
├── models/
│   ├── track_a_cross_encoder/     # Trained Track A models (generated)
│   └── track_b_embedder/          # Trained Track B model (generated)
├── output/
│   ├── track_a.jsonl              # Track A predictions (generated)
│   └── track_b.npy                # Track B embeddings (generated)
├── config.yaml                     # Hyperparameters and settings
├── requirements.txt                # Python dependencies
├── track_a.py                      # Track A inference script
├── track_b.py                      # Track B inference script
├── APPROACH.md                     # Technical approach documentation
└── README.md                       # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize hyperparameters:

### Track B Settings
```yaml
track_b:
  base_model: "all-MiniLM-L6-v2"  # Embedding model
  batch_size: 32
  epochs: 10
  learning_rate: 2.0e-5
  triplet_margin: 0.5
```

### Track A Settings
```yaml
track_a:
  base_model: "roberta-base"  # Transformer model
  batch_size: 16
  epochs: 5
  learning_rate: 2.0e-5
  use_kfold: true
  n_folds: 5
  use_ensemble: false  # Set to true for ensemble inference
```

## 🎯 Expected Performance

- **Baseline (random)**: ~50% accuracy
- **Track B (fine-tuned bi-encoder)**: 65-75% accuracy
- **Track A (fine-tuned cross-encoder)**: 75-85% accuracy

**Expected improvement: 25-35% accuracy gain over baseline**

## 💻 System Requirements

- **Python**: 3.10+
- **GPU**: 6GB+ VRAM (RTX 4050 tested, CPU also supported)
- **Disk Space**: ~2GB
- **Training Time** (GPU): 
  - Data preparation: ~1 minute
  - Track B: ~15-30 minutes
  - Track A: ~45-90 minutes (5 folds)

## 🔧 Troubleshooting

### Out of Memory Errors

**For Track A Training:**
Edit `config.yaml`:
```yaml
track_a:
  batch_size: 8  # Reduce from 16
  mixed_precision: true  # Ensure enabled
```

**For Track A Inference:**
The script automatically uses batch size 4 for 6GB GPUs. If still OOM:
1. Edit `track_a.py` line ~238 and change `batch_size=4` to `batch_size=2` or `1`
2. Clear GPU cache: `python -c "import torch; torch.cuda.empty_cache()"`

**For Track B Training:**
Edit `config.yaml`:
```yaml
track_b:
  batch_size: 16  # Reduce from 32
```

### No GPU Detected

Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Missing Models

If inference fails because no trained models exist:
- Track B will fall back to baseline `all-MiniLM-L6-v2`
- Track A will fall back to random predictions

Train the models first before running inference.

### Faster Training Options

**Single Model Training (faster):**
Edit `config.yaml`:
```yaml
track_a:
  use_kfold: false  # Train single model instead of 5 folds
```
This reduces Track A training time from ~60 minutes to ~12 minutes.

**Reduce Epochs:**
```yaml
track_b:
  epochs: 5  # Reduce from 10
  
track_a:
  epochs: 3  # Reduce from 5
```

## 📦 Submission

Create a submission ZIP file:

**Windows (PowerShell):**
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

## 📖 Documentation

See `APPROACH.md` for detailed technical approach and methodology.

## 🛠️ Development

### Key Technologies
- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: Pre-trained models
- **Sentence-Transformers**: Embedding models and metric learning
- **scikit-learn**: Cross-validation and metrics

### Key Features
- Mixed precision (FP16) training
- K-fold cross-validation
- TripletLoss + MultipleNegativesRankingLoss
- Early stopping and best model checkpointing
- Data augmentation (A/B swapping)
- Automatic fallback to baseline models
- Memory-efficient inference for small GPUs

## 🐛 Common Issues

**Issue: `ImportError: cannot import name 'AdamW' from 'transformers'`**
- Already fixed: We use `torch.optim.AdamW` instead

**Issue: DeBERTa tokenizer error**
- Already fixed: Default model is now `roberta-base`

**Issue: CUDA out of memory**
- See troubleshooting section above for batch size reduction

## 📝 Quick Reference

| Task | Command |
|------|---------|
| Setup | `pip install -r requirements.txt` |
| Prepare Data | `python training/prepare_data.py` |
| Train Track B | `python training/train_track_b.py` |
| Train Track A | `python training/train_track_a.py` |
| Inference Track B | `python track_b.py` |
| Inference Track A | `python track_a.py` |
| Evaluate | `python scripts/eval_local.py --track both` |
| Full Pipeline | `python scripts/run_full_pipeline.py --stage all` |

## 📄 License

This project follows the license terms of the original competition/task.
