# Narrative Similarity - V2 Architecture (Track A & Track B)

Concise guide to train and infer on Google Colab with Google Drive.

## V2 Architecture Overview

**Key Changes**:
- **Track B**: Primary model with 512-dim projection head + unified composite loss
- **Track A**: Lightweight MLP head operating on Track B embeddings (not cross-encoder)
- **Philosophy**: One strong bi-encoder + tiny interaction layer for better generalization

**Benefits**:
- Reduced overfitting (Track A now uses frozen Track B embeddings)
- Enforced consistency between tracks (same embedding space)
- Smaller model size for Track A (MLP head ~few MB vs cross-encoder ~1GB)

## Setup
- Python 3.11 on Colab GPU (T4).
- Data in Drive:
  - `MyDrive/narrative_similarity/data/dev_track_a.jsonl`
  - `MyDrive/narrative_similarity/data/dev_track_b.jsonl`

## Colab Configuration
```python
!nvidia-smi
!pip install -q transformers sentence-transformers datasets scikit-learn pyyaml accelerate

# Clone repo
REPO_URL = "https://github.com/ppertuzduran/SEMEVAL2026-TASK4.git"
!git clone {REPO_URL} /content/project
%cd /content/project

# Mount Drive and setup
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/project/training')
from colab_utils import setup_colab_environment
paths = setup_colab_environment(project_name="narrative_similarity")
```

## Drive Paths in `config.yaml`
- Set after mounting:
  - `data.dev_track_a: /content/drive/MyDrive/narrative_similarity/data/dev_track_a.jsonl`
  - `data.dev_track_b: /content/drive/MyDrive/narrative_similarity/data/dev_track_b.jsonl`
  - `data.output_dir: /content/project/output`
  - `track_a.model_save_path: /content/drive/MyDrive/narrative_similarity/models/track_a_mlp_head`
  - `track_a.track_b_model_path: /content/drive/MyDrive/narrative_similarity/models/track_b_embedder`
  - `track_b.model_save_path: /content/drive/MyDrive/narrative_similarity/models/track_b_embedder`

## Step-by-Step Workflow

### 1) Prepare data
```bash
python training/prepare_data.py
```

### 2) Train Track B (bi-encoder with projection head)
```bash
python training/train_track_b.py
```
**V2 Features**:
- Adds 512-dim projection head (reduces from 1024 to 512)
- Uses unified composite loss: MarginRanking + PairwiseSoftmax + MNR
- No more multi-phase curriculum (simplified to single unified training)
- Removes obsolete A→B distillation

**If T4 OOM**: Lower `batch_size` in config, or switch `base_model` to `bge-base`.

### 3) Train Track A (MLP head over Track B embeddings)
```bash
python training/train_track_a.py
```
**V2 Features**:
- Lightweight MLP head (not cross-encoder)
- Operates on frozen Track B embeddings
- Pairwise features: `[e_anchor, e_cand, |e_anchor - e_cand|, e_anchor ⊙ e_cand]`
- Distillation from Track B cosine similarities
- K-fold cross-validation for robustness
- Removes obsolete B→A distillation from cross-encoder

**Requirements**: Track B must be trained first!

### 4) Inference
```bash
python track_b.py   # embeddings, saves track_b.npy (512-dim)
python track_a.py   # predictions, saves track_a.jsonl (uses Track B + MLP heads)
```

**V2 Changes**:
- `track_b.py`: Verifies 512-dim embeddings from projection head
- `track_a.py`: Loads Track B model + MLP head ensemble (not cross-encoder)

### 5) Evaluation
```bash
python scripts/eval_local.py --track both
```

### 6) Save results to Drive
```bash
mkdir -p /content/drive/MyDrive/narrative_similarity/output
cp -r output/* /content/drive/MyDrive/narrative_similarity/output/
```

## Architecture Details

### Track B (Bi-Encoder with Projection Head)
```
Input Text → BGE-large-en-v1.5 (1024-dim) → Dense(512-dim) → LayerNorm → L2 Normalize → 512-dim embedding
```

**Training**: Composite loss = MarginRanking + PairwiseSoftmax + MNR

### Track A (MLP Head over Track B)
```
(anchor, A, B) → Track B encoder → (e_anchor, e_a, e_b)
                                  ↓
                   Pairwise features: φ(a) = [e_anchor, e_a, |e_anchor - e_a|, e_anchor ⊙ e_a]
                                  ↓
                   MLP: Linear(4d→2d) + GELU + Dropout + Linear(2d→1)
                                  ↓
                   Scores: (score_a, score_b) → softmax → prediction
```

**Training**: Cross-entropy + KL distillation from Track B cosine similarities

## Configuration Highlights

### Track B (`config.yaml`)
```yaml
track_b:
  projection_dim: 512              # V2: Projection head dimension
  loss_weights:
    margin: 1.0                    # MarginRankingLoss weight
    pairwise: 0.5                  # PairwiseSoftmaxLoss weight
    mnr: 0.5                       # MNR weight
  margin: 0.2                      # Margin for ranking loss
  temperature: 0.7                 # Temperature scaling
  batch_size: 4
  epochs: 5
```

### Track A (`config.yaml`)
```yaml
track_a:
  architecture: "mlp_head"         # V2: MLP head instead of cross-encoder
  track_b_model_path: "..."        # Required: Track B checkpoint
  freeze_track_b: true             # Head-only training (recommended)
  mlp_hidden_dim: 1024             # 2d where d=512
  mlp_dropout: 0.2
  batch_size: 8                    # Larger batch since MLP is tiny
  epochs: 10
  learning_rate: 1.0e-4            # Higher LR for head-only
  distill_weight: 0.3              # KL from Track B cosine similarities
```

## Expected Performance

- **Track B**: ~0.94 accuracy (improved with projection head + composite loss)
- **Track A**: ~0.91-0.93 accuracy (improved with MLP head, less overfitting)

## Minimal Local Notes (optional)
- Python 3.11 venv; install PyTorch CUDA wheel matching your GPU; `pip install -r requirements.txt`.
- Run inference locally (slower): `python track_b.py`, `python track_a.py`, then `python scripts/eval_local.py --track both`.
