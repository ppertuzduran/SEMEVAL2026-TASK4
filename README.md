# Narrative Similarity - V2 Architecture (Track A & Track B)

Concise guide to train and infer on Google Colab with Google Drive.

## V2 Architecture Overview

**Key Changes**:
- **Track B**: Primary model with 512-dim projection head + unified composite loss + **improvements**
- **Track A**: Lightweight MLP head operating on Track B embeddings (not cross-encoder)
- **Philosophy**: One strong bi-encoder + tiny interaction layer for better generalization

**Benefits**:
- Reduced overfitting (Track A now uses frozen Track B embeddings)
- Enforced consistency between tracks (same embedding space)
- Smaller model size for Track A (MLP head ~few MB vs cross-encoder ~1GB)
- **NEW**: Systematic improvements to push Track B to 0.95+ accuracy

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
This creates `data/prepared/` with triplets, pairs, and cross-encoder data.

### 2) [Optional] Generate augmented data
```bash
python scripts/augment_training_data.py
```
**What it does**:
- Reads from `data/prepared/`
- Applies T5 paraphrasing + synonym replacement
- Generates 2 augmented versions per example (3x total data)
- Saves to `data/augmented/` (peer directory to `prepared/`)

**Enable in training**:
```yaml
# config.yaml
augmentation:
  use_augmented_data: true  # Training scripts will use data/augmented/
```

**Expected impact**: +3-7% accuracy improvement on small datasets

**Note**: This step is optional but recommended for better generalization.

### 3) Train Track B (bi-encoder with projection head)

**V10 Approach: Systematic Hyperparameter Tuning**

You have 3 options:

#### **Option A: Automated Experiment Runner**  (Recommended - Best Results)
Let the script systematically search for the best hyperparameters:
```bash
python scripts/run_track_b_experiments.py
```

**What it does**:
1. Backs up your current `config.yaml` to `config.yaml.backup`
2. Runs a **baseline** with current settings
3. Tests incremental improvements:
   - **Regularization sweep**: Tests 9 combinations of projection dropout × weight decay
   - **Hyperparameter sweep**: Tests temperature × margin combinations
   - **Hard negative mining**: Tests 6 combinations of k × epochs
   - **SimCSE**: Optional consistency loss if needed
4. Only keeps experiments that improve validation accuracy
5. Saves all results in `experiments_track_b.json`
6. Writes the best configuration back into `config.yaml`

**Where to change search ranges**:
- Open `scripts/run_track_b_experiments.py`
- Edit the constants at the top:
  ```python
  REG_DROPOUTS = [0.0, 0.1, 0.2]           # Projection dropout values
  REG_WEIGHT_DECAYS = [0.03, 0.06, 0.1]    # Projection weight decay values
  HARD_K = [3, 5, 7]                       # Hard negative k values
  HARD_EPOCHS = [1, 2]                     # Hard negative epochs
  ```
- Temperature/margin grids are in `config.yaml`:
  ```yaml
  track_b:
    temperature_grid: [0.5, 0.7, 0.9, 1.1]
    margin_grid: [0.05, 0.1, 0.2, 0.3]
  ```

**Expected time**: ~60-90 minutes (depends on grid size)

#### **Option B: Quick Manual Test**  (Fast - 17 min)
Just enable hyperparameter sweep to optimize current model:
```yaml
# config.yaml
track_b:
  enable_hyperparam_sweep: true
```
```bash
python training/train_track_b.py
```
This tests temperature × margin combinations and picks the best.

#### **Option C: Manual Configuration**  (Custom)
Enable specific improvements in `config.yaml`:
```yaml
track_b:
  # Recommended baseline
  projection_dropout: 0.1
  projection_weight_decay: 0.06
  enable_hyperparam_sweep: true
  
  # High impact (if you have time)
  enable_hard_negatives: true
  hard_negative_k: 5
  hard_negative_epochs: 2
  
  # Optional
  enable_simcse: false
```
```bash
python training/train_track_b.py
```

**V10 Features**:
- Adds 512-dim projection head (reduces from 1024 to 512)
- Uses unified composite loss: MarginRanking + PairwiseSoftmax + MNR
- **NEW**: Grid search for regularization (dropout × weight decay)
- **NEW**: Grid search for hard negative mining (k × epochs)
- **NEW**: Hyperparameter sweep for optimal temperature & margin
- **NEW**: Optional SimCSE consistency loss

**Expected Accuracy Progression**:
- Baseline: 0.940
- + Best Regularization: 0.943-0.945
- + Hyperparam Sweep: 0.946-0.948
- + Best Hard Negatives: 0.950-0.955
- **Target: 0.950+** ✅

**If T4 OOM**: Lower `batch_size` in config, or switch `base_model` to `bge-base`.

**Data Augmentation**: If you ran `augment_training_data.py` and set `use_augmented_data: true`, training will automatically use the augmented dataset from `data/augmented/`.

### 4) Train Track A (MLP head over Track B embeddings)

**V10 Approach: Systematic Hyperparameter Tuning**

You have 2 options:

#### **Option A: Automated Experiment Runner**  (Recommended - Best Results)
Let the script systematically search for the best hyperparameters:
```bash
python scripts/run_track_a_experiments.py
```

**What it does**:
1. Backs up your current `config.yaml` to `config.yaml.backup`
2. Runs a **baseline** with current settings
3. Tests incremental improvements:
   - **Distillation sweep**: Tests different distillation weights (0.0, 0.3, 0.5)
   - **Head architecture sweep**: Tests 6 combinations of hidden_dim × dropout
   - **Learning rate sweep**: Tests different learning rates (5e-5, 1e-4, 2e-4)
   - **Joint fine-tuning**: Optional unfreezing of Track B
4. Only keeps experiments that improve validation accuracy
5. Saves all results in `experiments_track_a.json`
6. Writes the best configuration back into `config.yaml`

**Where to change search ranges**:
- Open `scripts/run_track_a_experiments.py`
- Edit the constants at the top:
  ```python
  HIDDEN_DIMS = [512, 1024]              # MLP hidden dimensions
  DROPOUTS = [0.1, 0.2, 0.3]             # MLP dropout values
  LEARNING_RATES = [5e-5, 1e-4, 2e-4]    # Learning rates to test
  DISTILL_WEIGHTS = [0.0, 0.3, 0.5]      # Distillation weights
  FREEZE_OPTIONS = [True, False]         # Whether to freeze Track B
  ```

**Expected time**: ~45-60 minutes (depends on grid size)

#### **Option B: Manual Training**  (Custom)
Train with current config settings:
```bash
python training/train_track_a.py
```

**V10 Features**:
- Lightweight MLP head (not cross-encoder)
- Operates on frozen Track B embeddings (or optionally fine-tuned)
- Pairwise features: `[e_anchor, e_cand, |e_anchor - e_cand|, e_anchor ⊙ e_cand]`
- **NEW**: Grid search for distillation weight
- **NEW**: Grid search for head architecture (hidden_dim × dropout)
- **NEW**: Grid search for learning rate
- **NEW**: Optional joint fine-tuning experiment
- K-fold cross-validation for robustness

**Requirements**: Track B must be trained first!

**Expected Accuracy**: 0.950-0.960

**Data Augmentation**: Same as Track B - will use augmented data if enabled.

### 5) Inference

#### Development Data (with labels - for validation)
```bash
python track_b.py   # embeddings, saves track_b.npy (512-dim) + reports accuracy
python track_a.py   # predictions, saves track_a.jsonl (uses Track B + MLP heads) + reports accuracy
```

**V2 Changes**:
- `track_b.py`: Verifies 512-dim embeddings from projection head
- `track_a.py`: Loads Track B model + MLP head ensemble (not cross-encoder)

#### Test Data (unlabeled - for submission)
For generating predictions on unlabeled test data:

```bash
# Track A: Generate predictions for test data
python inference_track_a.py --test_data data/test_track_a.jsonl --output output/track_a_test.jsonl --batch_size 16

# Track B: Generate embeddings for test data
python inference_track_b.py --test_data data/test_track_b.jsonl --output output/track_b_test.npy --batch_size 32
```

**Arguments**:
- `--test_data`: Path to test data JSONL file (required)
- `--output`: Output path (optional, defaults to `output/track_a_test.jsonl` or `output/track_b_test.npy`)
- `--batch_size`: Batch size for inference (optional, defaults: Track A=16, Track B=32)

**Format Requirements**:
- **Track A test data**: JSONL with columns `anchor_text`, `text_a`, `text_b` (no labels)
- **Track B test data**: JSONL with column `text` (no labels)

**Output Format**:
- **Track A**: JSONL file with `text_a_is_closer` predictions (boolean)
- **Track B**: NumPy `.npy` file with embeddings (shape: `[n_samples, 512]`)

**Note**: These scripts work both locally and in Google Colab

### 6) Evaluation
```bash
python scripts/eval_local.py --track both
```

### 7) Save results to Drive
```bash
mkdir -p /content/drive/MyDrive/narrative_similarity/output
cp -r output/* /content/drive/MyDrive/narrative_similarity/output/
```

---

## Track B Improvements Explained

### **Improvement 1: Better Regularization** (+0.3-0.5%)
- Higher weight decay for projection layer (0.08 vs 0.01)
- Adds constraints to prevent overfitting on small dataset
- Note: Projection dropout is currently handled via weight decay for improved stability

### **Improvement 2: Hyperparameter Sweep** (+0.5-1%)
- Tests temperature × margin combinations
- Finds optimal values for your data
- Only 2 minutes, high ROI

### **Improvement 3: Hard Negative Mining** (+1-2%)
- Mines difficult examples (similar but incorrect)
- Forces model to learn finer distinctions
- Highest impact improvement

### **Improvement 4: SimCSE** (+0.2-0.5%)
- Consistency loss between dropout views
- Acts as data augmentation
- Optional, use if still below target

---

## Data Augmentation Details

### Augmentation Strategies

**1. T5-Based Paraphrasing**:
- Model: `Vamsi/T5_Paraphrase_Paws`
- Generates semantically equivalent variations
- Parameters: `num_beams=5`, `temperature=1.2`

**2. Contextual Synonym Replacement**:
- WordNet-based substitution
- Preserves proper nouns and grammatical structure
- Parameters: `replacement_prob=0.15`, `max_replacements=3`

### Directory Structure

```
data/
├── prepared/           # Original prepared data (from prepare_data.py)
│   ├── triplets.jsonl
│   ├── pairs.jsonl
│   ├── cross_encoder_data.jsonl
│   ├── train_val_split.json
│   └── kfold_splits.json
└── augmented/          # Augmented data (from augment_training_data.py)
    ├── triplets.jsonl  # Original + 2x augmented = 3x total
    ├── pairs.jsonl
    └── cross_encoder_data.jsonl
```

**Important**: `augmented/` is a **peer directory** to `prepared/`, not a subdirectory.

### Configuration

```yaml
# config.yaml
augmentation:
  use_augmented_data: true  # Switch between prepared/ and augmented/
  
  strategies:
    paraphrasing:
      enable: true
      model: "Vamsi/T5_Paraphrase_Paws"
    synonym_replacement:
      enable: true
      replacement_prob: 0.15
  
  n_augmentations: 2  # Generate 2 versions per example
  min_similarity: 0.3  # Filter threshold
  max_similarity: 0.9
```

---

## Architecture Details

### Track B (Bi-Encoder with Projection Head)
```
Input Text → BGE-large-en-v1.5 (1024-dim) → Dense(512-dim) → Dropout → LayerNorm → L2 Normalize → 512-dim embedding
```

**Training**: Composite loss = MarginRanking + PairwiseSoftmax + MNR

**Improvements**:
- Hard negative mining (optional)
- Hyperparameter sweep (optional)
- Enhanced regularization

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

---

## Configuration Highlights

### Track B (`config.yaml`)
```yaml
track_b:
  # Architecture
  projection_dim: 512              # V2: Projection head dimension
  projection_dropout: 0.15         # Regularization
  projection_weight_decay: 0.08    # Higher for projection head
  
  # Loss
  loss_weights:
    margin: 1.0                    # MarginRankingLoss weight
    pairwise: 0.5                  # PairwiseSoftmaxLoss weight
    mnr: 0.5                       # MNR weight
  margin: 0.2                      # Margin for ranking loss
  temperature: 0.7                 # Temperature scaling
  
  # Improvements (all optional)
  enable_hard_negatives: false     # Set to true for hard negative mining
  hard_negative_k: 5               # Number of hard negatives
  hard_negative_epochs: 2          # Extra training epochs
  
  enable_hyperparam_sweep: true    # Post-training optimization
  temperature_grid: [0.5, 0.7, 1.0]
  margin_grid: [0.1, 0.2, 0.3]
  
  enable_simcse: false             # Consistency loss
  
  # Training
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

---

## Expected Performance

| Configuration | Track B | Track A | Total Time |
|--------------|---------|---------|------------|
| Baseline (v2) | 0.940 | 0.950 | 30 min |
| + Quick sweep | 0.946 | 0.950 | 32 min |
| + Full optimization | 0.952 | 0.950 | 60 min |
| **Target** | **0.950+** | **0.950+** | ✅ |

---

## Experiment Results

After running `python scripts/run_track_b_experiments.py`, check results:

**experiments_track_b.json**:
```json
{
  "experiments": [
    {"name": "Baseline", "accuracy": 0.940},
    {"name": "+ Regularization", "accuracy": 0.943},
    {"name": "+ Hyperparam Sweep", "accuracy": 0.946},
    {"name": "+ Hard Negatives", "accuracy": 0.952}
  ],
  "final_accuracy": 0.952,
  "target_reached": true
}
```

**models/track_b_embedder/best_hyperparams.json**:
```json
{
  "temperature": 0.7,
  "margin": 0.2,
  "accuracy": 0.952
}
```

---

## Troubleshooting

### Out of Memory (OOM)
```yaml
track_b:
  batch_size: 2  # Reduce from 4
  hard_negative_k: 3  # Reduce from 5
```

### Slow Training
```yaml
track_b:
  enable_hard_negatives: false  # Skip (saves 20 min)
  hard_negative_epochs: 1  # Reduce if enabled
```

### No Improvement
Try different hyperparameter ranges:
```yaml
track_b:
  temperature_grid: [0.3, 0.5, 0.7, 1.0]
  margin_grid: [0.05, 0.1, 0.15, 0.2, 0.3]
```

---

## Minimal Local Notes (optional)
- Python 3.11 venv; install PyTorch CUDA wheel matching your GPU; `pip install -r requirements.txt`.
- Run inference locally (slower): `python track_b.py`, `python track_a.py`, then `python scripts/eval_local.py --track both`.
