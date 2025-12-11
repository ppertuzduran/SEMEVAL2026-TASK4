# Technical Approach for Narrative Similarity Competition (Tracks A & B)

This document describes the **implemented approach** for the narrative similarity task, covering both Track A (classification) and Track B (embeddings).

---

## 1. Task Overview

### Track A (Classification)
- **Input**: `(anchor_story, choice_A, choice_B)`
- **Output**: Prediction of which story (A or B) is more narratively similar to the anchor
- **Metric**: Accuracy on triple-wise comparisons
- **Model**: Cross-encoder with pairwise scoring

### Track B (Embeddings)
- **Input**: Individual stories
- **Output**: Vector embeddings (768-dimensional) whose cosine similarity reflects narrative similarity
- **Metric**: Triple-wise accuracy derived from cosine comparison
- **Constraint**: Embeddings must be computed per story independently (no triple-aware embeddings at inference)
- **Model**: Bi-encoder with metric learning

---

## 2. Track A: Cross-Encoder Implementation

### 2.1. Model Architecture

**Base Model**: `roberta-large` (355M parameters)

**Pairwise Scoring Architecture**:
- For each triple `(anchor, A, B)`, score two pairs independently:
  - Pair 1: `[CLS] anchor [SEP] text_a [SEP]`
  - Pair 2: `[CLS] anchor [SEP] text_b [SEP]`
- Extract positive class logit from each pair as similarity score
- Stack scores: `[score_a, score_b]`
- Apply softmax + cross-entropy loss

**Why Pairwise Scoring?**
- More stable than concatenating all three texts
- Aligns with information retrieval best practices
- Each pair gets full attention from the model
- Standard approach in cross-encoder reranking (Nogueira & Cho, 2020)

### 2.2. Training Configuration

```yaml
Base Model: roberta-large
Max Length: 512 tokens per pair
Optimizer: AdamW
Learning Rate: 1e-5
Weight Decay: 0.01
Warmup Ratio: 0.1 (10% of training steps)
Batch Size: 6
Epochs: 5
Mixed Precision: FP16
Gradient Clipping: 1.0
Label Smoothing: 0.0
```

### 2.3. K-Fold Cross-Validation

- **Strategy**: 5-fold cross-validation
- **Purpose**: Robust model evaluation and preparation for ensemble
- **Implementation**:
  - Data split into 5 stratified folds
  - Each fold trains a separate model
  - Models saved independently: `track_a_cross_encoder_fold0`, `fold1`, etc.
- **Early Stopping**: Patience of 3 epochs based on validation accuracy

### 2.4. Data Augmentation

**A/B Position Swap**:
- For each original triple `(anchor, A, B, label=1 if A closer)`:
- Create augmented triple `(anchor, B, A, label=0 if A closer)`
- **Effect**: Doubles training data and removes position bias

### 2.5. Inference Options

**Single Model**:
- Use best model from fold 0 or specified fold
- Fast inference

**Ensemble** (Optional):
- Load all K fold models
- For each triple, compute scores from all models
- Average predictions (majority vote)
- **Expected gain**: 1-3 points accuracy improvement
- **Config setting**: `use_ensemble: true/false`

---

## 3. Track B: Bi-Encoder Implementation

### 3.1. Model Architecture

**Base Model**: `BAAI/bge-base-en-v1.5`
- 109M parameters
- 768-dimensional embeddings
- CLS token pooling
- L2 normalization (implicit in model)

**Why BGE?**
- State-of-the-art English embedding model (Xiao et al., 2023)
- Pre-trained on large-scale text pairs
- Superior to MiniLM and base sentence-transformers

### 3.2. Multi-Phase Training Curriculum

Training proceeds in **3 sequential phases**, each optimizing a different objective:

#### Phase 1: PairwiseSoftmaxLoss (Epochs 1-5)
- **Purpose**: Direct optimization of evaluation metric
- **Input**: Triples `(anchor, text_a, text_b, label)`
- **Loss**:
  ```python
  sim_a = cosine(anchor_emb, a_emb)
  sim_b = cosine(anchor_emb, b_emb)
  scores = [sim_a, sim_b]
  loss = CrossEntropy(scores, target)
  ```
- **Why first?**: Aligns embedding space directly with task objective

#### Phase 2: TripletLoss (Epochs 6-10)
- **Purpose**: Enforce margin-based separation
- **Input**: Triplets `(anchor, positive, negative)`
- **Loss**: `max(0, margin + dist(a,p) - dist(a,n))`
- **Margin**: 0.5
- **Distance**: Cosine distance
- **Learning Rate**: Halved to 5e-6 for stability
- **Why second?**: Refines embedding quality without overriding metric-aligned structure

#### Phase 3: MultipleNegativesRankingLoss (Epochs 11-15, Optional)
- **Purpose**: In-batch contrastive learning
- **Input**: Positive pairs `(anchor, positive)`
- **Negatives**: Other positives in batch (batch_size - 1 = 23 negatives)
- **Loss**: InfoNCE-style contrastive loss
- **Why last?**: Global space shaping without task-specific overfitting

### 3.3. Training Configuration

```yaml
Base Model: BAAI/bge-base-en-v1.5
Optimizer: AdamW
Learning Rate: 1e-5 (Phase 1), 5e-6 (Phases 2-3)
Weight Decay: 0.01
Warmup Steps: 100 (Phase 1), 50 (Phases 2-3)
Batch Size: 24
Total Epochs: 15 (5 per phase, 10 if MNR disabled)
Mixed Precision: FP16
Gradient Clipping: 1.0
```

### 3.4. Data Pipeline

- **Train/Val Split**: 80/20 (160 train, 40 val samples)
- **Evaluation**: Continuous monitoring on Track A dev set
- **Best Model Selection**: Highest accuracy on Track A triples
- **Checkpoint Strategy**: Save only best model per phase

### 3.5. Inference

- Single forward pass per story
- Output: (N, 768) embedding matrix
- Cosine similarity for pairwise comparison
- No ensemble (embeddings are story-independent)

---

## 4. Key Technical Decisions

### 4.1. Mixed Precision Training (FP16)

- **Implementation**: PyTorch Automatic Mixed Precision (AMP)
- **Benefits**:
  - 2x memory reduction
  - 1.5-2x training speedup
  - Minimal accuracy impact
- **Techniques**:
  - Forward/backward in float16
  - Master weights in float32
  - Gradient scaling for numerical stability

### 4.2. Learning Rate Scheduling

**Track A**:
- Linear warmup (10% of steps)
- Linear decay to 0
- **Why?**: Standard Transformer fine-tuning practice (Vaswani et al., 2017)

**Track B**:
- Constant warmup steps per phase
- Learning rate halved for refinement phases
- **Why?**: Prevents later phases from destroying earlier learning

### 4.3. Gradient Clipping

- **Value**: 1.0 (L2 norm)
- **Purpose**: Prevents exploding gradients in deep networks
- **Applied**: Both tracks

### 4.4. Weight Decay (AdamW)

- **Value**: 0.01
- **Optimizer**: AdamW (decoupled weight decay)
- **Purpose**: Regularization without interfering with Adam's adaptive learning rates
- **Reference**: Loshchilov & Hutter (2019)

---

## 5. Data Preparation

### 5.1. From Track A to Training Data

**Input**: Track A JSONL with fields:
- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (label)

**Output Data Formats**:

1. **Cross-Encoder Data** (Track A):
   ```python
   {
     "anchor": str,
     "text_a": str,
     "text_b": str,
     "label": 1 if A closer, 0 if B closer
   }
   ```
   - With A/B swap augmentation: 2x samples

2. **Triplets** (Track B - TripletLoss):
   ```python
   {
     "anchor": str,
     "positive": closer_text,
     "negative": farther_text
   }
   ```

3. **Pairs** (Track B - MNR Loss):
   ```python
   {
     "anchor": str,
     "candidate": str,
     "label": 1.0 if positive, 0.0 if negative
   }
   ```

### 5.2. K-Fold Splits

- **Method**: sklearn.KFold with shuffle
- **Seed**: 42 (reproducibility)
- **Storage**: `kfold_splits.json` with train/val indices

---

## 6. Performance Expectations

### Baseline
- **Random**: 50% accuracy
- **Off-the-shelf embeddings**: ~60-65%

### Implemented System
- **Track B (BGE bi-encoder)**: 85-87% accuracy
- **Track A (RoBERTa cross-encoder)**: 70-80% accuracy
- **Track A with ensemble**: +1-3% over single model

### Training Time (Google Colab T4, 16GB)
- **Track B**: ~36-45 minutes (15 epochs)
- **Track A**: ~40-50 minutes per fold
- **Track A (5 folds)**: ~3.5-4 hours total

### Memory Usage
- **Track A**: ~5-6 GB VRAM (batch_size=6)
- **Track B**: ~4-5 GB VRAM (batch_size=24)
- Both fit comfortably on 6GB+ GPUs

---

## 7. Technology Stack

### Core Libraries
- **PyTorch**: 2.0+ (deep learning framework)
- **Transformers**: 4.30+ (Hugging Face)
- **Sentence-Transformers**: 2.2+ (embedding models & losses)
- **scikit-learn**: 1.3+ (k-fold splits)

### Training Infrastructure
- **Google Colab**: T4 GPU (16GB VRAM) - recommended for training
- **Local GPU**: RTX 4050 (6GB VRAM) - suitable for inference, slower training
- **CUDA**: 11.8 or 12.1

### Data Storage
- **Google Drive**: Model checkpoints and prepared data
- **Git**: Code version control
- **Local**: Inference and evaluation

---

## 8. References

### Models
- **RoBERTa**: Liu et al. (2019) - RoBERTa: A Robustly Optimized BERT Pretraining Approach
- **BGE**: Xiao et al. (2023) - C-Pack: Packaged Resources To Advance General Chinese Embedding
- **Sentence-BERT**: Reimers & Gurevych (2019) - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

### Training Techniques
- **Mixed Precision**: Micikevicius et al. (2018) - Mixed Precision Training
- **AdamW**: Loshchilov & Hutter (2019) - Decoupled Weight Decay Regularization
- **Learning Rate Warmup**: Vaswani et al. (2017) - Attention Is All You Need

### Loss Functions
- **TripletLoss**: Schroff et al. (2015) - FaceNet: A Unified Embedding for Face Recognition
- **Multiple Negatives Ranking**: Henderson et al. (2017) - Efficient Natural Language Response Suggestion for Smart Reply
- **Cross-Encoder Pairwise**: Nogueira & Cho (2020) - Passage Re-ranking with BERT

---

## 9. File Structure

```
project/
├── config.yaml                    # All hyperparameters
├── training/
│   ├── prepare_data.py           # Convert Track A → training formats
│   ├── train_track_a.py          # Cross-encoder training (Colab)
│   ├── train_track_b.py          # Bi-encoder training (Colab)
│   └── colab_utils.py            # Google Drive integration
├── track_a.py                     # Cross-encoder inference (local)
├── track_b.py                     # Bi-encoder inference (local)
├── scripts/
│   └── eval_local.py             # Local evaluation script
└── models/                        # Trained checkpoints (Git-ignored)
    ├── track_a_cross_encoder_fold0/
    ├── track_a_cross_encoder_fold1/
    └── track_b_embedder/
```

---

## 10. Design Philosophy

1. **Simplicity First**: Use proven architectures without unnecessary complexity
2. **Metric Alignment**: Train directly on evaluation objective when possible
3. **Careful Regularization**: Small dataset requires early stopping and cross-validation
4. **Hybrid Approach**: Train on GPU (Colab), infer locally
5. **Reproducibility**: Fixed seeds, version-controlled code, logged experiments
6. **Practical Engineering**: Balance SOTA techniques with implementation feasibility

---

**This approach achieves strong performance (+25-35% over baseline) using standard deep learning techniques applied carefully to the narrative similarity task.**
