# State-of-the-Art Techniques for Narrative Similarity

This document describes the advanced techniques used for training both Track A and Track B models.

---

## Track A: Pairwise Scoring with Cross-Encoders

### 1. Pairwise Scoring Architecture

Instead of joint multi-text classification, we employ **pairwise scoring** followed by softmax normalization:

```
For triple (anchor, story_A, story_B):
  score_A = CrossEncoder(anchor, story_A)
  score_B = CrossEncoder(anchor, story_B)
  P(A closer) = exp(score_A) / (exp(score_A) + exp(score_B))
```

**Technique Origin**: Standard approach in information retrieval reranking (Nogueira & Cho, 2019; Gao et al., 2021)

**Why SOTA**: 
- Decouples scoring from comparison
- More stable optimization landscape
- Better probability calibration
- Aligns with listwise ranking objectives

### 2. Large Pre-trained Language Models

**Model**: RoBERTa-large (355M parameters)

**Technique**: Fine-tuning large transformers for semantic similarity
- Pre-trained on massive text corpora
- Deep bidirectional attention
- Rich contextualized representations

**Research Basis**: Liu et al. (2019) - RoBERTa optimization

### 3. Cross-Entropy with Softmax Over Pairs

**Loss Function**:
```
L = -log(softmax([score_A, score_B])[y])
```
where y ∈ {0, 1} indicates which story is closer.

**Technique**: Pairwise ranking loss with probabilistic interpretation
- Standard in learning-to-rank
- Direct optimization of comparison task
- Smooth gradients for stable training

### 4. K-Fold Cross-Validation

**Technique**: 5-fold stratified cross-validation
- Reduces variance in performance estimation
- Enables model ensemble (optional)
- Provides robust generalization metrics

**Research Basis**: Standard ML practice for limited data scenarios

---

## Track B: Metric Learning for Embeddings

### 1. Large Pre-trained Embedding Models

**Model**: BAAI/bge-base-en-v1.5 (109M parameters, 768-dimensional)

**Technique**: Specialized embedding models pre-trained for retrieval
- Optimized on large-scale semantic similarity datasets
- Contrastive pre-training objectives
- Superior to general sentence encoders

**Research Basis**: Xiao et al. (2023) - BGE: Better General Embeddings

### 2. Direct Metric Optimization via Pairwise Softmax

**Novel Loss**: Triple-wise contrastive softmax

```python
# Encode all three texts independently
e_anchor, e_A, e_B = Encoder(anchor), Encoder(A), Encoder(B)

# Compute similarities
sim_A = cosine(e_anchor, e_A)
sim_B = cosine(e_anchor, e_B)

# Apply softmax cross-entropy
L = -log(softmax([sim_A, sim_B])[y])
```

**Technique Origin**: Direct adaptation of pairwise ranking to embedding space

**Why SOTA**:
- Training objective = evaluation metric (cosine similarity comparison)
- No proxy loss (e.g., triplet margin)
- End-to-end differentiable
- Faster convergence to task objective

### 3. Multi-Phase Training Strategy

**Implementation Note**: The training uses **sequential phases**, not combined weighted losses. Each phase trains independently, loading the best model from the previous phase.

**Phase 1: Pairwise Softmax** (Direct Metric Optimization)
- Optimizes exactly what will be evaluated
- Establishes task-specific similarity structure

**Phase 2: TripletLoss** (Metric Learning)
```
L = max(0, d(anchor, pos) - d(anchor, neg) + margin)
```
- Refines embedding geometry
- Hard negative focus
- Improves discriminative power
- Lower learning rate (halved) and warmup steps (halved) for stability

**Phase 3: Multiple Negatives Ranking Loss** (Contrastive Learning)
```
L = -log(exp(sim_pos) / Σ_i exp(sim_neg_i))
```
- In-batch negatives as hard negatives
- Scales efficiently with batch size
- Further separates similar/dissimilar pairs
- Continues with reduced learning rate and warmup steps

**Technique**: Progressive curriculum from task-specific to representation-quality objectives

**Research Basis**: 
- TripletLoss: Schroff et al. (2015) - FaceNet
- MNR: Henderson et al. (2017) - Efficient Natural Language Response Suggestion
- Multi-phase: Common in transfer learning (Howard & Ruder, 2018)

### 4. Cosine Similarity as Semantic Distance

**Technique**: Cosine distance in normalized embedding space
- Scale-invariant
- Bounded [0, 2]
- Standard in semantic similarity tasks

**Mathematical Property**:
```
cosine(u, v) = u·v / (||u|| ||v||)
```
Captures angular distance, invariant to magnitude.

---

## Advanced Optimization Techniques

### 1. Mixed Precision Training (FP16)

**Technique**: Automatic Mixed Precision (AMP)
- Forward/backward in float16
- Master weights in float32
- Loss scaling for numerical stability

**Benefits**: 2x memory reduction, ~1.5-2x speedup, minimal accuracy impact

**Research**: Micikevicius et al. (2018) - Mixed Precision Training

### 2. Gradient Clipping

**Technique**: Clip gradient L2 norm to maximum value (1.0)
```
if ||g|| > threshold: g ← g * (threshold / ||g||)
```

**Purpose**: Prevents exploding gradients in deep networks

### 3. Learning Rate Scheduling

**Warmup + Linear Decay**:
```
lr(t) = lr_max * min(t/t_warmup, (T-t)/(T-t_warmup))
```

**Technique Origin**: Vaswani et al. (2017) - Transformer training
- Warmup: gradual increase (10% of steps)
- Decay: linear decrease to 0
- Stabilizes early training, improves convergence

### 4. Weight Decay (AdamW)

**Technique**: Decoupled weight decay optimization
```
w_t = w_{t-1} - lr * (∇L + λ * w_{t-1})
```

**Research**: Loshchilov & Hutter (2019) - Decoupled Weight Decay Regularization
- Prevents overfitting
- More effective than L2 regularization with Adam

---

## Data Augmentation Techniques

### 1. A/B Position Swap

**Technique**: Symmetric data augmentation
```
Original:  (anchor, A, B) → label = 1 if A closer
Augmented: (anchor, B, A) → label = 0 if A closer
```

**Purpose**: 
- Removes position bias
- Doubles training data
- Enforces order invariance

### 2. Label Smoothing Removal

**Technique**: Hard targets instead of smoothed
- Original: [0.95, 0.05] for positive class
- Implemented: [1.0, 0.0] for positive class

**Rationale**: Clear signal for pairwise comparison (less ambiguity than multi-class)

---

## Key State-of-the-Art Principles Applied

### 1. **Task-Specific Fine-tuning**
Pre-trained models adapted to narrative similarity via supervised fine-tuning on domain data.

### 2. **Pairwise Ranking Paradigm**
Standard in modern IR systems (MS MARCO, TREC DL). Proven superior to pointwise classification for comparison tasks.

### 3. **Metric Learning with Deep Embeddings**
Learn semantic spaces where geometric distance reflects similarity. Foundation of modern retrieval systems.

### 4. **Direct Metric Optimization**
Align training loss with evaluation metric to avoid proxy objective issues.

### 5. **Large-Scale Pre-training**
Leverage models trained on billions of tokens (RoBERTa) and million-scale retrieval datasets (BGE).

### 6. **Multi-Phase Progressive Training**
Curriculum learning from task-specific to general representation quality.

### 7. **Efficient Mixed-Precision Training**
Industry-standard technique for training large models on limited hardware.

---

## Research References

### Core Techniques
- **Pairwise Ranking**: Burges et al. (2005) - Learning to Rank using Gradient Descent
- **Cross-Encoders**: Humeau et al. (2020) - Poly-encoders: Architectures and Pre-training Strategies
- **Metric Learning**: Kaya & Bilge (2019) - Deep Metric Learning: A Survey
- **BGE Embeddings**: Xiao et al. (2023) - C-Pack: Packaged Resources To Advance General Chinese Embedding

### Optimization
- **AdamW**: Loshchilov & Hutter (2019) - Decoupled Weight Decay Regularization
- **Mixed Precision**: Micikevicius et al. (2018) - Mixed Precision Training
- **Learning Rate Warmup**: Goyal et al. (2017) - Accurate, Large Minibatch SGD

### Pre-trained Models
- **RoBERTa**: Liu et al. (2019) - RoBERTa: A Robustly Optimized BERT Pretraining Approach
- **Sentence-BERT**: Reimers & Gurevych (2019) - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

### Loss Functions
- **TripletLoss**: Schroff et al. (2015) - FaceNet: A Unified Embedding for Face Recognition
- **MNR Loss**: Henderson et al. (2017) - Efficient Natural Language Response Suggestion
- **Contrastive Learning**: Chen et al. (2020) - A Simple Framework for Contrastive Learning

---

---

## Training Implementation Details

### Track A: Cross-Encoder Training

**Model Configuration**:
```yaml
Base Model: roberta-large (355M parameters)
Max Sequence Length: 512 tokens per pair
Number of Labels: 2 (binary classification)
```

**Training Hyperparameters**:
```yaml
Optimizer: AdamW
Learning Rate: 1e-5
Weight Decay: 0.01
Warmup Ratio: 0.1 (10% of total steps)
Batch Size: 6 (reduced for pairwise = 2x inputs)
Epochs: 5
Gradient Clipping: 1.0
Mixed Precision: FP16
Label Smoothing: 0.0
```

**Cross-Validation**:
```yaml
Strategy: 5-fold stratified
Train/Val Split: 80/20 per fold
Early Stopping: patience=3 epochs
Save Strategy: Best model per fold
```

**Memory Management**:
- **Gradient Accumulation**: Not used (batch size feasible on 6GB GPU)
- **CUDA Cache Clearing**: After each validation epoch
- **Sequential Pair Processing**: During inference to reduce peak memory
- **Batch Size Calculation**: `effective_batch = 6 samples × 2 pairs = 12 forward passes`

**Training Time** (RTX 4050, 6GB):
- Per epoch: ~8-10 minutes
- Per fold: ~40-50 minutes
- Total (5 folds): ~3.5-4 hours

### Track B: Bi-Encoder Training

**Model Configuration**:
```yaml
Base Model: BAAI/bge-base-en-v1.5 (109M parameters)
Embedding Dimension: 768
Pooling Strategy: CLS token
Normalization: L2 normalization
```

**Training Hyperparameters**:
```yaml
Optimizer: AdamW
Learning Rate: 1e-5
Weight Decay: 0.01
Warmup Steps: 100
Batch Size: 24 (reduced from 32 for larger model)
Base Epochs: 10 (from config, but split across phases results in 15 total)
Gradient Clipping: 1.0
Mixed Precision: FP16
```

**Multi-Phase Training Schedule**:
```yaml
Phase 1 (Pairwise Softmax):
  Epochs: 5 (num_epochs // 2)
  Learning Rate: 1e-5
  Warmup Steps: 100
  Objective: Direct metric optimization

Phase 2 (TripletLoss):
  Epochs: 5 (num_epochs // 2)
  Learning Rate: 5e-6 (halved for stability)
  Warmup Steps: 50 (halved)
  Triplet Margin: 0.5
  Distance Metric: Cosine

Phase 3 (MNR - Optional):
  Epochs: 5 (num_epochs // 2, if enabled)
  Learning Rate: 5e-6
  Warmup Steps: 50 (halved)
  In-Batch Negatives: 23 (batch_size - 1)

Total Training Epochs: 15 (when all phases enabled)
```

**Data Split**:
```yaml
Train: 80% (160 samples)
Validation: 20% (40 samples)
Evaluation: Track A dev set (200 samples)
```

**Memory Management**:
- **Gradient Checkpointing**: Not required for BGE-base
- **Dynamic Padding**: Max length in batch (not global max)
- **Embedding Cache**: Cleared after evaluation
- **Model Saving**: Only best checkpoint retained

**Training Time** (RTX 4050, 6GB):
- Phase 1: ~12-15 minutes (5 epochs)
- Phase 2: ~12-15 minutes (5 epochs)
- Phase 3: ~12-15 minutes (5 epochs, if enabled)
- Total: ~36-45 minutes (15 total epochs when all phases enabled)

### Inference Implementation

**Track A Inference** (Pairwise Scoring):
```yaml
Batch Size: 4 (processes 8 pairs per batch)
Processing Strategy: Sequential pair batches
  - Batch A pairs → compute scores → free memory
  - Batch B pairs → compute scores → free memory
  - Compare scores → predictions

Memory Optimization:
  - CUDA cache clear after each pair batch
  - Detach gradients before numpy conversion
  - Delete intermediate tensors explicitly

Throughput: ~50-60 samples/minute (RTX 4050)
```

**Track B Inference** (Embedding Generation):
```yaml
Batch Size: 32 (standard bi-encoder inference)
Processing: Single forward pass per story
Output: (N, 768) embedding matrix

Memory Usage: ~2-3GB for 200 samples
Throughput: ~100-150 samples/minute
```

### Hardware Requirements

**Minimum Specifications**:
```yaml
GPU: 6GB VRAM (RTX 4050, RTX 2060, T4)
RAM: 12GB system memory
Storage: ~2GB for models + data
```

**Optimal Specifications**:
```yaml
GPU: 16GB VRAM (RTX 4090, V100, A100)
RAM: 32GB system memory

Benefits:
  - Track A batch_size: 6→16 (2.5x faster)
  - Track B batch_size: 24→48 (2x faster)
  - Full precision training option (FP32)
```

### Batch Size Scaling Guidelines

**Track A** (pairwise = 2x memory per sample):
```
6GB GPU:  batch_size = 6  (12 pairs)
12GB GPU: batch_size = 12 (24 pairs)
24GB GPU: batch_size = 24 (48 pairs)
```

**Track B** (independent encoding):
```
6GB GPU:  batch_size = 24
12GB GPU: batch_size = 48
24GB GPU: batch_size = 96
```

### Reproducibility Settings

```yaml
Random Seed: 42
  - Python random.seed(42)
  - NumPy np.random.seed(42)
  - PyTorch torch.manual_seed(42)
  - CUDA torch.cuda.manual_seed_all(42)

Deterministic Operations:
  - cudnn settings: Not explicitly set (uses PyTorch defaults)
  - Note: Full reproducibility not enforced for performance reasons

Version Pinning:
  - transformers >= 4.35.0
  - sentence-transformers >= 2.2.2
  - torch >= 2.0.0
```

### Loss Computation Details

**Track A - Pairwise Cross-Entropy**:
```python
# Forward passes
logits_a = model(anchor, text_a)  # Shape: (batch, 2)
logits_b = model(anchor, text_b)  # Shape: (batch, 2)

# Extract scores (positive class logit)
score_a = logits_a[:, 1]  # Shape: (batch,)
score_b = logits_b[:, 1]  # Shape: (batch,)

# Stack and apply cross-entropy
scores = torch.stack([score_a, score_b], dim=1)  # (batch, 2)
targets = (1 - labels).long()  # Invert: 1→0, 0→1
loss = CrossEntropyLoss(scores, targets)
```

**Track B - Pairwise Softmax**:
```python
# Encode independently
anchor_emb = model.encode(anchor)    # (batch, 768)
a_emb = model.encode(text_a)         # (batch, 768)
b_emb = model.encode(text_b)         # (batch, 768)

# Cosine similarities
sim_a = F.cosine_similarity(anchor_emb, a_emb)  # (batch,)
sim_b = F.cosine_similarity(anchor_emb, b_emb)  # (batch,)

# Stack and apply cross-entropy
sims = torch.stack([sim_a, sim_b], dim=1)  # (batch, 2)
targets = (1 - labels).long()
loss = CrossEntropyLoss(sims, targets)
```

---

## Summary

The approach combines **four pillars of modern NLP**:

1. **Large Pre-trained Models** (RoBERTa-large, BGE-base)
2. **Task-Specific Architecture** (Pairwise scoring, metric learning)
3. **Direct Optimization** (Training = evaluation metric)
4. **Robust Training** (Multi-phase, mixed precision, k-fold)

**Implementation**: Optimized for consumer hardware (6GB GPU) with careful memory management, mixed precision, and efficient batch processing.

All techniques are **proven, published, and widely adopted** in production systems and research benchmarks.
