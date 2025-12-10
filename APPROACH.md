# Technical Approach for Narrative Similarity

This document describes the technical methodology and design decisions for both Track A and Track B of the Narrative Similarity task.

## Problem Statement

Given a narrative anchor story and two candidate stories (A and B), determine which candidate is more narratively similar to the anchor. The task is split into two tracks:

- **Track A**: Direct comparison approach (which story is closer?)
- **Track B**: Embedding-based approach (generate fixed-size representations)

## Overall Architecture

We implement a two-model approach optimized for the specific characteristics of each track:

1. **Track B - Bi-encoder (Embedding Model)**: Learns to encode stories into a semantic space where cosine similarity reflects narrative similarity
2. **Track A - Cross-encoder**: Directly models the relationship between anchor and candidates using cross-attention

## Track B: Bi-encoder Approach

### Model Architecture

**Base Model**: Sentence-Transformers (`all-MiniLM-L6-v2` by default)
- Pre-trained on semantic similarity tasks
- Efficient: 384-dimensional embeddings
- Fast inference: ~90MB model size

### Training Strategy

We use a **two-phase training approach** with metric learning:

#### Phase 1: Triplet Loss
```
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

- **Triplets**: (anchor, closer_story, farther_story)
- **Distance metric**: Cosine distance
- **Margin**: 0.5 (configurable)
- **Goal**: Push positive pairs closer, negative pairs farther

#### Phase 2: Multiple Negatives Ranking Loss
- **In-batch negatives**: Treats other positives in batch as hard negatives
- **Contrastive learning**: More efficient than explicit negative sampling
- **Benefits**: Better discriminative representations

### Data Preparation

From each (anchor, A, B, label) triple, we create:

1. **Triplet dataset**: 
   - If A is closer: (anchor, A, B)
   - If B is closer: (anchor, B, A)

2. **Pairwise dataset**:
   - (anchor, A, label_A)
   - (anchor, B, label_B)
   - Used for MultipleNegativesRankingLoss

### Training Configuration

```yaml
batch_size: 32
epochs: 10
learning_rate: 2e-5
optimizer: AdamW
warmup_steps: 100
mixed_precision: FP16
```

### Inference

1. Encode all stories independently: `emb = model.encode(story)`
2. For each triple, compute cosine similarities:
   - `sim_A = cosine(anchor_emb, A_emb)`
   - `sim_B = cosine(anchor_emb, B_emb)`
3. Predict: A is closer if `sim_A > sim_B`

**Output**: `.npy` file with shape `(N, 384)` containing all story embeddings

### Key Advantages

✅ **Fast inference**: Encode once, compare many times  
✅ **Scalable**: Linear complexity with number of stories  
✅ **Transfer learning**: Pre-trained semantic knowledge  
✅ **Interpretable**: Embeddings capture semantic similarity

## Track A: Cross-encoder Approach

### Model Architecture

**Base Model**: Transformer encoder (`roberta-base` by default)
- Pre-trained language model
- 125M parameters
- Strong contextual understanding

**Architecture**:
```
Input: [CLS] anchor [SEP] story_A [SEP] story_B [SEP]
         ↓
   Transformer Encoder (12 layers)
         ↓
   Classification Head (2 logits)
         ↓
   Prediction: argmax(logits)
         ↓
   Output: 1 if A is closer, 0 if B is closer
```

### Training Strategy

#### K-Fold Cross-Validation
- **5-fold CV** for robust evaluation
- Train separate model per fold
- Option for ensemble at inference

#### Loss Function
```
CrossEntropyLoss with Label Smoothing (α = 0.1)
```

Label smoothing prevents overconfidence:
- Hard label: [0, 1] or [1, 0]
- Smoothed: [0.05, 0.95] or [0.95, 0.05]

#### Data Augmentation
For each sample, we create augmented version by swapping A and B:
- Original: (anchor, A, B) → label=1 if A closer
- Augmented: (anchor, B, A) → label=0 if A was closer
- **Effect**: Doubles dataset size, removes position bias

### Training Configuration

```yaml
batch_size: 16
epochs: 5
learning_rate: 2e-5
optimizer: AdamW (with weight decay)
warmup_ratio: 0.1
max_length: 512 tokens
early_stopping: patience=3
```

### Training Loop

1. **Warmup**: Linear warmup for 10% of steps
2. **Learning rate scheduling**: Linear decay after warmup
3. **Gradient clipping**: Max norm = 1.0
4. **Mixed precision**: FP16 for memory efficiency
5. **Early stopping**: Stop if no improvement for 3 epochs
6. **Checkpointing**: Save best model based on validation accuracy

### Inference

#### Single Model Mode
1. Load best model from training
2. For each triple: `input = "[CLS] anchor [SEP] A [SEP] B [SEP]"`
3. Forward pass: `logits = model(input)`
4. Predict: `pred = argmax(logits)`

#### Ensemble Mode (Optional)
1. Load all 5 fold models
2. Get predictions from each model
3. **Majority voting** or **logit averaging**
4. Final prediction

**Memory Optimization**:
- Batch size: 4 (for 6GB GPUs)
- Clear CUDA cache after each batch
- Sequential processing for ensemble

**Output**: `.jsonl` file with predictions per triple

### Key Advantages

✅ **High accuracy**: Cross-attention between all text pairs  
✅ **Rich interactions**: Models complex narrative relationships  
✅ **End-to-end**: Direct optimization for comparison task  
✅ **Ensemble robustness**: K-fold reduces overfitting

## Comparison: Track A vs Track B

| Aspect | Track B (Bi-encoder) | Track A (Cross-encoder) |
|--------|---------------------|------------------------|
| **Architecture** | Separate encoding | Joint encoding |
| **Complexity** | O(N) for N stories | O(1) per comparison |
| **Scalability** | High | Low |
| **Accuracy** | Good (65-75%) | Better (75-85%) |
| **Speed** | Very fast | Moderate |
| **Memory** | Low (4-6GB) | Higher (6-8GB) |
| **Use case** | Retrieval, clustering | Direct comparison |

## Design Decisions

### Why Sentence-Transformers for Track B?

1. **Pre-trained on similarity tasks**: Already understands semantic similarity
2. **Efficient architecture**: Small model, fast inference
3. **Metric learning**: Built-in support for TripletLoss, MNR
4. **Production-ready**: Well-tested, stable library

### Why RoBERTa for Track A?

1. **Strong baseline**: Robust pre-trained model
2. **Stable tokenizer**: Fewer compatibility issues than DeBERTa
3. **Good performance/size tradeoff**: 125M parameters
4. **Wide support**: Excellent Hugging Face integration

### Why K-Fold Cross-Validation?

1. **Limited data**: Dev set may be small
2. **Robust evaluation**: Reduces variance in performance estimates
3. **Ensemble option**: Can combine models for better accuracy
4. **Debugging**: See model consistency across folds

### Why Two-Phase Training for Track B?

1. **Triplet Loss**: Establishes basic similarity structure
2. **MNR Loss**: Refines with hard negatives from batch
3. **Progressive learning**: Coarse-to-fine similarity learning

### Why Data Augmentation (A/B Swap)?

1. **Position bias**: Model might learn position instead of content
2. **Double data**: More training signal from limited data
3. **Symmetry**: Forces model to focus on content, not order

## Implementation Details

### Memory Optimizations

For **6GB GPUs**:
- Mixed precision (FP16) training
- Gradient accumulation (if needed)
- Reduced batch sizes: 16 (training), 4 (inference)
- CUDA cache clearing after each batch
- Sequential ensemble processing

### Reproducibility

- **Fixed seeds**: `seed=42` throughout
- **Deterministic ops**: Where possible
- **Version pinning**: requirements.txt specifies versions
- **Configuration-driven**: All hyperparameters in config.yaml

### Error Handling

1. **Missing models**: Graceful fallback to baseline
2. **Tokenizer issues**: Fallback to slow tokenizer
3. **OOM errors**: Automatic batch size reduction
4. **Import errors**: Use torch.optim instead of transformers

## Evaluation Metrics

### Primary Metric: Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### Additional Metrics
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives  
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### Validation Strategy

- **Track B**: Evaluated on Track A dev set during training
- **Track A**: K-fold CV with held-out validation per fold
- **Final evaluation**: Both tracks on Track A ground truth

## Performance Expectations

Based on architecture and data characteristics:

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| Random Baseline | 50% | - |
| Baseline SBERT | ~60% | - |
| **Fine-tuned Track B** | **65-75%** | 15-30 min |
| **Fine-tuned Track A** | **75-85%** | 45-90 min |
| **Track A Ensemble** | **78-87%** | 45-90 min |

**Improvement**: 25-35% accuracy gain over baseline

## Future Improvements

### Short-term
- [ ] Hyperparameter tuning (learning rate, margin, epochs)
- [ ] Try larger models (RoBERTa-large, DeBERTa-v3-base)
- [ ] Test alternative embedding models (BGE, E5)
- [ ] Implement focal loss for hard examples

### Medium-term
- [ ] Distillation from GPT-4 (teacher model)
- [ ] Hard negative mining for Track B
- [ ] Curriculum learning (easy → hard samples)
- [ ] Multi-task learning (combine tracks)

### Long-term
- [ ] Custom narrative-aware pre-training
- [ ] Graph neural networks for story structure
- [ ] Attention visualization for interpretability
- [ ] Model compression (quantization, pruning)

## Conclusion

This approach combines the **efficiency of bi-encoders** (Track B) with the **accuracy of cross-encoders** (Track A), providing complementary solutions to the narrative similarity task.

**Key strengths**:
- Metric learning for semantic spaces
- Cross-attention for complex comparisons
- Robust validation with k-fold CV
- Memory-efficient for consumer GPUs
- Production-ready with fallback mechanisms

**Expected outcome**: Significant improvement over baseline with practical inference times on modest hardware.

