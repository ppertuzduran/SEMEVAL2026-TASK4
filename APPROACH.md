# APPROACH v11 – Qwen3-Embedding-0.6B Backbone Upgrade

## Overview

This document describes **APPROACH v11**, an updated methodology for the Codabench narrative similarity
competition.
The central change in this version is the **replacement of the embedding backbone**
`BAAI/bge-large-en-v1.5` with **Qwen3-Embedding-0.6B**.

This model is selected as the *initial and primary backbone* to balance:
- Embedding quality
- Training stability
- GPU memory constraints
- Faster experimentation cycles

All other components of the pipeline (training logic, losses, evaluation, and inference format)
are intentionally kept consistent to ensure that observed performance changes are attributable
to the backbone upgrade.

---

## Motivation for the Backbone Change

After extensive experimentation with the previous approach, including:

- Hard-negative mining
- SimCSE-style objectives
- Curriculum learning
- Data augmentation
- Cross-encoder and distillation-based models

the system reached a stable performance plateau:
- **Track A:** ~0.95 accuracy
- **Track B:** ~0.94 accuracy

These results indicate that the current bottleneck is no longer the training strategy but the
**representational capacity of the embedding backbone**.

### Why Qwen3-Embedding-0.6B?

Qwen3-Embedding models represent the latest generation of dedicated embedding architectures.
The **0.6B variant** provides a strong trade-off between quality and efficiency:

- Significantly stronger semantic representations than older BGE models
- Designed explicitly for embedding tasks (not instruction tuning)
- Competitive performance on MTEB English benchmarks
- Lower memory footprint than 4B/8B variants
- Suitable for rapid iteration and stable fine-tuning

This makes Qwen3-Embedding-0.6B an ideal backbone for this competition stage.

---

## Selected Model

- **Model:** `Qwen/Qwen3-Embedding-0.6B`
- **Context length:** up to 8192 tokens
- **Embedding objective:** semantic similarity
- **Inference compatibility:** cosine similarity

---

## Track B: Embedding Model Training

### Objective

Produce a vector representation for each individual story such that cosine similarity aligns
with narrative similarity, while strictly respecting the rule that **embeddings are generated
independently per story at inference time**.

### Architecture

- Base encoder: Qwen3-Embedding-0.6B
- Pooling strategy: **Last-token pooling** (recommended by the model authors)
- Projection head:
  - Shallow linear or MLP projection to a fixed embedding dimension
  - L2 normalization applied to final embeddings

```
Story Text
   ↓
Qwen3 Encoder
   ↓
Last Token Representation
   ↓
Projection Head
   ↓
L2 Normalization
   ↓
Final Embedding
```

### Training Strategy

The Track B training pipeline reuses the existing implementation with minimal modifications:

- Pairwise supervision derived from triples
- Triple-wise margin ranking
- Cosine similarity as the primary metric
- Early stopping based on local triple accuracy

To improve generalization, the loss design is intentionally **simple and conservative**:

- Pairwise softmax loss
- Light margin ranking loss
- No aggressive hard-negative mining
- No SimCSE-style self-supervision

This choice is motivated by empirical findings that excessive loss complexity degraded performance.

---

## Track A: Pairwise Similarity Decision

### Objective

Given a triple consisting of an anchor story and two candidate stories (A, B), determine which
candidate is more similar to the anchor.

### Architecture

Track A leverages the embeddings produced by Track B:

1. Independently embed:
   - Anchor
   - Candidate A
   - Candidate B
2. Construct pairwise feature vectors:
   - `[e_anchor, e_candidate, |e_anchor − e_candidate|, e_anchor ⊙ e_candidate]`
3. Pass features through a lightweight MLP classifier
4. Output logits corresponding to candidate A vs B

This architecture was chosen due to its robustness under limited supervision and its ability
to exploit high-quality embeddings effectively.

### Robustness Enhancements

- Multiple MLP heads trained with different random seeds
- Logit averaging across heads at inference time
- Optional temperature scaling on validation data

---

## Inference

### Track B

- Each story is encoded **once and independently**
- The output is a single fixed-size vector per story
- Cosine similarity is used for all similarity computations

### Track A

- For each triple:
  - Compute embeddings for anchor, A, and B
  - Apply the MLP classifier
  - Select the candidate with the higher predicted similarity

All inference outputs strictly follow Codabench submission specifications.

---

## Migration Checklist (from BGE-large to Qwen3-Embedding-0.6B)

1. Replace the backbone model name:
   ```python
   model_name = "Qwen/Qwen3-Embedding-0.6B"
   ```
2. Update pooling logic to use **last-token pooling**
3. Reduce batch size if necessary (depending on GPU memory)
4. Keep loss functions and evaluation logic unchanged
5. Re-train Track B, then re-train Track A heads on the new embeddings

---

## Expected Benefits

- Improved semantic alignment for narrative similarity
- Better generalization on unseen triples
- Reduced sensitivity to noisy augmentation
- Faster iteration cycles compared to larger backbones
- A stronger foundation for future scaling (4B / 8B variants)

---

## Summary of Changes

| Component | Previous | v11 |
|---------|---------|----|
| Backbone | BAAI/bge-large-en-v1.5 | Qwen3-Embedding-0.6B |
| Pooling | CLS / mean | Last-token |
| Loss design | Complex | Simplified |
| Track A model | MLP on embeddings | Same (stronger embeddings) |
| Inference rules | Unchanged | Unchanged |

---

## Final Notes

APPROACH v11 focuses on **raising the embedding quality ceiling** while preserving a proven,
stable training and inference pipeline. By upgrading to Qwen3-Embedding-0.6B, this approach
aims to unlock additional performance gains with minimal architectural risk and full
compliance with competition constraints.
