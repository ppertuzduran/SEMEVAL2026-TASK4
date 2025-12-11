# Improved Approach for Track A and Track B

## Overview

This document outlines a refined, state-of-the-art approach for improving performance on both Track A (pairwise narrative similarity classification) and Track B (narrative embedding learning) of the competition.

---

# Track B – Embedding Model (Current: 0.82 Accuracy)

## Goals
- Produce embeddings whose cosine similarities reflect narrative similarity.
- Improve over MiniLM-based baseline while respecting rule: **embeddings must be produced per story independently**.

## Improvements

### 1. Upgrade the Backbone
Replace `all-MiniLM-L6-v2` with stronger embedding architectures:
- **BGE-base / BGE-large**
- **E5-base-v2 / E5-large-v2**
- **GTE-base / GTE-large**

These models consistently outperform MiniLM in semantic similarity and retrieval tasks.

---

### 2. Use Pairwise/Listwise Loss Directly from Triples
Triplet loss + MNRL works, but evaluation is *triple-wise*. Align the training objective:

For each triple (anchor, A, B, label):
1. Encode independently:  
   `e_anchor, e_A, e_B`
2. Compute similarities:  
   `s_A = cos(e_anchor, e_A)`  
   `s_B = cos(e_anchor, e_B)`
3. Apply pairwise softmax:  
   `P(A closer) = softmax([s_A, s_B])`

Loss: Cross-entropy with the true label.

This makes the embedding model directly optimize the evaluation metric.

---

### 3. Hard Negative Mining
Use first-stage embeddings to retrieve *near-miss* negatives for each anchor. Add them as:
- Extra contrastive negatives
- Additional triplets (anchor, positive, hard negative)

Hard negatives significantly boost retrieval-style tasks.

---

### 4. Optional: LLM Distillation for Smoother Targets
Query GPT-4/4o for similarity *scores* to create soft labels:

- Distill pairwise similarity distributions
- Regress cosine similarity toward teacher-provided scales  
  (e.g., 0–5 Likert similarity)

Soft supervision stabilizes training and increases generalization.

---

# Track A – Pairwise Ranking Model (Current: 0.55 Accuracy)

## Goals
Classify which story (A or B) is closer to the anchor.

### Issues Identified
- Using `roberta-base` is likely too weak for long narratives.
- Classifying `[anchor; A; B]` jointly is harder than scoring pairs.
- Label smoothing and heavy augmentation may dilute signal.
- Possible truncation of long stories at 512 tokens.

---

## Improvements

### 1. Switch to Pairwise Scoring Architecture
Instead of a three-segment classifier:

1. Score `(anchor, A)` → `s_A`  
2. Score `(anchor, B)` → `s_B`  
3. Predict using:  
   `P(A closer) = softmax([s_A, s_B])`

This is the standard approach in IR reranking and produces more stable optimization.

---

### 2. Use a Stronger Backbone
Recommended:
- **DeBERTa-v3-base or large**
- **RoBERTa-large**
- If text is long: **Longformer-base**, **LED**, or any 1k–2k token model.

This addresses underfitting on long narrative texts.

---

### 3. Reduce Label Smoothing and Re-tune Hyperparameters
- Lower smoothing from 0.1 → 0 or 0.05.
- Test with and without A/B swap augmentation.
- Re-tune LR, warmup, and epochs.

---

### 4. Share Representations with Track B (Optional but Strong)
Use Track B embeddings as auxiliary features or distillation targets:
- Add embedding similarity as an extra input to the classifier.
- Penalize disagreement between CE score difference `(s_A - s_B)` and embedding similarity difference `(sim_A - sim_B)`.

This merges complementary strengths of both tracks.

---

### 5. Optional: Distill LLM Soft Preferences
Ask a strong LLM for:
- Probability distribution over “A closer” vs “B closer”
- Or continuous similarity scores

Train the model using KL-divergence + gold labels (mixed loss).

---

# Final Recommended Roadmap

## Track A (largest performance win)
1. Move to pairwise scoring architecture.  
2. Upgrade backbone to DeBERTa-v3-base or large.  
3. Fix possible truncation issues (increase max length or use long-context models).  
4. Reduce label smoothing and re-tune.  
5. Add optional distillation or Track B auxiliary features.

## Track B
1. Replace MiniLM with BGE/E5/GTE.  
2. Add pairwise triple-based softmax loss.  
3. Use hard negative mining.  
4. Add optional similarity-score distillation.

---

This combined approach is aligned with the best practices in modern retrieval, STS, ranking, and embedding learning, and should provide meaningful gains over the current 0.55 (Track A) and 0.82 (Track B) performance.

