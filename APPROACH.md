# Distillation-Centric Approach for Narrative Similarity (Tracks A & B)

This document describes a unified, **distillation-centric** strategy for the competition, where:
- **Track B** trains a strong bi-encoder first.
- **Track A** then builds on the same encoder and adds interaction layers.
- Track A and Track B are coupled via **knowledge distillation**, improving both accuracy and consistency between tasks.

The focus is on **maximum performance**, choosing strong models directly (without inline “advice comments” in config-like snippets).

---

## 1. Task Recap and Constraints

- **Track A (Classification / Reranking)**  
  Input: `(anchor_story, choice_A, choice_B)`  
  Output: which of A or B is more narratively similar to the anchor.  
  Metric: accuracy on triple-wise comparisons.

- **Track B (Embeddings)**  
  Input: single stories.  
  Output: embeddings such that **cosine similarity** between embeddings aligns with narrative similarity.  
  Evaluation: triple-wise comparison derived from cosine similarities.  
  Constraint: at **inference time**, embeddings must be computed **independently per story** (no triple-aware encoding).  
  Embedding dimension must be between **10 and 8192**.

---

## 2. High-Level Strategy

1. **Train a powerful bi-encoder for Track B** as the core narrative representation model.
2. **Reuse the same encoder for Track A**, adding cross-encoder interaction layers on top.
3. Use **distillation in both directions**:
   - Track A cross-encoder → Track B bi-encoder (classic teacher → student for retrieval).
   - Track B bi-encoder → Track A cross-encoder (regularization and warm start).
4. Use **ensembling**, **hard-negative mining**, and carefully designed **curricula** to align both tracks with the evaluation metric.

This creates a single strong narrative representation backbone serving both tracks, and reduces fragmentation between models.

---

## 3. Base Models and Representations

### 3.1. Track B Base Encoder

- **Model:** `BAAI/bge-large-en-v1.5`  
- **Embedding dimension:** 1024 (well within the 8192 limit).  
- **Tokenization and pooling:** use the official BGE settings (CLS pooling with L2-normalization of final embeddings).

### 3.2. Track A Cross-Encoder

- **Encoder backbone:** initialized from the **fine-tuned Track B bi-encoder weights** (same transformer encoder).  
- **Architecture:**
  - Input: concatenation of texts with explicit markers, e.g.  
    `[ANCHOR] anchor_text [/ANCHOR] [CAND] candidate_text [/CAND]`
  - Encode with the shared transformer.
  - Take a pooled representation (CLS or mean-pool over all tokens).
  - Pass through a small feed-forward head to produce a scalar **score(anchor, candidate)**.

The same encoder backbone parameters are shared conceptually between Track A and Track B (with different heads), encouraging the system to learn a consistent narrative space.

---

## 4. Training Pipeline Overview

The training is organized into several stages:

1. **Stage 0 – Optional external warmup (if allowed).**
2. **Stage 1 – Track B core training (bi-encoder).**
3. **Stage 2 – Initialize Track A cross-encoder from Track B and train on triples.**
4. **Stage 3 – Distillation from Track A → Track B.**
5. **Stage 4 – Mutual refinement and ensembling.**

Each stage is described below.

---

## 5. Stage 1: Track B – Core Bi-Encoder Training

### 5.1. Data Construction

Use the competition triples `(anchor, A, B, label)` to create bi-encoder training samples:

- **Positives:** `(anchor, positive)` and `(positive, anchor)` where `positive` is the story labeled as more similar to `anchor`.
- **Negatives:** `(anchor, negative)` and `(negative, anchor)` using the less similar story.
- Optionally, include **symmetric variants** and extra pairs `(A, B)` if there is any derived signal between choices.

### 5.2. Loss Curriculum

Use a **multi-phase curriculum** to shape the embedding space:

1. **Phase 1 – MultipleNegativesRankingLoss (global structure).**
   - Build batches of anchor–positive pairs.
   - Use MultipleNegativesRankingLoss so that each positive is contrasted with all other positives in the batch as negatives.
   - This gives a good global semantic structure for narrative similarity.

2. **Phase 2 – PairwiseSoftmaxLoss (competition-aligned).**
   - For each triple `(anchor, A, B)`:
     - Compute `s_A = cosine(anchor, A)`, `s_B = cosine(anchor, B)`.
     - Stack `[s_A, s_B]` and apply a softmax + CrossEntropy w.r.t the label (A-more-similar vs B-more-similar).
   - Optionally introduce a temperature τ on the scores before the softmax.
   - This phase aligns the embedding geometry directly with the evaluation metric.

3. **Phase 3 – TripletLoss with mined negatives (fine-grained discrimination).**
   - After Phase 2, encode all stories and construct **hard negatives**:
     - For each anchor, find nearest neighbors which are not labeled as positives.
   - Build triplets `(anchor, positive, hard_negative)` and train with TripletLoss, margin tuned on validation.

Phase durations can be tuned empirically, but a practical rule is:
- 30–40% epochs for Phase 1,
- 30–40% for Phase 2,
- 20–30% for Phase 3.

### 5.3. Regularization and Optimization

- Optimizer: **AdamW** with linear warmup and cosine decay.
- Regularization: weight decay and dropout as in the base BGE config.
- Early stopping based on a Track B dev metric (triple-wise accuracy reconstructed from cosine scores).

At the end of Stage 1, we have a strong Track B bi-encoder, used as the **initial backbone** for Track A.

---

## 6. Stage 2: Track A – Cross-Encoder Training from Track B Backbone

### 6.1. Initialization

1. Clone the trained Track B encoder weights into a new model instance.
2. Attach a small **cross-encoder head**:
   - Input: sequence of `[ANCHOR] anchor [/ANCHOR] [CAND] candidate [/CAND]` tokens.
   - Encoder: same transformer as Track B, now used jointly on concatenated texts.
   - Head: MLP mapping the pooled representation to a scalar similarity score.

### 6.2. Pairwise Classification Objective

For each triple `(anchor, A, B, label)`:

1. Compute:
   - `score_A = model(anchor, A)`
   - `score_B = model(anchor, B)`
2. Form logits `logits = [score_A, score_B]`.
3. Apply a **temperature-scaled** softmax and CrossEntropyLoss:
   - `logits' = logits / τ`
   - `loss_cls = CrossEntropyLoss(logits', label)`

Use symmetric augmentation:
- Include both `(anchor, A, B, label)` and `(anchor, B, A, 1-label)` in training.

### 6.3. Distillation from Track B → Track A (Warm Regularization)

To ensure Track A does not drift arbitrarily far from the embedding-based behavior learned in Stage 1, add a **regularization term based on Track B**:

1. For each `(anchor, A, B)`:
   - Bi-encoder (Track B) gives cosine similarities:
     - `sim_A = cosine_B(anchor, A)`
     - `sim_B = cosine_B(anchor, B)`
   - Normalize them into a soft distribution with temperature `τ_B`:
     - `p_B = softmax([sim_A, sim_B] / τ_B)`
2. Cross-encoder produces logits `[score_A, score_B]` and a soft distribution `p_A = softmax([score_A, score_B] / τ_B)`.
3. Add a **KL-divergence distillation loss**:
   - `loss_distill_B_to_A = KL(p_B || p_A)`

Total loss for Track A:

`loss_A = loss_cls + λ_BA * loss_distill_B_to_A`

where `λ_BA` controls the strength of the distillation from Track B to Track A.

This uses the bi-encoder as a **prior** that stabilizes the cross-encoder, especially at the beginning of training.

---

## 7. Stage 3: Distillation from Track A → Track B

Once the Track A cross-encoder is strong (it typically becomes the more accurate scorer because it attends jointly to anchor and candidate), we use it as a **teacher** to refine Track B.

### 7.1. Teacher–Student Setup

- **Teacher:** Track A cross-encoder (frozen during this phase, or updated slowly).
- **Student:** Track B bi-encoder.

For each triple `(anchor, A, B)`:

1. Teacher computes:
   - `t_A = score_A_teacher(anchor, A)`
   - `t_B = score_B_teacher(anchor, B)`
   - `p_T = softmax([t_A, t_B] / τ_T)`

2. Student bi-encoder computes:
   - `sim_A = cosine_B(anchor, A)`
   - `sim_B = cosine_B(anchor, B)`
   - `p_S = softmax([sim_A, sim_B] / τ_T)`

3. Distillation loss:
   - `loss_T_to_B = KL(p_T || p_S)`

Optionally combine with the original supervised PairwiseSoftmaxLoss:

`loss_B_total = loss_pairwise + λ_TB * loss_T_to_B`

This forces the embedding-based model to mimic the finer-grained judgments of the cross-encoder while preserving its ability to embed stories independently.

### 7.2. Hard-Example Focus

During this phase, prioritize triples where the student is currently wrong or has low margin between `sim_A` and `sim_B`. This accelerates learning and focuses distillation on the challenging regions of the space.

---

## 8. Stage 4: Mutual Refinement and Ensembling

### 8.1. Iterative Refinement

If time allows, we can alternate:

1. Update Track B with distillation from the current Track A.
2. Re-train or fine-tune Track A with updated regularization from the new Track B.

This “mutual teaching” can be done for a small number of cycles, each time re-evaluating on dev sets.

### 8.2. Ensembling Strategies

- **Track A (Cross-Encoder):**
  - Train **K folds** with different seeds and data splits.
  - At inference, compute logits for each fold and **average** them before decision.
- **Track B (Bi-Encoder):**
  - Train multiple variants of the same `bge-large-en-v1.5` model (different seeds, small config changes).
  - At inference, either:
    - **Average** embeddings from all models, or
    - **Concatenate** embeddings (still within the 8192 dimension limit), then L2-normalize the result.

Ensembling is especially powerful in small/medium datasets and typically yields a noticeable boost in accuracy.

---

## 9. Evaluation and Metrics

### 9.1. Track A

- Metric: accuracy on triples `(anchor, A, B)`.
- Always evaluate:
  - Single-cross-encoder models.
  - Fold ensembles.
- Log misclassified triples to guide error analysis and potential synthetic data generation.

### 9.2. Track B

- Metric: triple-wise accuracy reconstructed from cosine comparisons:
  - Predict A if `cosine(anchor, A) > cosine(anchor, B)`.
- Evaluate:
  - Single bi-encoder model from Stage 1.
  - After distillation from Track A (Stage 3).
  - After ensembling across multiple models.

---

## 10. Summary of the Distillation-Based Approach

1. **Train a strong Track B bi-encoder** using `BAAI/bge-large-en-v1.5` with a curriculum of MultipleNegativesRankingLoss → PairwiseSoftmaxLoss → TripletLoss and hard-negative mining.
2. **Initialize Track A cross-encoder** from the Track B backbone and train it on triple classification with symmetric augmentation and temperature-scaled CrossEntropy.
3. **Regularize Track A** with a distillation signal from the bi-encoder (Track B → Track A) to stabilize training.
4. **Refine Track B** by distilling from the stronger Track A cross-encoder (Track A → Track B), aligning embeddings with the cross-encoder’s decisions.
5. **Use ensembling** for both tracks and, optionally, iterative mutual refinement to squeeze out additional performance.

This setup leverages a **shared encoder** and **two-way distillation** to make Tracks A and B reinforce each other, while strictly respecting the constraint that Track B embeddings at inference time are computed independently per story.
