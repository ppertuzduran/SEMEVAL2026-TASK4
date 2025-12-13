# Next-Gen Approach for Narrative Similarity (Tracks A & B)

This document describes a **v2 approach** to improve both Track A and Track B beyond the current
~0.89 (Track A) and ~0.94 (Track B) accuracy levels. The goal is to:

- Push **Track B** closer to the theoretical ceiling by making the bi-encoder the primary model,
  trained with stronger supervision and regularization.
- Make **Track A** a *lightweight interaction head* on top of Track B, instead of a full
  cross-encoder that is currently more prone to overfitting.
- Keep the implementation compatible with the existing `train_track_a.py`, `train_track_b.py`,
  `track_a.py`, and `track_b.py` structure, so changes are incremental rather than a full rewrite.

---

## 1. Current System: Short Diagnostic

### 1.1 What we have now

From the existing implementation and logs:

- **Track B**
  - Backbone: `BAAI/bge-large-en-v1.5` SentenceTransformer.
  - Multi-phase training:
    - Phase 1: MultipleNegativesRankingLoss on (anchor, positive) pairs.
    - Phase 2: custom PairwiseSoftmaxLoss on (anchor, A, B) triples.
    - Phase 3: TripletLoss on (anchor, positive, negative).
  - Optional Phase 4: Distillation from the Track A cross-encoder ensemble.
  - Model selection is driven by accuracy on the Track A dev triples (cosine-based).

- **Track A**
  - Backbone: `BAAI/bge-large-en-v1.5` loaded as `AutoModelForSequenceClassification`.
  - Pairwise scoring cross-encoder:
    - Score (anchor, A) and (anchor, B) separately using the positive class logit.
    - Apply temperature scaling and softmax over `[score_a, score_b]`.
  - Uses 5-fold cross-validation + ensemble (average logits).
  - Regularized with **B→A distillation** from the Track B bi-encoder.
  - Achieves ~0.93 cross-val accuracy but ~0.89 actual inference accuracy on dev.

### 1.2 Observed gap

- **Track B (bi-encoder) already outperforms Track A** on the evaluation metric
  even though Track A is a more expressive model.
- This is a strong signal that:
  - The **bi-encoder is well-aligned** to the triple-wise metric.
  - The **cross-encoder is slightly overfitting**, despite k-fold and distillation,
    and is not delivering extra value over the bi-encoder in terms of generalization.

This motivates a v2 design where **Track B is the primary model**, and **Track A becomes a thin
reranking / interaction head on top of Track B embeddings**, rather than a fully independent model.

---

## 2. High-Level v2 Strategy

1. **Elevate Track B to the main narrative similarity model**:
   - Keep the current 3-phase training, but strengthen metric alignment and regularization.
   - Introduce harder negatives and better supervision on the triple-level decision.

2. **Redesign Track A as a lightweight “interaction head” over Track B embeddings**:
   - Replace the heavy cross-encoder with a small MLP that operates on frozen (or lightly
     fine-tuned) Track B embeddings.
   - Directly optimize a ranking-style loss on (anchor, A, B) using the embedding features.

3. **Late fusion between B and A at inference**:
   - Use Track B cosine similarities *and* the Track A head scores as features in a simple
     calibration layer (e.g., logistic regression) learned on the dev triples.

This setup is more data-efficient (tiny labeled dataset) and enforces **full consistency between
Track A and Track B**, since both are built over the same embedding space.

---

## 3. Improved Track B: Stronger, More Regularized Bi-Encoder

### 3.1 Backbone and embedding dimension

- Keep using `BAAI/bge-large-en-v1.5` as backbone (already strong and compatible).
- Consider **reducing the embedding dimension** via an additional linear projection head
  to **512 dimensions**:
  - Add a linear layer on top of the BGE embedding: `W ∈ R^{512×1024}` + LayerNorm.
  - This reduces capacity and can **improve generalization** on small datasets.
  - Track rules allow any dimension between 10 and 8192, so 512 is legal.

In practice:

- Modify the SentenceTransformer setup to include a final `Dense` layer with output dim 512.
- Keep L2 normalization after the projection.

### 3.2 Loss design: move to a unified, metric-aligned loss

The current curriculum is strong, but we can simplify and better align with the evaluation metric.

**New composite loss for Track B**:

For each triple (anchor, A, B) with label 1 if A is closer, 0 if B is closer:

1. Encode: `e_anchor, e_a, e_b` (normalized).
2. Compute cosine similarities: `s_a = cos(e_anchor, e_a)`, `s_b = cos(e_anchor, e_b)`.
3. Define **margin ranking loss**:
   - Let `y = +1` if A is closer, `y = -1` if B is closer.
   - Use `MarginRankingLoss` on `(s_a, s_b, y)` with margin `m` (e.g., 0.2).
4. Add **pairwise softmax classification loss** on scores `[s_a, s_b]` with temperature τ:
   - This is similar to your current PairwiseSoftmaxLoss, but you combine it with the margin loss.

Overall:

```text
L_total = L_rank_margin + α * L_pairwise_softmax + β * L_MNR + γ * L_simcse
```

Where:

- `L_rank_margin`: `MarginRankingLoss` on (s_a, s_b).
- `L_pairwise_softmax`: cross-entropy over `[s_a, s_b]` with temperature `τ ≈ 0.7`.
- `L_MNR`: MultipleNegativesRankingLoss on (anchor, positive) pairs, as you already use.
- `L_simcse`: optional SimCSE-style **consistency loss** where you apply two different
  dropout/noise views to the same story and maximize agreement between embeddings.

Suggested coefficients (to tune coarsely on dev):

- α = 0.5
- β = 0.5
- γ = 0.1

### 3.3 Hard negative mining

With only 200 triples, a lot of the generalization comes from **how negatives are chosen**.

Add a simple **offline hard-negative mining** step:

1. After an initial training pass (e.g., after the current Phase 2),
   encode all stories with Track B.
2. For each anchor A_i, retrieve its **K nearest neighbors** (by cosine) from the pool
   of *non-positive* candidates.
3. Build new triples where:
   - Positive = known closer story.
   - Negative = one of these **top-K nearest but incorrect** candidates.

Use these mined triples in an extra “Phase HN” with the composite loss above. This focuses
the model on the hardest distinctions and often adds +1–2% accuracy on small ranking datasets.

### 3.4 Regularization and small-data strategies

To avoid overfitting:

- Use **weight decay ~0.05–0.1** for the new projection head, slightly lower (0.01) for the
  backbone.
- Add **dropout 0.1–0.2** in the projection head.
- Keep `max_seq_length` at 384; do not increase beyond that to avoid noise from long tails.
- Enable gradient checkpointing (already done) and mixed precision (already done).

### 3.5 Model selection for Track B

Instead of relying only on intermediate evaluation during `fit`, add a final **hyperparameter
sweep (small grid)** on:

- Temperature τ ∈ {0.5, 0.7, 1.0}
- Margin m ∈ {0.1, 0.2, 0.3}

For each checkpoint, recompute the triple-wise accuracy on dev using cosine similarity, and pick
the best combination. Since dev is tiny, this is very cheap and can yield +0.5–1% accuracy.

---

## 4. New Track A: Lightweight Head over Track B Embeddings

Instead of a full cross-encoder that reprocesses text, we build **Track A directly on top of Track B**.

### 4.1 Architecture

Given Track B embeddings `e_anchor, e_a, e_b` (dim 512 after projection):

1. Construct **pairwise features** for A and B:

```text
φ(a) = [e_anchor, e_a, |e_anchor - e_a|, e_anchor ⊙ e_a]  ∈ R^{4d}
φ(b) = [e_anchor, e_b, |e_anchor - e_b|, e_anchor ⊙ e_b]
```

2. Pass each φ through a **small 2-layer MLP**:

```text
h_a = MLP(φ(a))  → scalar score s_a
h_b = MLP(φ(b))  → scalar score s_b
```

MLP example:

- Layer 1: Linear(4d → 2d) + GELU + Dropout(0.2)
- Layer 2: Linear(2d → 1)

3. Track A prediction uses the same **pairwise softmax** as now:

```text
scores = [s_a, s_b] / τ_a
p = softmax(scores)
```

This is analogous to your current pairwise cross-encoder, but the heavy transformer is replaced
by one forward pass of Track B plus a tiny MLP. The **model capacity is much smaller**, which is
ideal for a 200-example dataset.

### 4.2 Training objective for Track A head

For each triple (anchor, A, B, label):

- Use **cross-entropy loss** on `[s_a, s_b]` (label in {0,1} as “A closer?”).
- Add **distillation from the current Track B** decision, but now in the opposite direction:
  - Teacher scores: `[cos(e_anchor, e_a) / τ_b, cos(e_anchor, e_b) / τ_b]`.
  - Student scores: `[s_a / τ_b, s_b / τ_b]`.
  - KL divergence between teacher softmax and student softmax.

Total loss:

```text
L_A = L_ce + λ_distill * KL(softmax_b || softmax_head)
```

Suggested λ_distill ≈ 0.3.

### 4.3 Fine-tuning strategy

Two modes:

1. **Head-only training (recommended first)**:
   - Freeze all Track B parameters.
   - Train only the MLP head on top of frozen embeddings.
   - This greatly reduces overfitting risk and training instability.

2. **Light joint fine-tuning (optional second stage)**:
   - Unfreeze the last 2–4 transformer layers of Track B + projection head.
   - Apply a **much smaller learning rate** (e.g., 1e-6) to those layers, 1e-4 to the MLP head.
   - Train for 1–2 more epochs with strong weight decay and early stopping on dev.

Because the head is tiny, you can still use **5-fold cross-validation** for robustness and then
ensemble the heads (average scores).

### 4.4 Inference for Track A

Given a triple (anchor, A, B):

1. Encode `anchor`, `A`, `B` once with Track B → get embeddings.
2. Construct φ(a), φ(b), run through the MLP head(s).
3. For an ensemble of heads:
   - Average `[s_a, s_b]` over heads, then apply temperature scaling and compare.

Compute accuracy exactly as the competition expects. This approach is fully compliant with Track A
because the decision is still made directly from the triple; the fact that embeddings come from
Track B is internal to the system.

---

## 5. Joint Calibration & Late Fusion

Once you have a strong Track B and a calibrated Track A head, you can combine them at inference.

For each triple:

- Feature 1: Δ_cos = cos(e_anchor, e_a) − cos(e_anchor, e_b).
- Feature 2: Δ_head = s_a − s_b.

Use the dev triples to fit a **tiny logistic regression or even a 1D threshold** on the pair
(Δ_cos, Δ_head):

```text
P(A closer) = σ(w1 * Δ_cos + w2 * Δ_head + b)
```

Then, at inference, you use this calibrated probability to choose A or B.

Because this “meta-model” has only 2–3 parameters, it is very robust even with 200 samples, and it
lets you **trust the bi-encoder more when it is confident**, and the head more when the cross-
feature agrees. In practice this often gives another +0.5–1.5% absolute accuracy.

This late-fusion layer is purely post-processing on scores and does **not** violate any Track A or
Track B constraints.

---

## 6. Practical Implementation Notes

### 6.1 Where to modify Track B (train_track_b.py)

- **Projection head**:
  - Add a `Dense` / linear module after the encoder to project to 512 dims + LayerNorm.
- **Unified loss**:
  - Replace or augment the current phase structure with a loop that, for each batch of
    triples, computes `L_rank_margin`, `L_pairwise_softmax`, and, optionally, `L_MNR` and `L_simcse`.
  - You can reuse your current PairwiseSoftmaxLoss implementation with minor adjustments.

- **Hard negative mining**:
  - After a first training run, add a small script that:
    - Encodes all stories.
    - Builds mined triples.
    - Saves them as an extra JSONL.
  - In a second run, mix original and mined triples for training.

- **Hyperparameter sweep**:
  - Add a small grid search loop over τ and margin m using the dev triples evaluation already
    implemented at the end of `train_track_b.py`.

### 6.2 Where to modify Track A (train_track_a.py)

- Replace the `AutoModelForSequenceClassification` backbone with a **simple MLP head** that
  consumes frozen Track B embeddings instead of token IDs.
- Reuse the existing dataset and dataloader logic, but change the `__getitem__` to return raw
  texts, then inside the training loop call the Track B model to get embeddings.
- Implement the head as a `torch.nn.Module` with the feature construction described above.
- Keep k-fold logic and early stopping unchanged.

If GPU memory is a concern, embeddings can be **precomputed and cached** for all dev examples, and
only the MLP head is trained over cached vectors.

### 6.3 Inference scripts (track_a.py, track_b.py)

- **Track B (track_b.py)**:
  - No change in external API: still encode each story into a vector and save it.
  - Ensure you use the final 512-dim projected embedding.

- **Track A (track_a.py)**:
  - Replace the cross-encoder predictor with:
    - Load Track B model.
    - Load the trained MLP head (or ensemble of heads).
    - For each triple, compute embeddings once and pass through the head(s).
  - Optionally include the late-fusion logistic regression layer:

    ```python
    delta_cos = sim_a - sim_b
    delta_head = score_a - score_b
    logit = w1 * delta_cos + w2 * delta_head + b
    pred = logit > 0
    ```

---

## 7. Expected Impact

While actual gains depend on hidden test data, this v2 approach should:

- **Track B**:
  - Improve robustness via:
    - Better metric-aligned loss (margin + softmax).
    - Hard negative mining.
    - Projection + stronger regularization.
  - Expected gain: **+1–2% absolute** over the current ~0.94, if there is remaining headroom.

- **Track A**:
  - Eliminate heavy cross-encoder overfitting by using a small head on a strong embedding space.
  - Enforce full consistency with Track B because both operate on the same embeddings.
  - Late fusion can leverage differences between cosine and head scores.
  - Expected gain: **+2–3% absolute** over the current ~0.89, making Track A competitive with or
    slightly better than Track B.

The key philosophy is: **One powerful, metric-aligned bi-encoder (Track B) + a tiny interaction
layer (Track A) + simple calibration** is a better match to the very low-data regime of this task
than two large, independently trained transformers.
