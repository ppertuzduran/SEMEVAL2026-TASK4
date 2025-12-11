# Improved Approach for Narrative Similarity Competition (Tracks A & B)

This document refines the existing approach and outlines **next-step, higher-impact improvements** to push accuracy beyond the current implementation. It assumes the current codebase described in the previous `APPROACH.md` and training scripts for Track A and Track B, and focuses on **modeling and training strategy changes**, not code-level details.

---

## 1. Task Recap and Constraints

- **Track A (Classification)**  
  Input: `(anchor_story, choice_A, choice_B)`  
  Output: which of A or B is more narratively similar to the anchor.  
  Metric: accuracy on triple-wise comparisons.

- **Track B (Embeddings)**  
  Input: individual stories.  
  Output: vector embeddings (dimension 10–8192) whose **cosine similarity** reflects narrative similarity.  
  Evaluation: triple-wise accuracy derived from cosine comparison (similar to Track A).  
  Constraint: At **inference**, embeddings must be computed per story independently (no triple-aware embeddings at inference time).

The current system already applies strong ideas:
- Track A: **RoBERTa-large cross-encoder** with pairwise scoring and k-fold training. fileciteturn0file0  
- Track B: **BGE-base bi-encoder** with a curriculum of Pairwise Softmax → TripletLoss → MultipleNegativesRankingLoss. fileciteturn0file0  

The goal here is to move closer to **state-of-the-art (SOTA)** for semantic similarity / retrieval with **minimal code churn** but **maximum gain per change**.

---

## 2. High-Level Strategy

1. **Exploit stronger base models** where allowed by compute:
   - Upgrading from base → large encoders when feasible.
   - Consider more recent architectures (DeBERTa-v3, ModernBERT, newer BGE/GTE variants).

2. **Tighten alignment between training objectives and evaluation:**
   - Track A: keep pairwise scoring, but improve regularization, sampling, and ensembling.
   - Track B: keep triple-wise optimization, but improve hard-negative mining and curriculum.

3. **Use ensembling and cross-validation more aggressively:**
   - Exploit the existing k-fold infrastructure in Track A for **model averaging at inference**.
   - Introduce analogous data-splitting and ensembling logic for Track B embeddings.

4. **Better negatives and augmentations:**
   - Leverage the track data itself to build **hard negatives** for both tracks.
   - Use symmetric transformations and light textual augmentations for robustness.

5. **Guard against overfitting on small data:**
   - Calibrated regularization, early stopping, and cross-validation.
   - Avoid leaking dev set signal in Track B when using Track A triples to evaluate.

---

## 3. Track A – Cross-Encoder Improvements

### 3.1. Model Architecture and Base Encoder

**Current:** RoBERTa-large cross-encoder with pairwise scoring and softmax over `[score_A, score_B]`. fileciteturn0file0turn0file1  

**Recommended upgrades (prioritized):**

1. **Try a stronger encoder**:
   - **DeBERTa-v3-large** or **modern long-context LMs** (e.g., Longformer/ModernBERT) if story lengths are close to or exceed 512 tokens.
   - Keep the same pairwise architecture: `encode(anchor ⊕ SEP ⊕ candidate)`.

2. **Explicit segment marking** (if not already supported):
   - Surround anchor and candidate with special tokens, e.g. `[ANCHOR] ... [/ANCHOR] [CAND] ... [/CAND]` fed into the encoder.
   - Helps the model disambiguate roles of anchor vs. candidate.

3. **Longer context when needed:**
   - If narratives often get truncated, consider increasing `max_length` (e.g. 768–1024) with a more memory-efficient model, or using a long-context encoder.

### 3.2. Loss, Calibration, and Regularization

1. **Temperature-scaled pairwise loss:**
   - Keep the current CrossEntropy over `[score_A, score_B]`, but introduce a learnable or fixed temperature `τ`:
     

     `scores' = scores / τ`, then apply `CrossEntropyLoss`.

   - A smaller `τ` (e.g. 0.5) sharpens the distribution; tune on validation or via grid search.

2. **Symmetric training with A/B swaps (systematic):**
   - You already have position-swap augmentation in the conceptual approach; enforce it strictly in data prep: for each triple, include both `(anchor, A, B, label)` and `(anchor, B, A, 1-label)`.
   - This removes position bias and doubles effective data.

3. **R-Drop style consistency regularization (if compute allows):**
   - For each input pair, do two forward passes with dropout on, and add a KL-divergence loss between their logits.
   - Encourages more stable predictions and improves generalization on small datasets.

4. **Careful label smoothing:**
   - If you reintroduce label smoothing, keep it **small** (e.g. 0.02–0.05). Too much smoothing harms a binary pairwise task.

### 3.3. Data and Sampling Strategy

1. **Curriculum over difficulty:**
   - Initially sample triples uniformly.
   - After a warmup stage, compute model confidence; prioritize **borderline cases** where |score_A − score_B| is small for continued training (hard-example mining at Triple-level).

2. **Text augmentations (light only):**
   - Mild word-level synonym replacements or paraphrases for anchors and candidates, preserving narrative meaning.
   - Small random deletion of unimportant adjectives/adverbs to encourage robustness.

3. **Leverage Track B embeddings for hard negatives:**
   - Use a **frozen Track B encoder** to find stories that are embedding-close but not the gold answer.
   - Build additional triples where the negative is “semantically close but wrong”, improving discriminative power.

### 3.4. K-Fold Training and Ensembling

**Current:** K-fold training implemented but likely used mainly for evaluation. fileciteturn0file1  

**Improvement:** Use **fold models as an ensemble at inference**:

- Train `K` models (one per fold).
- At inference, for a given triple:
  - Compute `(score_A_k, score_B_k)` for each model `k`.
  - Aggregate (e.g., average logits or probabilities) across folds.
  - Decide via aggregated scores.
- Ensembling of 3–5 strong models often yields **1–3 points** of accuracy improvement with no change to the training script’s core logic.

### 3.5. Evaluation & Error Analysis Loop

- Maintain an **error log of dev triples** where the model consistently fails (across folds).
- Analyze patterns:
  - Temporal reasoning? Character identity confusion? Plot twist handling?
- Use this to design **targeted synthetic data** (via LLMs) that mirror those failure modes and add them as additional training triples.

---

## 4. Track B – Embedding Model Improvements

### 4.1. Base Encoder and Dimensionality

**Current:** `BAAI/bge-base-en-v1.5` (109M parameters, 768-dim embeddings). fileciteturn0file0turn0file2  

**Improvements:**

1. **Try a stronger encoder within the 8192-dim limit:**
   - **BGE-large-en**, **GTE-large**, or a similar high-quality English embedding model (typically 1024-dim).
   - The dimensionality constraint (10–8192) is generous, allowing more expressive representations.
   - Verify that hardware constraints (VRAM) remain acceptable.

2. **Explicit normalization and scaling:**
   - Ensure embeddings are **L2-normalized** before cosine similarity (either via configuration or explicit normalization layer).
   - Experiment with a **scaling factor / temperature** on cosine scores during training to sharpen gradients.

### 4.2. Training Curriculum and Losses

**Current:** Multi-phase training with:
1. PairwiseSoftmaxLoss on triples.
2. TripletLoss on triplets.
3. Optional MultipleNegativesRankingLoss on positive pairs. fileciteturn0file0turn0file2  

**Refinements:**

1. **Re-balance phase lengths and monitor Track B-specific metrics:**
   - Instead of fixed `epochs // 2` per phase, adapt based on validation performance.
   - Introduce a small, **true Track B-dev set** (no overlap with Track A dev) to monitor embedding quality directly (via triple reconstruction or STS-style scoring).

2. **Hard-negative mining loop:**
   - After initial training, encode all stories.
   - For each anchor, retrieve top-K nearest neighbors (excluding the gold positives).
   - Build new triplets `(anchor, gold_positive, hard_negative)` where negatives are **embedding-near but label-far**.
   - Run an additional fine-tuning stage **only** on these hard triplets with TripletLoss and/or PairwiseSoftmaxLoss.

3. **Order of losses:**
   - Consider starting with **MultipleNegativesRankingLoss** on large positive-pair data to shape a global semantic space.
   - Then apply **PairwiseSoftmaxLoss** focused specifically on competition triples, aligning final geometry to the evaluation metric.

4. **Mix-up between pairwise and triplet objectives:**
   - In a final fine-tuning stage, alternate batches from PairwiseSoftmax and TripletLoss instead of full-phase separation, to avoid the later loss overwriting previous structure.

### 4.3. External Data and Multi-Task Learning (if allowed)

If the competition rules permit additional data:

1. **Pre-fine-tune on general STS / story similarity datasets:**
   - Use generic sentence-embedding datasets (STS, NLI-based contrastive sets, etc.) to warm up the embedding model.

2. **Narrative-specific auxiliary tasks:**
   - Next-sentence prediction or ordering tasks on narrative corpora (story shuffling, story cloze-style datasets) to help the encoder internalize story flow and coherence.

3. **Multi-task regime:**
   - Alternate between generic STS batches and competition-specific triple/pair batches, so the model does not overfit the small competition dataset.

### 4.4. Using Track A as a Teacher (Knowledge Distillation)

Without violating the Track B inference rule (embeddings must be story-wise):

1. **Teacher = Track A cross-encoder**, Student = Track B bi-encoder.
2. For each triple `(anchor, A, B)`:
   - Teacher produces logits `(s_A, s_B)`.
   - Student embeddings produce similarities `(sim_A, sim_B)`.
   - Minimize a **distillation loss**:
     - KL-divergence between softmax(teacher_scores / τ) and softmax(student_scores / τ),
       where `student_scores = [sim_A, sim_B]`.
3. This aligns the embedding geometry with the **fine-grained judgments** of the cross-encoder while maintaining independent per-story embeddings at inference.

This is especially powerful when the cross-encoder is stronger (e.g. DeBERTa-v3-large) than the bi-encoder.

### 4.5. Ensembling Embeddings

To ensemble for Track B:

- Train **multiple bi-encoders** with different seeds / architectures (e.g., BGE-base, BGE-large, GTE-large).
- At inference: compute embeddings from each model and **concatenate or average** them:
  - Concatenation: increases dimensionality but stays under 8192 easily.
  - Averaging: keeps dimension fixed but aggregates knowledge.
- Normalize the final embeddings and use cosine similarity as usual.

This can be especially robust when each model is trained with slightly different loss schedules or augmentations.

---

## 5. Shared Improvements Across Tracks

### 5.1. Reproducibility and Robust Validation

- Keep strict seed control (already present) and consider enabling deterministic flags if training variance is too high.
- Use **stratified splits** across narrative types (if metadata available) to avoid domain shift between train and dev folds.

### 5.2. Self-Training / Pseudo-Labeling

1. Use the best current models to annotate additional unlabeled story pairs/triples (if you have access to more raw stories).
2. Filter pseudo-labels by high-confidence predictions.
3. Re-train or fine-tune both Track A and Track B models on the union of gold + high-confidence pseudo-labeled examples.

### 5.3. Model Selection and Checkpoint Averaging

- For each model, instead of picking a single best epoch checkpoint, consider **weight averaging** across the top N checkpoints (e.g. last 3 epochs before overfitting) to reduce variance.
- This is often a cheap gain in stability and generalization.

---

## 6. Concrete “Next Experiments” Roadmap

Here is an ordered list of experiments that are likely to give the **best return on time**:

1. **Track A – DeBERTa-v3-large + fold ensemble:**
   - Swap base model to DeBERTa-v3-large.
   - Train with existing k-fold setup.
   - At inference, ensemble all folds by averaging logits.

2. **Track B – Upgrade to BGE-large (or similar) + same curriculum:**
   - Keep the same PairwiseSoftmax → Triplet → MNR pipeline.
   - Carefully watch VRAM and slightly lower batch size if needed.

3. **Track A & B – Systematic A/B swap augmentation and light text augmentation.**

4. **Track B – Hard-negative mining and second TripletLoss stage:**
   - Mine hard negatives using the current encoder.
   - Fine-tune for a few epochs only on these hard triplets.

5. **Cross-task Distillation (Track A → Track B):**
   - Train Track B with an additional distillation loss that matches its pairwise cosine scores to Track A’s logits.

6. **Error-driven synthetic data generation:**
   - For recurrent failure patterns, generate synthetic triples via a large LLM (if allowed) and incorporate them into training.

Executing these steps iteratively, monitoring dev accuracy after each change instead of stacking all changes at once, should give a clear view of which ideas are truly moving the needle.

---

## 7. Summary

- The **current approach is already solid and aligned with modern IR and metric-learning best practices**. fileciteturn0file0turn0file1turn0file2  
- Major remaining gains are likely to come from:
  - Stronger backbones (DeBERTa-v3 / larger embedding models),
  - Better ensembling and cross-validation usage,
  - Hard-negative mining and distillation between Track A and Track B,
  - Carefully designed curricula and augmentations rather than radically new architectures.

This plan keeps your existing codebase and training scripts as the backbone, while layering on **SOTA-inspired enhancements** in a controlled, incremental way.
