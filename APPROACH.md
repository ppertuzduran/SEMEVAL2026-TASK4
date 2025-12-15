# APPROACH v11 – Qwen3-Embedding Backbone Upgrade

## 1. Overview

This document describes **APPROACH v11**, an evolution of the previous methodology used in this competition.

The key change in this version is the **replacement of the backbone embedding model**
`BAAI/bge-large-en-v1.5` with **Qwen3-Embedding**, starting with:

1. **Qwen/Qwen3-Embedding-4B** (initial validation and tuning), then  
2. **Qwen/Qwen3-Embedding-8B** (final high-accuracy runs, subject to GPU resources).

All other components (training scripts, loss design, evaluation logic, and inference format) are kept as
consistent as possible to **isolate the impact of the backbone change**.

The rest of this document is organized as:

- Repository & script mapping
- Motivation for the backbone change
- Detailed Track B approach (embeddings)
- Detailed Track A approach (pairwise decision)
- Inference and submission format
- **Migration checklist** (what to change in code/configs)
- **Risks, constraints & mitigations**

---

## 2. Repository & Script Mapping

The approach assumes the following core files and roles in the repository:

- **High-level documentation**
  - `APPROACH.md` (this document)

- **Training – Track B (embeddings)**
  - `train_track_b.py`  
    Main training entrypoint for Track B. Loads a base encoder, wraps it in a SentenceTransformer,
    applies a projection head, and trains using triple-wise/narrative similarity supervision.
  - `run_track_b_experiments.py`  
    Orchestrates multiple training runs with different hyperparameters and writes logs.

- **Training – Track A (triple decision)**
  - `train_track_a.py`  
    Loads a trained Track B model (frozen or lightly tuned), attaches an MLP classifier on top of
    pairwise features, and trains on triples.

- **Inference**
  - `track_b.py`  
    Generates final Track B embeddings for all stories to be submitted to Codabench.
  - `track_a.py`  
    Uses Track B embeddings and the Track A classifier to generate A/B decisions for all triples.

- **Evaluation & utilities**
  - `eval_local.py`  
    Local evaluation script to compute accuracy and/or the Codabench metric based on generated
    embeddings or predictions.
  - `augment_training_data.py`  
    Script used to build an augmented training set for Track B (lexical perturbations, filters, etc.).

- **Configuration**
  - `config.yaml`, `best_config.yaml`, `configv8.yaml`  
    YAML files describing model, training, and data parameters for different approach versions.
    In v11, the model `name` or `base_model` field is updated to point to Qwen3-Embedding.

This document focuses on how these components change conceptually, and the **Migration checklist**
(Section 8) provides concrete steps for modifying them.

---

## 3. Motivation for the Backbone Change

After extensive experimentation with the BGE-large backbone, the following techniques were tried:

- Hard negative mining
- SimCSE-style objectives
- Curriculum learning and loss re-weighting
- Data augmentation with lexical perturbations
- Cross-encoder teacher and distillation (APPROACH v8)
- Multiple projection head and regularization settings

Despite the sophistication of the pipeline, the system **plateaued** at:

- **Track A:** ~0.95 accuracy  
- **Track B:** ~0.94 accuracy  

Many additional tweaks either left performance unchanged or **slightly degraded** it, which strongly
suggests that the **representation quality of the backbone** (and/or dataset noise) has become the main
bottleneck, rather than head architecture or loss design.

### Why Qwen3-Embedding?

Qwen3-Embedding models achieve top performance on the **MTEB English** family of benchmarks and are
explicitly designed as **embedding models** (not generic instruction-following LLMs). Key properties:

- Very strong semantic representations for English sentence and document similarity
- Long-context support (token windows up to 8192), which is valuable for narrative stories
- Architecturally optimized for embedding extraction (e.g., last-token pooling)
- Compatible with cosine similarity and SentenceTransformers-style usage
- Open weights, suitable for research competitions

Given these properties, upgrading to Qwen3-Embedding is a **high-leverage change** that can improve both
Track A and Track B without re-architecting the entire pipeline.

---

## 4. Selected Models

### Phase 1: Initial Validation – `Qwen/Qwen3-Embedding-4B`

- Lower memory footprint and faster iterations
- Used to:
  - Verify training scripts and configs
  - Tune basic hyperparameters (batch size, LR, projection size)
  - Validate that the new embeddings perform at least as well as BGE-large

### Phase 2: High-Accuracy Runs – `Qwen/Qwen3-Embedding-8B`

- Higher capacity and stronger performance on public benchmarks
- Used for:
  - Final Track B training runs
  - Embedding generation for final submissions
- Only adopted once:
  - The 4B version is stable
  - Hardware constraints (VRAM) are confirmed to be sufficient

---

## 5. Track B: Embedding Model Approach

### 5.1 Objective

The goal of Track B is to produce a **single embedding per story** such that cosine similarity between
embeddings reflects narrative similarity. Importantly, **each story must be embedded independently** at
inference time (no triple-level or cross-story conditioning is allowed).

### 5.2 Architecture

- **Base encoder:** Qwen3-Embedding (4B or 8B)
- **Tokenizer:** The official tokenizer shipped with Qwen3-Embedding
- **Pooling strategy:** **Last-token pooling** (recommended by Qwen for embeddings)
- **Projection head:**
  - Optional linear or shallow MLP projection from encoder dimension to a fixed embedding size
  - Final L2 normalization of the projection output

High-level flow:

```text
Text → Qwen3 Encoder → Last Token Representation → Projection Head → Normalize → Embedding
```

The projection head allows us to:
- Control final embedding dimensionality (if needed for Codabench limits)
- Adapt the representation slightly to the task without overfitting the full transformer

### 5.3 Training Strategy

The training pipeline is intentionally **simplified** compared to earlier versions, because experience
showed that excessive loss complexity (e.g. multi-stage curricula, aggressive hard negatives) often
degraded generalization on this relatively small, noisy dataset.

Core elements:

- Supervision based on triples (anchor, positive, negative)
- Embedding similarity measured via cosine distance
- Early stopping on local triple-wise accuracy (using `eval_local.py`)

Recommended loss combination:

1. **Pairwise softmax loss** on (anchor, positive, negative) similarity scores
2. **Light margin ranking loss** on the same triples

Hard negatives, SimCSE, and complex multi-stage curricula from previous approaches are **disabled**
or kept extremely conservative in v11.

### 5.4 Relationship with Previous Versions

- The input data preparation, batching, and evaluation logic remain the same.
- The **only major change** is the base encoder (`Qwen/Qwen3-Embedding-*` instead of BGE-large).
- This design allows a clean attribution of any performance gains to the backbone change.

---

## 6. Track A: Pairwise Decision Model

### 6.1 Objective

Given a triple (anchor, A, B), pick which candidate (A or B) is **more narratively similar** to the anchor.

### 6.2 Architecture

Track A continues to operate as a **lightweight classifier on top of Track B embeddings**:

1. Use the trained Track B model (Qwen3-based) to embed:
   - Anchor story → `e_anchor`
   - Candidate A  → `e_A`
   - Candidate B  → `e_B`

2. For each candidate pair (anchor, candidate) build feature vectors such as:
   - Concatenation: `[e_anchor, e_candidate]`
   - Absolute difference: `|e_anchor − e_candidate|`
   - Elementwise product: `e_anchor ⊙ e_candidate`

   Typically combined as:
   ```text
   features = [e_anchor, e_candidate, |e_anchor − e_candidate|, e_anchor ⊙ e_candidate]
   ```

3. Feed features into a small MLP classifier:
   - 1–2 hidden layers
   - GELU/ReLU activations
   - Dropout for regularization
   - Output: logits `[logit_A, logit_B]` for the two candidates

4. Apply cross-entropy loss over the two logits using the ground truth label (which candidate is closer).

### 6.3 Improvements in v11

- The **only major change** on Track A is the **stronger embedding backbone** (Qwen3 instead of BGE-large).
- Optional robustness enhancements:
  - Train multiple MLP heads with different random seeds; average logits at inference
  - Use very light fine-tuning of the last few encoder layers with a small learning rate, if useful

The decision rule and output format remain unchanged and continue to satisfy Codabench requirements.

---

## 7. Inference & Submission

### 7.1 Track B Inference

- For each story in the dataset:
  1. Tokenize the text with the Qwen3-Embedding tokenizer
  2. Run the Qwen3 encoder to obtain hidden states
  3. Apply last-token pooling (or the official pooling layer)
  4. Pass through the projection head (if used)
  5. Normalize the resulting vector

- Store the resulting embeddings as:
  - Arrays of floats (one vector per story), or
  - A serialization format expected by Codabench

Each story is processed **independently**, which fully respects Track B rules.

### 7.2 Track A Inference

- For each triple (anchor, A, B):
  1. Load or compute `e_anchor`, `e_A`, `e_B` using the Track B model
  2. Build features for (anchor, A) and (anchor, B)
  3. Pass features through the trained MLP head (or an ensemble of heads)
  4. Select the candidate with higher logit / probability

The prediction for each triple is written in the same structure as previous approaches so that
`track_a.py` and the Codabench submission format do not need to change.

---

## 8. Migration Checklist (from BGE-large to Qwen3-Embedding)

This section enumerates the **practical steps** required to migrate the repository from
`BAAI/bge-large-en-v1.5` to Qwen3-Embedding.

### 8.1 Configuration Files

1. Open `config.yaml` (and `best_config.yaml` if used as a template).
2. Locate the field defining the base model, e.g.:
   - `model_name: "BAAI/bge-large-en-v1.5"`  
   or  
   - `base_model: "BAAI/bge-large-en-v1.5"`
3. Replace with:
   - Phase 1:
     ```yaml
     model_name: "Qwen/Qwen3-Embedding-4B"
     ```
   - Phase 2 (after validation):
     ```yaml
     model_name: "Qwen/Qwen3-Embedding-8B"
     ```
4. If the config specifies pooling:
   - Set to `"last_token"` or the corresponding Qwen3-Embedding pooling option.
5. Review embedding dimension settings:
   - Ensure any `embedding_dim` parameters match the Qwen3 encoder or the projection-head output size.

### 8.2 `train_track_b.py`

1. Update the model loading to use Qwen3-Embedding:
   - Replace references to BGE-large with Qwen model names.
2. Ensure the tokenizer and config are taken from the Qwen3-Embedding checkpoint.
3. Confirm pooling logic:
   - Use the official pooling for Qwen3 or implement last-token pooling explicitly.
4. Verify the projection head input dimension:
   - It must match the encoder hidden size of Qwen3-Embedding.
5. Keep existing training loops, loss functions, and evaluation hooks unchanged for the first runs
   (to isolate the backbone effect).

### 8.3 `run_track_b_experiments.py`

1. Keep the experiment structure and hyperparameter ranges as-is initially.
2. If GPU memory is tight:
   - Adjust batch sizes and gradient accumulation steps for Qwen3-Embedding-4B / 8B.
3. Clearly tag v11 runs in logs (e.g. experiment name or output folder) to distinguish from BGE runs.

### 8.4 `train_track_a.py`

1. Update the code that loads the Track B model:
   - Point it to the **Qwen3-based** Track B checkpoint directories.
2. Keep the MLP head architecture unchanged initially.
3. Optional: add support for training multiple heads with different seeds for ensembling.

### 8.5 `track_b.py` and `track_a.py` (Inference)

1. Update any hardcoded base model names to Qwen3-Embedding.
2. Ensure the embedding dimension used when serializing to Codabench matches the new model:
   - If a projection head is used, the dimension is the projection output size.
3. No changes are required in the **output format**; only the underlying model is swapped.

### 8.6 Local Evaluation

1. Use `eval_local.py` to:
   - Compare the local triple-wise metrics of Qwen3-based embeddings against BGE-based ones.
2. Only after the Qwen3 version is clearly non-regressing locally, proceed to Codabench submissions.

---

## 9. Risks, Constraints & Mitigations

### 9.1 Increased Computational Cost

- **Risk:** Qwen3-Embedding-4B/8B requires more GPU memory and compute than BGE-large.
- **Mitigation:**
  - Start with **4B** for development and hyperparameter tuning.
  - Use smaller batch sizes and gradient accumulation steps.
  - Use mixed precision (fp16/bf16) if supported.

### 9.2 Overfitting to Small / Noisy Data

- **Risk:** A stronger backbone can overfit if the dataset is small and noisy.
- **Mitigation:**
  - Keep the loss design simple and robust.
  - Use early stopping on a held-out validation set.
  - Prefer regularization via dropout and weight decay instead of very complex objectives.

### 9.3 Incompatibility with Existing Code

- **Risk:** Model dimension or pooling assumptions for BGE-large may not match Qwen3-Embedding.
- **Mitigation:**
  - Explicitly check encoder hidden size and adjust the projection head accordingly.
  - Ensure pooling logic (last-token vs CLS/mean) is correct and consistently applied.

### 9.4 Leaderboard Variance

- **Risk:** Even if Qwen3-Embedding improves average embedding quality, leaderboard scores may fluctuate
  by ±1% due to data and evaluation variance.
- **Mitigation:**
  - Run multiple seeds and average results when feasible.
  - Use ensembling for Track A heads.
  - Only adopt major config changes if they show consistent benefits across seeds.

---

## 10. Summary

APPROACH v11 upgrades the embedding backbone from **BAAI/bge-large-en-v1.5** to **Qwen3-Embedding** while
keeping the proven Track A / Track B pipeline largely intact. The main goals are:

- Leverage state-of-the-art embedding quality for narrative similarity
- Respect all Codabench rules (independent story embeddings, compatible output formats)
- Minimize architectural churn by isolating the change to the backbone
- Provide a clear migration path and highlight risks and mitigations

By combining a stronger backbone with a stable training procedure, v11 aims to push performance beyond
the previous accuracy plateau without introducing unnecessary complexity.
