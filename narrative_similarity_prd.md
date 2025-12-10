# PRD – Narrative Similarity Tracks (A & B)

## 1. Context

We participate in a narrative-similarity shared task with two tracks:

- **Track A – Triple-wise choice:** Given an anchor story and two candidate stories (A, B), the system must choose which candidate is narratively closer to the anchor.
- **Track B – Story embeddings:** Given a single story, the system must output a vector representation whose cosine similarity correlates with narrative similarity. Evaluation is done indirectly via the Track A triples.

Baselines provided by organizers:

- **Track A baseline:** Direct prompting of `gpt-4o-mini` given (anchor, A, B).fileciteturn0file0  
- **Track B baseline:** `sentence-transformers/all-MiniLM-L6-v2` used to encode all Track B stories; evaluation uses cosine similarity and Track A labels.fileciteturn0file1  

Constraints & environment:

- GPU: **RTX 4050 6 GB** (single card).
- Data size: ~**400 triples**.
- Embedding dimensionality for Track B must be between **10 and 8192**.
- Track B inference must encode **individual stories only** (no triple-specific embeddings at inference).

Goal: Design a modern, practical, and reproducible approach that significantly improves accuracy over baselines on both tracks and is feasible on the given hardware & data size.

---

## 2. Goals & Non-goals

### 2.1 Primary Goals

1. **Improve Track A accuracy** over GPT-4o-mini baseline on dev data.
2. **Improve Track B accuracy** over `all-MiniLM-L6-v2` baseline when evaluated with organizers’ triple-wise metric.
3. **Ensure submission compatibility:** Output formats identical to baselines, runnable with `python track_a.py` and `python track_b.py`.
4. **Keep all training local** (offline) using the 6 GB GPU.

### 2.2 Secondary Goals

- Reuse as much logic and code between tracks as possible.
- Provide deterministic and fast inference (no external API calls at evaluation time).
- Provide clean, well-structured code for future experimentation.

### 2.3 Non-goals

- Beating large closed LLMs with huge prompting budgets.
- Building a large multi-lingual or domain-general embedding model; scope is the competition domain only.
- Perfect explainability; some interpretability is nice but not a primary objective.

---

## 3. High-level Strategy

We use a **two-model strategy**:

1. **Bi-encoder / sentence-embedding model (shared core, Track B-focused):**
   - Start from a strong, compact sentence-transformer (e.g. `all-MiniLM-L6-v2` or `bge-small`).
   - Fine-tune on the provided triples using **metric-learning losses** (triplet / contrastive / cosine-similarity) to shape the embedding space around narrative similarity.
   - Export this model as the **Track B submission**.

2. **Cross-encoder ranking model (Track A-focused):**
   - Start from a small/medium transformer (e.g. `deberta-v3-base` or `bert-base-uncased`).
   - Input format: `[CLS] anchor [SEP] story A [SEP] story B [SEP]`.
   - Train as a **binary classifier** (A-closer vs B-closer) using cross-entropy.
   - Optionally, distill knowledge from the bi-encoder (teacher–student or auxiliary loss).

Rationale:

- Cross-encoders are typically **SOTA for pairwise ranking** on small datasets.
- Bi-encoders are needed for Track B, and metric learning using Track A labels is expressly allowed as long as inference is single-story-only.
- Starting from strong general-purpose encoders avoids overfitting with tiny data.

---

## 4. Data Design

### 4.1 Available Data

- **Track A JSONL:** Each row has `anchor_text`, `text_a`, `text_b`, `text_a_is_closer` (boolean ground-truth).fileciteturn0file0
- **Track B JSONL:** Each row has `text` (unique stories for embedding). Evaluation built via Track A labels.fileciteturn0file1

### 4.2 Derived Datasets

To maximize signal from ~400 triples:

1. **Triplet dataset for metric learning (Track B training):**
   - For each triple:
     - Anchor = `anchor_text`
     - Positive = `text_a` if `text_a_is_closer == True`, else `text_b`
     - Negative = the other candidate.
   - Store as `(anchor, positive, negative)`.

2. **Pairwise dataset (anchor–candidate pairs):**
   - For each triple, create two pairs:
     - `(anchor, positive, label=1.0)`
     - `(anchor, negative, label=0.0)`

3. **Cross-encoder dataset (Track A):**
   - Use original triple as a single row:
     - Inputs: `(anchor, text_a, text_b)`
     - Label: `0` if A-closer, `1` if B-closer.

4. **Data augmentation (lightweight):**
   - Optional but useful given small data:
     - Randomly swap A/B in some triples and invert the label.
     - Use simple textual augmentations like:
       - Mild noise: random deletion of low-information words.
       - Sentence-level shuffles for obviously unordered lists (if present).
   - Avoid heavy paraphrasing to prevent changing narrative semantics.

5. **Data splits & validation:**
   - Use **k-fold cross-validation** (e.g. 5-fold) on Track A triples to:
     - Tune hyperparameters.
     - Monitor variance and prevent overfitting.
   - Reserve a small **hold-out dev set** (~15–20%) for final model selection.

---

## 5. Track B – Embedding Model

### 5.1 Model Choice

- Base: `sentence-transformers`-compatible model with ~384–768 dimensions and small footprint, e.g.:
  - `all-MiniLM-L6-v2` (baseline) or
  - A similar-sized modern model (e.g. BGE small variant), as long as it fits 6 GB with a reasonable batch size.

We keep dimensionality within allowed limits (e.g. **384-d**).

### 5.2 Training Objective

We fine-tune with a combination of **metric-learning** losses:

1. **Triplet loss (primary):**
   - For each `(anchor, positive, negative)`:
     - Minimize `max(0, margin + d(anchor, positive) - d(anchor, negative))` with cosine distance.
   - Margin (hyperparameter): ~0.2–0.4.

2. **CosineSimilarityLoss (optional auxiliary):**
   - For pairwise data:
     - Positive pair label ≈ 1.0.
     - Negative pair label ≈ 0.0–0.2.
   - Encourages smooth similarity scores, not just ranking.

3. **In-batch negatives:**
   - Use multiple triples per batch so that other positives in the batch act as negatives (MultipleNegativesRankingLoss style).

Implementation via `sentence-transformers` library:

- Build two datasets:
  - A `TripletDataset` for TripletLoss.
  - A `SentenceLabelDataset` for cosine similarity.
- Use the `JointLoss` or manual multi-loss training (alternate mini-batches).

### 5.3 Training Setup

- **Batch size:** 16–32 (depending on VRAM); use gradient accumulation if needed.
- **Max seq length:** 256–384 tokens (stories trimmed with head/tail strategy).
- **Optimizer:** AdamW with linear warmup + cosine decay.
- **Learning rate:** 2e-5 – 5e-5.
- **Epochs:** 5–20 (tiny dataset; use early stopping based on Track A-style dev eval).
- **Regularization:**
  - Weight decay (0.01).
  - Dropout from base model.
  - Shuffle triples every epoch.

### 5.4 Validation & Model Selection

Because evaluation is triple-wise (Track A style), we use the same protocol as in the baseline:

1. After each epoch:
   - Encode all unique stories from **Track B** file.
   - Build an embedding lookup `text -> vector`.
   - Run the provided `evaluate` function using **Track A dev triples** (same as baseline).fileciteturn0file1  
2. Track accuracy and choose the checkpoint with best dev performance.

### 5.5 Inference & Submission

- Final Track B script (`track_b.py`) will:

  1. Load the fine-tuned sentence-transformer.
  2. Load `dev_track_b.jsonl`, encode `data["text"]` to numpy array.
  3. Save embeddings as `output/track_b.npy` (same format as baseline).fileciteturn0file1  

- No triple information is used at inference; each story is encoded independently, respecting competition rules.

---

## 6. Track A – Cross-encoder Ranking Model

### 6.1 Model Choice

- Base: a small/medium pretrained encoder with strong NLU:
  - Candidate models: `deberta-v3-base`, `roberta-base`, or `bert-base-uncased`.
  - Pick the best one fitting within VRAM + training limits (likely `deberta-v3-base` or `roberta-base`).

- Architecture:
  - Input: `[CLS] anchor [SEP] story A [SEP] story B [SEP]`.
  - Output: 2-logit classification head (A-closer vs B-closer).

Alternative (optional) architecture:

- Multi-input formulation with two forward passes per triple:
  - Pair 1: `[CLS] anchor [SEP] story A`
  - Pair 2: `[CLS] anchor [SEP] story B`
  - Combine pooled outputs (e.g. difference or concatenation) and feed to a classifier.
- Chosen only if VRAM or max-seq-length is a constraint.

### 6.2 Training Objective

- **Primary loss:** Cross-entropy over {A-closer, B-closer}.
- **Optional auxiliary loss:** KL-divergence distillation from:
  - (a) Our best bi-encoder similarity scores, or
  - (b) GPT-4o-mini soft predictions (if we generate them offline once).

This yields a model that leverages both **pairwise ranking supervision and embedding-based structure**.

### 6.3 Training Setup

- **Input length:** Up to 512 tokens; stories truncated intelligently:
  - Retain beginning and end segments of each story (to keep narrative arc).
- **Batch size:** 8–16 triple instances (limited by VRAM).
- **Learning rate:** ~2e-5.
- **Epochs:** 5–15, early stopping on dev accuracy with k-fold validation.
- **Regularization:** dropout, weight decay, label smoothing (e.g. 0.05).

### 6.4 Validation & Selection

- Use **stratified k-fold** (e.g. 5-fold) across triples:
  - For each fold:
    - Train on k-1 folds.
    - Evaluate accuracy on held-out fold.
  - Average accuracy and variance give robustness signal.
- Final model:
  - Either retrain on full dataset using hyperparameters selected from CV.
  - Or ensemble k models at inference (majority vote / averaged logits). (Optional; might cost 5x runtime but triples are small.)

### 6.5 Inference & Submission

- Replace baseline logic in `track_a.py` with local model inference:
  - Remove random and GPT-based branches.
  - Load trained cross-encoder from local checkpoint.
  - For each row in `dev_track_a.jsonl`, compute logits and choose A if `logit_A > logit_B`.
  - Populate `predicted_text_a_is_closer` and write output JSONL exactly as baseline.fileciteturn0file0  

---

## 7. Use of GPT-4o(-mini) & Data Augmentation (Optional Enhancements)

While primary models are local, LLMs can still help **offline**:

1. **Soft label generation:**
   - Use GPT-4o-mini or a stronger model (if available) with a more elaborate prompt and few-shot examples.
   - Gather soft probabilities `P(A-closer)` for each triple.
   - Use these as:
     - Distillation targets for the cross-encoder (auxiliary KL loss).
     - Additional weighting for noisy examples (disagree strongly with our models).

2. **Augmented triples:**
   - For each story, ask GPT to generate minor variants (paraphrases) that preserve the narrative.
   - Use them to enrich triplet and pairwise training for the embedding model.

These enhancements are optional; they raise compute and complexity but can push closer to SOTA performance for such small datasets.

---

## 8. Engineering & Implementation Plan

### 8.1 Repository Structure

- `track_a.py` – modified to use trained cross-encoder (same output format).  
- `track_b.py` – modified to use fine-tuned sentence-transformer (same output format).  
- `models/`
  - `track_b_embedder/` (saved sentence-transformer).  
  - `track_a_cross_encoder/` (saved transformer + tokenizer).  
- `training/`
  - `prepare_data.py` – generates triplet & pairwise datasets from Track A file.  
  - `train_track_b.py` – fine-tunes sentence-transformer for embeddings.  
  - `train_track_a.py` – trains cross-encoder ranking model.  
- `configs/`
  - YAML/JSON configs for hyperparameters (model name, batch size, lr, epochs, etc.).  
- `scripts/`
  - `run_all.sh` – run both tracks and zip outputs.  
  - `eval_local.py` – re-implements competition metric for sanity checks.

### 8.2 Step-by-step Execution

1. **Data preparation:**
   - Parse `dev_track_a.jsonl` and `dev_track_b.jsonl`.
   - Generate triplets and pairwise pairs.
   - Save as `data/triplets.jsonl`, `data/pairs.jsonl`.

2. **Train Track B embedding model:**
   - Run `train_track_b.py`:
     - Load base encoder.
     - Train with metric-learning losses.
     - Evaluate after each epoch using baseline `evaluate` logic.
   - Export best checkpoint to `models/track_b_embedder`.

3. **Train Track A cross-encoder:**
   - Run k-fold cross-validation via `train_track_a.py`.
   - Select hyperparameters with best average accuracy.
   - Train final model on full data and save to `models/track_a_cross_encoder`.

4. **Wire up inference:**
   - Update `track_a.py` and `track_b.py` to:
     - Load saved models.
     - Run inference on dev data.
     - Generate outputs in `outputs/`.fileciteturn0file0turn0file1  

5. **Submission packaging:**
   - `zip -j your_submission.zip outputs/*`
   - Upload to CodaBench.

### 8.3 Performance & Resource Considerations

- Ensure inference fits within runtime constraints:
  - Bi-encoder: encode ~400–1000 stories – trivial on RTX 4050.
  - Cross-encoder: evaluate ~400 triples – also trivial, even with 2–3 models ensembled.
- GPU memory:
  - Cap max sequence length and batch sizes to stay under 6 GB.
  - Mixed precision training (FP16) to reduce memory footprint and speed up training.

---

## 9. Risks & Mitigations

1. **Overfitting on tiny dataset (~400 triples):**
   - Use strong pretrained models (no training from scratch).
   - Employ cross-validation and early stopping.
   - Limit number of epochs and monitor validation accuracy curve.
   - Consider simple ensembling to reduce variance.

2. **Label noise / ambiguous narrative similarity:**
   - Use label smoothing in cross-entropy.
   - Use soft labels from GPT-based distillation when available.
   - Down-weight examples where models and GPT strongly disagree.

3. **Rule violation for Track B (embedding dependence on triples at inference):**
   - Strict design: embedding function is `f(text)` only.
   - Use Track A labels **only during training**.
   - At inference, call encoder once per story; no triple-wise post-processing.

4. **Hardware limitations (6 GB):**
   - Choose compact models (MiniLM, RoBERTa-base/DeBERTa-base).
   - Mixed-precision (FP16).
   - Gradient accumulation when needed.

---

## 10. Success Metrics

1. **Primary metric:** Accuracy on the Triple-wise dev set (same as baseline scripts).fileciteturn0file0turn0file1  
2. **Target improvements (approximate):**
   - **Track A:** +5–10 percentage points over GPT-4o-mini baseline.
   - **Track B:** +5–10 percentage points over `all-MiniLM-L6-v2` baseline.
3. **Engineering quality:**
   - Training runs reproducible via config files and fixed seeds.
   - Inference scripts (`track_a.py`, `track_b.py`) require no manual modifications for submission.

---

## 11. Future Extensions (Out of Scope but Possible)

- Joint multi-task training where a single encoder feeds both:
  - A similarity head (for Track B) and
  - A classification head (for Track A).
- Larger distillation from a strong LLM cross-encoder teacher to a smaller student.
- Multi-lingual support or style-conditioning in embeddings.
- More advanced regularization (e.g. MixUp in embedding space, R-Drop).

---

## 12. Summary

We will:

1. **Fine-tune a compact sentence-transformer** with triplet and contrastive objectives using Track A labels to produce **narrative-aware embeddings** for Track B.
2. **Train a cross-encoder ranking model** on triples for Track A, optionally distilled from the embedding model and/or an LLM teacher.
3. **Keep inference simple and local**, preserving the baselines’ input/output formats while substantially improving accuracy under the given hardware and data constraints.
