# APPROACH v10 – Focused Improvements for Tracks A & B

This version of the approach document focuses on **what to change going forward**, not on restating the whole existing pipeline.

You already have:

- Track B: a strong bi‑encoder (BGE‑large) with a projection head and composite loss.
- Track A: a lightweight MLP head on top of Track B embeddings.
- One experiment runner: `run_track_b_experiments.py`.

The goal of v10 is to:
1. Turn Track B into a **properly tuned metric learner** via systematic hyperparameter sweeps.
2. Give Track A its **own experiment runner** with sensible search spaces.
3. Make `README.md` explicit about **how to run sweeps** and **where to change ranges** for non‑experts.

---

## 1. Design Goals

- **Respect the competition rules**: Track B must produce embeddings story‑by‑story; Track A must operate on triples.
- **Exploit what you already built**: keep the current architecture; optimize the hyperparameters.
- **Stay Colab‑friendly**: small, discrete search spaces instead of giant Bayesian optimization.

---

## 2. Track B – Embedding Model

### 2.1 Training skeleton (kept as‑is)

We keep the existing training logic in `train_track_b.py`:

- Base model: `BAAI/bge-large-en-v1.5`.
- Projection head: linear (1024 → projection_dim) + normalization.
- Losses: MultipleNegativesRankingLoss + CompositeLoss (margin ranking + pairwise softmax).
- Optional phases: hard negative mining, SimCSE‑style consistency loss, temperature/margin sweep.

We **do not** change the core trainer in v10. All changes are about **how we call it**.

### 2.2 Make `run_track_b_experiments.py` the main entry point

From v10 on, **Track B is not trained directly with `train_track_b.py` anymore** in normal usage. Instead:

- Users call:

  ```bash
  python scripts/run_track_b_experiments.py
  ```

- That script:
  - Backs up `config.yaml` → `config.yaml.backup`.
  - Runs a baseline training with current settings.
  - Runs several incremental experiments, each mutating a *small set* of hyperparameters.
  - Only keeps an experiment if validation accuracy improves.
  - Writes all results to `experiments_track_b.json`.
  - Updates `config.yaml` with the best‑performing configuration.

This gives you **reproducible, logged model selection** without touching the trainer.

### 2.3 Track B hyperparameters to search

We expose a **compact, hand‑picked search space** via `config.yaml` and `run_track_b_experiments.py`.

#### 2.3.1 `config.yaml` ranges (Track B section)

Add/adjust the following under `track_b` in `config.yaml`:

```yaml
track_b:
  # Projection & regularization
  projection_dim: 512        # optionally test 384 vs 512 via experiments
  weight_decay: 0.01         # backbone weight decay
  projection_weight_decay: 0.05   # tuned via experiments

  # Loss weights
  loss_weights:
    margin:   1.0            # margin ranking loss weight
    pairwise: 0.5            # pairwise softmax loss weight
    mnr:      0.5            # MultipleNegativesRankingLoss weight
    simcse:   0.0            # 0 by default; only enabled by experiments

  # Temperature × margin sweep (used when enable_hyperparam_sweep: true)
  temperature_grid: [0.5, 0.7, 0.9, 1.1]
  margin_grid:      [0.05, 0.1, 0.2, 0.3]

  # Hard negative mining defaults (can be overridden by experiments)
  hard_negative_k:      5
  hard_negative_epochs: 2

  # Training schedule (example values)
  epochs: 4
  batch_size: 16
  learning_rate: 2e-5
```

These are baseline values; the sweeps below decide which ones to keep.

#### 2.3.2 Experiment 1 – Regularization sweep

At the top of `scripts/run_track_b_experiments.py`, define the regularization search space:

```python
REG_DROPOUTS = [0.0, 0.1, 0.2]
REG_WEIGHT_DECAYS = [0.03, 0.06, 0.1]
```

Then, in the section currently labeled **“Experiment 2: + Better Regularization”**, replace the single hard‑coded setting with a loop:

```python
best_reg_result = None

for d in REG_DROPOUTS:
    for wd in REG_WEIGHT_DECAYS:
        cfg = load_config(backup_path)
        cfg['track_b']['projection_dropout'] = d
        cfg['track_b']['projection_weight_decay'] = wd

        result = run_experiment(f'+ Reg (drop={d}, wd={wd})', cfg, config_path)

        if best_reg_result is None or result['accuracy'] > best_reg_result['accuracy']:
            best_reg_result = result
            base_config = cfg
            baseline_accuracy = result['accuracy']
```

This turns regularization into a **small grid search** over dropout and projection‑head weight decay.

#### 2.3.3 Experiment 2 – Temperature × margin sweep

No structural changes needed; we just rely on the `temperature_grid` and `margin_grid` from `config.yaml`.

- `run_track_b_experiments.py` sets `enable_hyperparam_sweep: true` in a temporary config.
- `train_track_b.py` runs through all temperature/margin combinations and writes the best ones into `best_hyperparams.json` and your logs.
- The experiment script reads the final accuracy and decides whether to keep it.

If you want to tweak the range, you only touch `config.yaml` (not the trainer).

#### 2.3.4 Experiment 3 – Hard negative mining sweep (optional)

At the top of `run_track_b_experiments.py` add:

```python
HARD_K      = [3, 5, 7]
HARD_EPOCHS = [1, 2]
```

Then, instead of a single `(k=5, epochs=2)` in the “Hard Negatives” experiment, loop:

```python
best_hard_result = None

for k in HARD_K:
    for e in HARD_EPOCHS:
        cfg = load_config(backup_path)
        cfg['track_b']['enable_hard_negatives'] = True
        cfg['track_b']['hard_negative_k'] = k
        cfg['track_b']['hard_negative_epochs'] = e

        result = run_experiment(f'+ HardNeg (k={k}, e={e})', cfg, config_path)

        if best_hard_result is None or result['accuracy'] > best_hard_result['accuracy']:
            best_hard_result = result
            base_config = cfg
            baseline_accuracy = result['accuracy']
```

Again, the trainer is unchanged; we just vary the config around it.

#### 2.3.5 Experiment 4 – SimCSE toggle

Leave SimCSE as a final, optional regularization:

- Baseline: `simcse_weight = 0.0` (off).
- Experiment: set `simcse_weight` to something small, e.g. `0.1` in `config.yaml` or directly in the experiment script.

Only keep it if it improves dev accuracy.

---

## 3. Track A – Triple Classification Model

For Track A, the main idea in v10 is the same: **do not redesign the trainer**, but **add an experiment runner** and standardize the search space.

### 3.1 Keep the model architecture

We keep:

- Track B as the encoder (usually loaded frozen).
- A single MLP head operating on concatenations of `(anchor, candidate)` embeddings:
  - `φ(x) = [ e_anchor, e_x, |e_anchor - e_x|, e_anchor ⊙ e_x ]`.
  - MLP: `4d → hidden_dim → 1` with GELU and dropout.
- Optional distillation from Track B’s cosine‑based decision via `distill_weight` and `distill_temperature`.

No architecture change in v10; only hyperparameters.

### 3.2 New script: `run_track_a_experiments.py`

Create a new script (e.g. in `scripts/run_track_a_experiments.py`) mirroring the Track B runner:

- Backs up `config.yaml` → `config.yaml.backup`.
- Runs a baseline training using `training/train_track_a.py`.
- Applies several incremental improvements (different hyperparameter combinations).
- Keeps only configurations that improve dev accuracy.
- Saves all results to `experiments_track_a.json`.
- Writes the best configuration back to `config.yaml`.

#### 3.2.1 Hyperparameter ranges for Track A

Define ranges at the top of `run_track_a_experiments.py`:

```python
HIDDEN_DIMS     = [512, 1024]
DROPOUTS        = [0.1, 0.2, 0.3]
LEARNING_RATES  = [5e-5, 1e-4, 2e-4]
DISTILL_WEIGHTS = [0.0, 0.3, 0.5]
FREEZE_OPTIONS  = [True, False]  # unfreeze only with lower LR / fewer epochs
```

This keeps the search space small and easy to modify.

#### 3.2.2 Suggested experiment schedule

**Step 0 – Baseline**  
Use the existing `track_a` config as baseline. Run `train_track_a.py`, parse the validation accuracy from logs (e.g. last “validation accuracy” line), and store it.

**Step 1 – Distillation sweep**  
For each `distill_weight` in `DISTILL_WEIGHTS` (and a fixed `distill_temperature`, e.g. 2.0):

- Clone the baseline config.
- Set `track_a.distill_weight` and `track_a.distill_temperature`.
- Train and record accuracy.
- Keep the best setting (if it beats baseline).

**Step 2 – Head size × dropout sweep**  
For the best distillation config:

- Loop over `HIDDEN_DIMS × DROPOUTS`.
- Set `track_a.mlp_hidden_dim` and `track_a.mlp_dropout`.
- Train and record accuracy.
- Keep the best pair.

**Step 3 – Learning rate sweep**  
For the best head config:

- Loop over `LEARNING_RATES`.
- Set `track_a.learning_rate`.
- Train and record accuracy.
- Keep the best one.

**Step 4 – Light joint fine‑tuning (optional)**  

Only if you want to test slight improvements and you’re not overfitting:

- For the best config so far:
  - Set `track_a.freeze_track_b: False`.
  - Set a **small LR** for encoder parameters (e.g. via a second param group in `train_track_a.py`, or by lowering global LR in config to `2e-5` and updating param‑group logic in the trainer).
  - Reduce `track_a.epochs` (e.g. to `3`).
- Train and keep the unfreezed variant only if validation accuracy improves.

This gives you a controlled way to see if letting Track A fine‑tune the encoder helps, without making it the default.

### 3.3 Recommended default values in `config.yaml` (Track A)

Under `track_a` in `config.yaml`, set reasonable starting values:

```yaml
track_a:
  mlp_hidden_dim:      512
  mlp_dropout:         0.2
  learning_rate:       1e-4
  distill_weight:      0.3
  distill_temperature: 2.0
  freeze_track_b:      true
  epochs:              5
  batch_size:          16
```

These will be the starting point for `run_track_a_experiments.py` and will be overwritten by the best‑performing combination.

---

## 4. README.md Updates (Hyperparameter Sweeps)

Below is a **ready‑to‑paste** section for `README.md` so non‑experts can run the sweeps and tweak ranges.

### 4.1 Section to add to README.md

```markdown
## Hyperparameter Sweeps

This project includes simple scripts to run **hyperparameter sweeps** for both tracks.
They are designed so you can try a few configurations without touching the training code.

---

### Track B – Embedding Model

Use `scripts/run_track_b_experiments.py` to search for better hyperparameters for the
embedding (Track B) model.

```bash
python scripts/run_track_b_experiments.py
```

What this script does:

1. Backs up your current `config.yaml` to `config.yaml.backup`.
2. Trains a **baseline** model with the current settings.
3. Tries several incremental improvements:
   - Different projection dropout and projection‑head weight decay.
   - Temperature × margin sweep for the pairwise loss.
   - Hard negative mining settings.
   - Optional SimCSE regularization.
4. Only keeps experiments that improve validation accuracy.
5. Saves all results in `experiments_track_b.json`.
6. Writes the best‑performing configuration back into `config.yaml`.

**Where to change ranges**

- Temperature and margin grids are defined in `config.yaml` under `track_b`:

  ```yaml
  track_b:
    temperature_grid: [0.5, 0.7, 0.9, 1.1]
    margin_grid:      [0.05, 0.1, 0.2, 0.3]
  ```

- Regularization ranges (projection dropout and projection weight decay) are defined as Python
  lists at the top of `scripts/run_track_b_experiments.py`:

  ```python
  REG_DROPOUTS = [0.0, 0.1, 0.2]
  REG_WEIGHT_DECAYS = [0.03, 0.06, 0.1]
  ```

  To try more or fewer values, just edit these lists.

- Hard negative settings are also controlled in `scripts/run_track_b_experiments.py`:

  ```python
  HARD_K      = [3, 5, 7]
  HARD_EPOCHS = [1, 2]
  ```

  The script loops over these values and keeps only the best combination.

---

### Track A – Triple Classification

We recommend using a similar script for Track A: `scripts/run_track_a_experiments.py`.
This script runs the triple classifier training with different hyperparameters and selects
the best configuration.

```bash
python scripts/run_track_a_experiments.py
```

This script should:

1. Backup `config.yaml` to `config.yaml.backup`.
2. Run a baseline training with the current `track_a` settings.
3. Try different combinations of:

   - `mlp_hidden_dim` (e.g. 512, 1024)
   - `mlp_dropout` (e.g. 0.1, 0.2, 0.3)
   - `learning_rate` (e.g. 5e-5, 1e-4, 2e-4)
   - `distill_weight` (e.g. 0.0, 0.3, 0.5)
   - `freeze_track_b` (True vs False, for light joint fine‑tuning)

4. Parse the validation accuracy from the training logs.
5. Save all results to `experiments_track_a.json`.
6. Update `config.yaml` with the best‑performing configuration.

**Where to change ranges**

At the top of `scripts/run_track_a_experiments.py`, define the ranges as simple Python lists:

```python
HIDDEN_DIMS     = [512, 1024]
DROPOUTS        = [0.1, 0.2, 0.3]
LEARNING_RATES  = [5e-5, 1e-4, 2e-4]
DISTILL_WEIGHTS = [0.0, 0.3, 0.5]
FREEZE_OPTIONS  = [True, False]
```

To explore a different range, just edit these lists; you do **not** need to touch the
training code.
```

---

## 5. Summary of v10 Changes

- **Track B**:
  - Trainer architecture unchanged.
  - `run_track_b_experiments.py` becomes the primary interface for training.
  - Regularization, temperature/margin, and hard negative settings are searched via small grids
    defined in `config.yaml` and at the top of the script.

- **Track A**:
  - Model architecture unchanged.
  - New `run_track_a_experiments.py` script controls hyperparameter sweeps for the MLP head and
    distillation strength.
  - Ranges are declared in one place (lists at the top of the script) so they are easy to modify.

- **README**:
  - New section explaining, in simple terms, how to run sweeps and where to edit the search ranges,
    aimed at users who are not familiar with hyperparameter tuning.

This approach focuses squarely on **systematic tuning** around your already strong models, which is the
most realistic way to push accuracy beyond the current ~0.94/0.95 without over‑engineering the system.
