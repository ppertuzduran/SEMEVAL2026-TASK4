# Narrative Similarity - Colab Workflow (Track A & Track B)

Concise guide to train and infer on Google Colab with Google Drive.

## Setup
- Python 3.11 on Colab GPU (T4).
- Data in Drive:
  - `MyDrive/narrative_similarity/data/dev_track_a.jsonl`
  - `MyDrive/narrative_similarity/data/dev_track_b.jsonl`

## Colab Configuration
```python
!nvidia-smi
!pip install -q transformers sentence-transformers datasets scikit-learn pyyaml accelerate

# Clone repo
REPO_URL = "https://github.com/ppertuzduran/SEMEVAL2026-TASK4.git"
!git clone {REPO_URL} /content/project
%cd /content/project

# Mount Drive and setup
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/project/training')
from colab_utils import setup_colab_environment
paths = setup_colab_environment(project_name="narrative_similarity")
```

## Drive Paths in `config.yaml`
- Set after mounting:
  - `data.dev_track_a: /content/drive/MyDrive/narrative_similarity/data/dev_track_a.jsonl`
  - `data.dev_track_b: /content/drive/MyDrive/narrative_similarity/data/dev_track_b.jsonl`
  - `data.output_dir: /content/project/output`
  - `track_a.model_save_path: /content/drive/MyDrive/narrative_similarity/models/track_a_cross_encoder`
  - `track_b.model_save_path: /content/drive/MyDrive/narrative_similarity/models/track_b_embedder`

## Step-by-Step Workflow
1) Prepare data
```bash
python training/prepare_data.py
```
2) Train Track B (bi-encoder, curriculum)
```bash
python training/train_track_b.py
```
   - Uses MNR → PairwiseSoftmax → Triplet.
   - If T4 OOM: lower `batch_size_pairwise`, `max_seq_length`, or switch `base_model` to `bge-base`.

3) Train Track A (cross-encoder, with B→A distill prior)
```bash
python training/train_track_a.py
```
   - Requires Track B checkpoint at `track_a.teacher_biencoder_path`.

4) (Optional) Distill A → B
   - In `config.yaml`: set `track_b.distill_from_teacher: true`, `teacher_model_path` to Track A ckpt.
   - Rerun `python training/train_track_b.py` (or disable other phases to run distill-only).

5) Inference
```bash
python track_b.py   # embeddings, saves track_b.npy
python track_a.py   # predictions, saves track_a.jsonl (ensemble if enabled)
```

6) Evaluation
```bash
python scripts/eval_local.py --track both
```

7) Save results to Drive
```bash
mkdir -p /content/drive/MyDrive/narrative_similarity/output
cp -r output/* /content/drive/MyDrive/narrative_similarity/output/
```

## Minimal Local Notes (optional)
- Python 3.11 venv; install PyTorch CUDA wheel matching your GPU; `pip install -r requirements.txt`.
- Run inference locally (slower): `python track_b.py`, `python track_a.py`, then `python scripts/eval_local.py --track both`.
