# Google Drive Folder Setup - Quick Reference

## 📁 Required Folder Structure

```
MyDrive/
└── narrative_similarity/              ← EXACT NAME (or change in colab_utils.py line 26)
    ├── config.yaml
    ├── data/
    │   ├── dev_track_a.jsonl
    │   └── dev_track_b.jsonl
    └── training/
        ├── colab_utils.py
        ├── prepare_data.py
        ├── train_track_a.py
        └── train_track_b.py
```

## ✅ Folder Names (Must Match Exactly)

| Folder/File | Location in Drive | Required |
|-------------|-------------------|----------|
| `narrative_similarity` | `MyDrive/narrative_similarity/` | ✅ YES |
| `data` | `MyDrive/narrative_similarity/data/` | ✅ YES |
| `training` | `MyDrive/narrative_similarity/training/` | ✅ YES |
| `config.yaml` | `MyDrive/narrative_similarity/` | ✅ YES |
| `dev_track_a.jsonl` | `MyDrive/narrative_similarity/data/` | ✅ YES |
| `dev_track_b.jsonl` | `MyDrive/narrative_similarity/data/` | ✅ YES |
| `colab_utils.py` | `MyDrive/narrative_similarity/training/` | ✅ YES |
| `prepare_data.py` | `MyDrive/narrative_similarity/training/` | ✅ YES |
| `train_track_a.py` | `MyDrive/narrative_similarity/training/` | ✅ YES |
| `train_track_b.py` | `MyDrive/narrative_similarity/training/` | ✅ YES |

## 📤 Auto-Generated Folders (Created During Training)

| Folder | Created By | Contents |
|--------|-----------|----------|
| `models/` | Training scripts | Trained model checkpoints |
| `data/prepared/` | prepare_data.py | Processed training data |
| `output/` | Training scripts | Logs and temporary files |

## 🔄 Changing the Root Folder Name

If you want to use a different name than `narrative_similarity`, edit `training/colab_utils.py`:

```python
# Line 26 - change the default project name
def setup_colab_environment(project_name="YOUR_CUSTOM_NAME"):
```

Or when calling in Colab:

```python
from colab_utils import setup_colab_environment
paths = setup_colab_environment(project_name="my_project")
```

## 📋 Checklist Before Training

```
□ Created MyDrive/narrative_similarity/ folder
□ Created MyDrive/narrative_similarity/data/ folder
□ Created MyDrive/narrative_similarity/training/ folder
□ Uploaded config.yaml to narrative_similarity/
□ Uploaded dev_track_a.jsonl to narrative_similarity/data/
□ Uploaded dev_track_b.jsonl to narrative_similarity/data/
□ Uploaded colab_utils.py to narrative_similarity/training/
□ Uploaded prepare_data.py to narrative_similarity/training/
□ Uploaded train_track_a.py to narrative_similarity/training/
□ Uploaded train_track_b.py to narrative_similarity/training/
□ Opened Google Colab notebook
□ Enabled T4 GPU in Colab (Runtime → Change runtime type)
```

## 🎯 Direct Drive Links

After setup, your files should be at:

- Config: `https://drive.google.com/drive/folders/YOUR_ID` → `narrative_similarity/config.yaml`
- Data: `https://drive.google.com/drive/folders/YOUR_ID` → `narrative_similarity/data/`
- Scripts: `https://drive.google.com/drive/folders/YOUR_ID` → `narrative_similarity/training/`

## ⚡ Quick Upload Via Drive Web

1. Go to https://drive.google.com
2. Click **New → Folder** → Name it `narrative_similarity`
3. Open `narrative_similarity` folder
4. Click **New → Folder** → Create `data` and `training`
5. Drag and drop files to respective folders
6. Done! ✓

---

**Total Upload Size**: ~5MB (config + data + scripts)  
**Time to Setup**: ~5 minutes  
**Training Time**: ~1 hour on T4 GPU

