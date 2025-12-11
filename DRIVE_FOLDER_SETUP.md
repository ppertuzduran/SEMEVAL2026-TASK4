# Google Drive Folder Setup - Quick Reference (Git Workflow)

## 🎯 New Approach: Git for Code, Drive for Data

**Scripts**: Clone from GitHub (automatic)  
**Data**: Upload to Drive (one-time)  
**Models**: Saved to Drive (automatic)

---

## 📁 Required Folder Structure (SIMPLIFIED!)

```
MyDrive/
└── narrative_similarity/              ← EXACT NAME (or change in colab_utils.py)
    ├── data/                          ← Only upload data files!
    │   ├── dev_track_a.jsonl         ← Upload this
    │   └── dev_track_b.jsonl         ← Upload this
    ├── models/                        ← Auto-created during training
    └── data/prepared/                 ← Auto-created during training
```

**That's it!** No scripts needed in Drive anymore.

---

## ✅ What to Upload to Drive

| File/Folder | Location in Drive | Size | Required |
|-------------|-------------------|------|----------|
| `narrative_similarity` | `MyDrive/narrative_similarity/` | - | ✅ YES (create folder) |
| `data` | `MyDrive/narrative_similarity/data/` | - | ✅ YES (create folder) |
| `dev_track_a.jsonl` | `MyDrive/narrative_similarity/data/` | ~500KB | ✅ YES |
| `dev_track_b.jsonl` | `MyDrive/narrative_similarity/data/` | ~500KB | ✅ YES |

**Total upload**: 2 files (~1MB) + 2 folders

---

## ❌ What NOT to Upload (Use Git Instead)

```
❌ config.yaml              ← Get from GitHub
❌ training/colab_utils.py  ← Get from GitHub
❌ training/prepare_data.py ← Get from GitHub
❌ training/train_track_a.py ← Get from GitHub
❌ training/train_track_b.py ← Get from GitHub
❌ track_a.py               ← Get from GitHub
❌ track_b.py               ← Get from GitHub
❌ Any other .py files      ← Get from GitHub
```

**These come from `git clone` in Colab!**

---

## 📤 Auto-Generated Folders (During Training)

| Folder | Created By | Contents | Location |
|--------|-----------|----------|----------|
| `models/` | Training scripts | Trained checkpoints (~7GB) | Drive |
| `data/prepared/` | prepare_data.py | Processed data (~5MB) | Drive |
| `output/` | Training scripts | Logs | Drive |

---

## 📋 New Checklist Before Training

### One-Time Setup

```
□ Push code to GitHub repo
□ Created MyDrive/narrative_similarity/ folder on Drive
□ Created MyDrive/narrative_similarity/data/ folder
□ Uploaded dev_track_a.jsonl to narrative_similarity/data/
□ Uploaded dev_track_b.jsonl to narrative_similarity/data/
```

### In Colab (Every Session)

```
□ Enabled T4 GPU (Runtime → Change runtime type)
□ Run setup cell (installs packages)
□ Run git clone cell (gets code from GitHub)
□ Run mount Drive cell (connects to your data)
□ Run training cells
```

**Much simpler!** Only 2 files to upload vs 10+ before.

## 🔄 Changing the Root Folder Name

If you want a different name than `narrative_similarity`:

**In Colab:**
```python
paths = setup_colab_environment(
    project_name="my_custom_folder",  # Change this
    use_repo=True,
    repo_url="https://github.com/YOU/your-repo.git"
)
```

---

## ⚡ Quick Upload Via Drive Web

1. Go to https://drive.google.com
2. Click **New → Folder** → Name it `narrative_similarity`
3. Open `narrative_similarity` folder
4. Click **New → Folder** → Create `data`
5. Open `data` folder
6. Drag and drop: `dev_track_a.jsonl` and `dev_track_b.jsonl`
7. Done! ✓

---

## 🆚 Old vs New Workflow

### Old Workflow (Manual Upload)
```
Upload to Drive:
✅ config.yaml
✅ dev_track_a.jsonl
✅ dev_track_b.jsonl
✅ colab_utils.py
✅ prepare_data.py
✅ train_track_a.py
✅ train_track_b.py
───────────────────
Total: 7+ files to upload manually
Time: ~10 minutes
```

### New Workflow (Git) ⚡
```
Upload to Drive:
✅ dev_track_a.jsonl
✅ dev_track_b.jsonl
───────────────────
Total: 2 files to upload
Time: ~2 minutes

Scripts: git clone (automatic)
```

**5x faster setup!**

---

**Total Upload Size**: ~1MB (just data)  
**Time to Setup**: ~2 minutes  
**Training Time**: ~1 hour on T4 GPU

See **[COLAB_SETUP_NEW.md](COLAB_SETUP_NEW.md)** for complete Colab workflow.

