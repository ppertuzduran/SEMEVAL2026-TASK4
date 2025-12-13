"""
Training script for Track A: Lightweight MLP Head over Track B embeddings (v2).

Google Colab training script.

This script implements the v2 approach from APPROACH.md:
- Lightweight MLP head operating on frozen Track B embeddings
- Pairwise feature construction (concat, diff, element-wise product)
- Distillation from Track B cosine similarities
- K-fold cross-validation for robustness
- Head-only training (freeze Track B) or joint fine-tuning
- Removes obsolete B→A distillation from cross-encoder
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml
import pandas as pd
from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, print_gpu_info


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class MLPHead(nn.Module):
    """
    Lightweight MLP head for pairwise ranking over embeddings.
    
    Architecture (from APPROACH.md section 4.1):
    - Input: pairwise features φ(a) or φ(b) of dimension 4d
    - Layer 1: Linear(4d → 2d) + GELU + Dropout
    - Layer 2: Linear(2d → 1)
    - Output: scalar score
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        super(MLPHead, self).__init__()
        self.embedding_dim = embedding_dim
        input_dim = 4 * embedding_dim  # [e1, e2, |e1-e2|, e1⊙e2]
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def construct_pairwise_features(self, anchor_emb, candidate_emb):
        """
        Construct pairwise features: [e_anchor, e_cand, |e_anchor - e_cand|, e_anchor ⊙ e_cand]
        
        Args:
            anchor_emb: (batch, dim)
            candidate_emb: (batch, dim)
        
        Returns:
            features: (batch, 4*dim)
        """
        diff = torch.abs(anchor_emb - candidate_emb)
        prod = anchor_emb * candidate_emb
        features = torch.cat([anchor_emb, candidate_emb, diff, prod], dim=1)
        return features
    
    def forward(self, anchor_emb, candidate_emb):
        """
        Forward pass through MLP head.
        
        Args:
            anchor_emb: (batch, dim)
            candidate_emb: (batch, dim)
        
        Returns:
            score: (batch, 1)
        """
        features = self.construct_pairwise_features(anchor_emb, candidate_emb)
        score = self.mlp(features)
        return score.squeeze(-1)  # (batch,)


class NarrativeSimilarityDataset(Dataset):
    """
    Dataset for MLP head training.
    
    Returns raw texts (not tokenized) since we'll use Track B to generate embeddings.
    """
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'anchor': item['anchor'],
            'text_a': item['text_a'],
            'text_b': item['text_b'],
            'label': item['label']
        }


def load_cross_encoder_data(path: str) -> List[Dict]:
    """Load cross-encoder dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def augment_swap(data: List[Dict], enable: bool) -> List[Dict]:
    """Duplicate samples with A/B swapped when enabled."""
    if not enable:
        return data
    augmented = []
    for item in data:
        augmented.append(item)
        augmented.append({
            'anchor': item['anchor'],
            'text_a': item['text_b'],
            'text_b': item['text_a'],
            'label': 1 - item['label']
        })
    return augmented


def collate_fn(batch):
    """Custom collate function to handle text batches."""
    return {
        'anchor': [item['anchor'] for item in batch],
        'text_a': [item['text_a'] for item in batch],
        'text_b': [item['text_b'] for item in batch],
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long)
    }


def train_epoch(
    mlp_head,
    track_b_model,
    dataloader,
    optimizer,
    scheduler,
    device,
    temperature=1.0,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
    use_amp=True,
    clip_value=1.0
):
    """
    Train for one epoch with MLP head over Track B embeddings.
    
    Args:
        mlp_head: MLP head model
        track_b_model: Frozen (or fine-tunable) Track B bi-encoder
        dataloader: Training data
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        temperature: Temperature for MLP head scores
        distill_weight: Weight for distillation from Track B cosine similarities
        distill_temperature: Temperature for distillation
        use_amp: Use mixed precision
        clip_value: Gradient clipping value
    """
    mlp_head.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    loss_fn = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean') if distill_weight > 0 else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        anchor_texts = batch['anchor']
        text_a_list = batch['text_a']
        text_b_list = batch['text_b']
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device.startswith('cuda'):
            with torch.cuda.amp.autocast():
                # Generate embeddings with Track B (frozen or fine-tunable)
                with torch.no_grad() if track_b_model.training == False else torch.enable_grad():
                    anchor_emb = track_b_model.encode(
                        anchor_texts, convert_to_tensor=True, device=device, show_progress_bar=False
                    )
                    a_emb = track_b_model.encode(
                        text_a_list, convert_to_tensor=True, device=device, show_progress_bar=False
                    )
                    b_emb = track_b_model.encode(
                        text_b_list, convert_to_tensor=True, device=device, show_progress_bar=False
                    )
                
                # Get scores from MLP head
                score_a = mlp_head(anchor_emb, a_emb)
                score_b = mlp_head(anchor_emb, b_emb)
                
                # Stack and apply temperature
                scores = torch.stack([score_a, score_b], dim=1) / temperature
                targets = (1 - labels).long()
                
                # Cross-entropy loss
                loss = loss_fn(scores, targets)
                
                # Distillation from Track B cosine similarities
                if distill_weight > 0 and kl_loss:
                    with torch.no_grad():
                        sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)
                        sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)
                        teacher_scores = torch.stack([sim_a, sim_b], dim=1) / distill_temperature
                        teacher_probs = F.softmax(teacher_scores, dim=1)
                    
                    student_log_probs = F.log_softmax(scores / distill_temperature, dim=1)
                    loss = loss + distill_weight * kl_loss(student_log_probs, teacher_probs)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mlp_head.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Generate embeddings with Track B
            with torch.no_grad() if track_b_model.training == False else torch.enable_grad():
                anchor_emb = track_b_model.encode(
                    anchor_texts, convert_to_tensor=True, device=device, show_progress_bar=False
                )
                a_emb = track_b_model.encode(
                    text_a_list, convert_to_tensor=True, device=device, show_progress_bar=False
                )
                b_emb = track_b_model.encode(
                    text_b_list, convert_to_tensor=True, device=device, show_progress_bar=False
                )
            
            # Get scores from MLP head
            score_a = mlp_head(anchor_emb, a_emb)
            score_b = mlp_head(anchor_emb, b_emb)
            
            # Stack and apply temperature
            scores = torch.stack([score_a, score_b], dim=1) / temperature
            targets = (1 - labels).long()
            
            # Cross-entropy loss
            loss = loss_fn(scores, targets)
            
            # Distillation from Track B cosine similarities
            if distill_weight > 0 and kl_loss:
                with torch.no_grad():
                    sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)
                    sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)
                    teacher_scores = torch.stack([sim_a, sim_b], dim=1) / distill_temperature
                    teacher_probs = F.softmax(teacher_scores, dim=1)
                
                student_log_probs = F.log_softmax(scores / distill_temperature, dim=1)
                loss = loss + distill_weight * kl_loss(student_log_probs, teacher_probs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_head.parameters(), clip_value)
            optimizer.step()
        
        scheduler.step()
        
        # Calculate accuracy
        predictions = torch.argmax(scores, dim=-1)
        correct += (predictions == targets).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(mlp_head, track_b_model, dataloader, device, temperature=1.0):
    """Evaluate the MLP head."""
    mlp_head.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        anchor_texts = batch['anchor']
        text_a_list = batch['text_a']
        text_b_list = batch['text_b']
        labels = batch['label'].to(device)
        
        # Generate embeddings with Track B
        anchor_emb = track_b_model.encode(
            anchor_texts, convert_to_tensor=True, device=device, show_progress_bar=False
        )
        a_emb = track_b_model.encode(
            text_a_list, convert_to_tensor=True, device=device, show_progress_bar=False
        )
        b_emb = track_b_model.encode(
            text_b_list, convert_to_tensor=True, device=device, show_progress_bar=False
        )
        
        # Get scores from MLP head
        score_a = mlp_head(anchor_emb, a_emb)
        score_b = mlp_head(anchor_emb, b_emb)
        
        # Stack and apply temperature
        scores = torch.stack([score_a, score_b], dim=1) / temperature
        targets = (1 - labels).long()
        
        # Loss and predictions
        loss = loss_fn(scores, targets)
        predictions = torch.argmax(scores, dim=-1)
        correct += (predictions == targets).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_single_model(
    train_data: List[Dict],
    val_data: List[Dict],
    config: dict,
    fold: int = None
) -> Tuple[float, str]:
    """
    Train a single MLP head model (either on full data or one fold).
    
    Returns:
        (best_val_accuracy, model_save_path)
    """
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    # Apply swap augmentation if enabled
    train_data = augment_swap(train_data, config.get('augmentation', {}).get('swap_ab', False))
    
    # Load Track B model
    track_b_path = config['track_a']['track_b_model_path']
    if not Path(track_b_path).exists():
        print(f"❌ Track B model not found at: {track_b_path}")
        print("Please train Track B first!")
        sys.exit(1)
    
    print(f"\nLoading Track B model: {track_b_path}")
    track_b_model = SentenceTransformer(track_b_path, device=device)
    
    # Freeze or unfreeze Track B
    freeze_track_b = config['track_a'].get('freeze_track_b', True)
    if freeze_track_b:
        print("✓ Freezing Track B (head-only training)")
        track_b_model.eval()
        for param in track_b_model.parameters():
            param.requires_grad = False
    else:
        print("⚠️  Track B will be fine-tuned (joint training)")
        track_b_model.train()
    
    # Get embedding dimension from Track B
    embedding_dim = track_b_model.get_sentence_embedding_dimension()
    print(f"Track B embedding dimension: {embedding_dim}")
    
    # Create MLP head
    mlp_head = MLPHead(
        embedding_dim=embedding_dim,
        hidden_dim=config['track_a']['mlp_hidden_dim'],
        dropout=config['track_a']['mlp_dropout']
    )
    mlp_head.to(device)
    
    print(f"\nMLP Head architecture:")
    print(f"  Input: {4 * embedding_dim} (pairwise features)")
    print(f"  Hidden: {config['track_a']['mlp_hidden_dim']}")
    print(f"  Output: 1 (score)")
    print(f"  Dropout: {config['track_a']['mlp_dropout']}")
    
    # Create datasets
    train_dataset = NarrativeSimilarityDataset(train_data)
    val_dataset = NarrativeSimilarityDataset(val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['track_a']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['track_a']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Setup optimizer (only for MLP head if Track B is frozen)
    if freeze_track_b:
        optimizer = AdamW(
            mlp_head.parameters(),
            lr=config['track_a']['learning_rate'],
            weight_decay=config['track_a']['weight_decay']
        )
    else:
        # Joint training: different LRs for Track B and MLP head
        optimizer = AdamW([
            {'params': track_b_model.parameters(), 'lr': config['track_a']['learning_rate'] / 10},
            {'params': mlp_head.parameters(), 'lr': config['track_a']['learning_rate']}
        ], weight_decay=config['track_a']['weight_decay'])
    
    total_steps = len(train_loader) * config['track_a']['epochs']
    warmup_steps = int(total_steps * config['track_a']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0.0
    patience = 3
    patience_counter = 0
    
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    model_save_path = config['track_a']['model_save_path'] + fold_suffix
    
    print(f"\nStarting training{' for fold ' + str(fold) if fold is not None else ''}...")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    for epoch in range(config['track_a']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['track_a']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            mlp_head,
            track_b_model,
            train_loader,
            optimizer,
            scheduler,
            device,
            temperature=config['track_a'].get('temperature', 1.0),
            distill_weight=config['track_a'].get('distill_weight', 0.0),
            distill_temperature=config['track_a'].get('distill_temperature', 1.0),
            use_amp=config['track_a']['mixed_precision'],
            clip_value=config['track_a']['gradient_clip']
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            mlp_head,
            track_b_model,
            val_loader,
            device,
            temperature=config['track_a'].get('temperature', 1.0)
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            print(f"New best validation accuracy! Saving model to {model_save_path}")
            Path(model_save_path).mkdir(parents=True, exist_ok=True)
            torch.save(mlp_head.state_dict(), Path(model_save_path) / "mlp_head.pt")
            # Save config for inference
            with open(Path(model_save_path) / "config.json", 'w') as f:
                json.dump({
                    'embedding_dim': embedding_dim,
                    'mlp_hidden_dim': config['track_a']['mlp_hidden_dim'],
                    'mlp_dropout': config['track_a']['mlp_dropout'],
                    'temperature': config['track_a'].get('temperature', 1.0),
                    'track_b_model_path': track_b_path
                }, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return best_val_accuracy, model_save_path


def train_with_kfold(config: dict):
    """Train with k-fold cross-validation."""
    # Load data
    prepared_dir = Path(config['data']['prepared_data_dir'])
    all_data = load_cross_encoder_data(prepared_dir / "cross_encoder_data.jsonl")
    
    # Load k-fold splits
    with open(prepared_dir / "kfold_splits.json", 'r') as f:
        splits = json.load(f)
    
    # Get folds to train (default: all)
    folds_to_train = config['track_a'].get('folds_to_train')
    if folds_to_train is None:
        folds_to_train = list(range(len(splits)))

    print(f"Training with {len(splits)}-fold cross-validation (running folds: {folds_to_train})")
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        if fold_idx not in folds_to_train:
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(splits)}")
        print(f"{'='*60}")
        
        # Split data
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        
        # Train
        val_acc, model_path = train_single_model(
            train_data,
            val_data,
            config,
            fold=fold_idx
        )
        
        fold_results.append({
            'fold': fold_idx,
            'val_accuracy': val_acc,
            'model_path': model_path
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-Fold Cross-Validation Results")
    print(f"{'='*60}")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val Acc = {result['val_accuracy']:.4f}")
    
    if fold_results:
        mean_acc = np.mean([r['val_accuracy'] for r in fold_results])
        std_acc = np.std([r['val_accuracy'] for r in fold_results])
        print(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Save results
    results_path = Path(config['track_a']['model_save_path']).parent / "kfold_results.json"
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return fold_results


def train_full_model(config: dict):
    """Train a single model on all data."""
    # Load data
    prepared_dir = Path(config['data']['prepared_data_dir'])
    all_data = load_cross_encoder_data(prepared_dir / "cross_encoder_data.jsonl")
    
    # Create a simple 90/10 split for validation
    n_val = int(0.1 * len(all_data))
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    
    val_data = [all_data[i] for i in indices[:n_val]]
    train_data = [all_data[i] for i in indices[n_val:]]
    
    print("Training single model on full dataset")
    val_acc, model_path = train_single_model(train_data, val_data, config)
    
    print(f"\nFinal validation accuracy: {val_acc:.4f}")
    print(f"Model saved to: {model_path}")
    
    return val_acc, model_path


def main():
    """Main training pipeline - Google Colab only."""
    if not is_colab():
        print("⚠️  This script is designed for Google Colab only!")
        print("Please run in Google Colab with GPU enabled.")
        sys.exit(1)
    
    print("="*60)
    print("TRACK A TRAINING (V2: MLP HEAD) - GOOGLE COLAB")
    print("="*60)
    
    # Setup Colab environment
    colab_paths = setup_colab_environment()
    print_gpu_info()
    
    # Load and update config
    config = load_config(colab_paths['config_path'])
    config = update_config_for_colab(config, colab_paths)
    
    set_seed(config['seed'])
    
    if config['track_a']['use_kfold']:
        train_with_kfold(config)
    else:
        train_full_model(config)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
