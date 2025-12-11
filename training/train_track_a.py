"""
Training script for Track A: Cross-encoder model.

This script:
- Loads a transformer model (DeBERTa, RoBERTa, etc.)
- Trains it as a cross-encoder to predict which of two stories is closer to an anchor
- Supports k-fold cross-validation
- Uses mixed precision training
- Implements early stopping

Works in both local and Google Colab environments.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
import yaml
import pandas as pd

# Import Colab utilities
try:
    from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, install_colab_dependencies, print_gpu_info
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, install_colab_dependencies, print_gpu_info
    except ImportError:
        def is_colab(): return False
        def setup_colab_environment(x): return None
        def update_config_for_colab(c, p): return c
        def install_colab_dependencies(): pass
        def print_gpu_info(): pass


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


class NarrativeSimilarityDataset(Dataset):
    """
    Dataset for pairwise scoring cross-encoder training.
    
    NEW APPROACH: Score (anchor, A) and (anchor, B) separately, then apply softmax.
    This is more stable and aligns with IR reranking best practices.
    
    Each sample returns both pairs for a triple.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Pairwise scoring: encode (anchor, text_a) and (anchor, text_b) separately
        # Pair A: anchor + text_a
        text_a_pair = f"{item['anchor']} {self.tokenizer.sep_token} {item['text_a']}"
        encoding_a = self.tokenizer(
            text_a_pair,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Pair B: anchor + text_b
        text_b_pair = f"{item['anchor']} {self.tokenizer.sep_token} {item['text_b']}"
        encoding_b = self.tokenizer(
            text_b_pair,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': encoding_a['input_ids'].squeeze(0),
            'attention_mask_a': encoding_a['attention_mask'].squeeze(0),
            'input_ids_b': encoding_b['input_ids'].squeeze(0),
            'attention_mask_b': encoding_b['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def load_cross_encoder_data(path: str) -> List[Dict]:
    """Load cross-encoder dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def train_epoch(model, dataloader, optimizer, scheduler, device, use_amp=True, clip_value=1.0, label_smoothing=0.0):
    """
    Train for one epoch with pairwise scoring.
    
    NEW APPROACH: Score (anchor, A) and (anchor, B) separately,
    then apply softmax cross-entropy loss.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Get pairs
        input_ids_a = batch['input_ids_a'].to(device)
        attention_mask_a = batch['attention_mask_a'].to(device)
        input_ids_b = batch['input_ids_b'].to(device)
        attention_mask_b = batch['attention_mask_b'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device == 'cuda':
            with torch.cuda.amp.autocast():
                # Score pair A: (anchor, text_a)
                outputs_a = model(
                    input_ids=input_ids_a,
                    attention_mask=attention_mask_a
                )
                score_a = outputs_a.logits[:, 1]  # Take positive class logit as score
                
                # Score pair B: (anchor, text_b)
                outputs_b = model(
                    input_ids=input_ids_b,
                    attention_mask=attention_mask_b
                )
                score_b = outputs_b.logits[:, 1]  # Take positive class logit as score
                
                # Stack scores and apply softmax cross-entropy
                scores = torch.stack([score_a, score_b], dim=1)  # Shape: (batch, 2)
                # Label: 1 if A is closer (index 0), 0 if B is closer (index 1)
                # Need to invert: if label=1, target should be index 0 (score_a)
                targets = (1 - labels).long()  # Invert: 1→0, 0→1
                
                loss = loss_fn(scores, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Score pair A
            outputs_a = model(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a
            )
            score_a = outputs_a.logits[:, 1]
            
            # Score pair B
            outputs_b = model(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b
            )
            score_b = outputs_b.logits[:, 1]
            
            # Compute loss
            scores = torch.stack([score_a, score_b], dim=1)
            targets = (1 - labels).long()
            loss = loss_fn(scores, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        
        scheduler.step()
        
        # Calculate accuracy: argmax over [score_a, score_b]
        predictions = torch.argmax(scores, dim=-1)
        correct += (predictions == targets).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, label_smoothing=0.0):
    """Evaluate the model with pairwise scoring."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Get pairs
        input_ids_a = batch['input_ids_a'].to(device)
        attention_mask_a = batch['attention_mask_a'].to(device)
        input_ids_b = batch['input_ids_b'].to(device)
        attention_mask_b = batch['attention_mask_b'].to(device)
        labels = batch['labels'].to(device)
        
        # Score pair A
        outputs_a = model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a
        )
        score_a = outputs_a.logits[:, 1]
        
        # Score pair B
        outputs_b = model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b
        )
        score_b = outputs_b.logits[:, 1]
        
        # Compute loss and predictions
        scores = torch.stack([score_a, score_b], dim=1)
        targets = (1 - labels).long()
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
    Train a single model (either on full data or one fold).
    
    Returns:
        (best_val_accuracy, model_save_path)
    """
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer and model
    print(f"\nLoading model: {config['track_a']['base_model']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config['track_a']['base_model'],
            use_fast=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Error loading fast tokenizer: {e}")
        print("Trying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(
            config['track_a']['base_model'],
            use_fast=False,
            trust_remote_code=True
        )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['track_a']['base_model'],
        num_labels=config['track_a']['num_labels'],
        trust_remote_code=True
    )
    model.to(device)
    
    # Create datasets
    train_dataset = NarrativeSimilarityDataset(
        train_data,
        tokenizer,
        max_length=config['track_a']['max_length']
    )
    val_dataset = NarrativeSimilarityDataset(
        val_data,
        tokenizer,
        max_length=config['track_a']['max_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['track_a']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['track_a']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['track_a']['learning_rate'],
        weight_decay=config['track_a']['weight_decay']
    )
    
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
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_amp=config['track_a']['mixed_precision'],
            clip_value=config['track_a']['gradient_clip'],
            label_smoothing=config['track_a'].get('label_smoothing', 0.0)
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            model, 
            val_loader, 
            device,
            label_smoothing=config['track_a'].get('label_smoothing', 0.0)
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            print(f"New best validation accuracy! Saving model to {model_save_path}")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
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
    
    print(f"Training with {len(splits)}-fold cross-validation")
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
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
    """Main training pipeline."""
    # Check if we're using Drive-based setup or cloned repo
    import os
    using_drive_setup = os.path.exists('/content/drive/MyDrive/narrative_similarity/config.yaml')
    
    # Setup Colab environment if running in Drive-based mode
    if is_colab() and using_drive_setup:
        print("="*60)
        print("RUNNING IN GOOGLE COLAB - DRIVE MODE - TRACK A")
        print("="*60)
        install_colab_dependencies()
        colab_paths = setup_colab_environment()
        print_gpu_info()
        
        # Load and update config
        config = load_config(colab_paths['config_path'])
        config = update_config_for_colab(config, colab_paths)
    else:
        # Standard mode (local or cloned repo in Colab)
        if is_colab():
            print("="*60)
            print("RUNNING IN GOOGLE COLAB - REPO MODE - TRACK A")
            print("="*60)
            print_gpu_info()
        else:
            print("Running locally")
        
        # Just load config normally
        config = load_config()
    
    set_seed(config['seed'])
    
    if config['track_a']['use_kfold']:
        train_with_kfold(config)
    else:
        train_full_model(config)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

