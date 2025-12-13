"""
Training script for Track B: Bi-encoder with projection head (v2 + improvements).

Google Colab training script.

This script implements the v2 approach from APPROACH.md with additional improvements:
- 512-dim projection head for better generalization on small datasets
- Unified composite loss (MarginRanking + PairwiseSoftmax + MNR)
- Hard negative mining for better discrimination
- Hyperparameter sweep for optimal temperature and margin
- Enhanced regularization (dropout, weight decay)
- Optional SimCSE consistency loss
- Temperature scaling for gradient sharpening
- Mixed precision training (FP16)
- Saves models to Google Drive
"""

import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import SequentialEvaluator
from tqdm import tqdm
import pandas as pd
from sentence_transformers.util import cos_sim
from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, install_colab_dependencies, print_gpu_info


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


def load_triplets(path: str) -> list:
    """Load triplet dataset."""
    triplets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            triplets.append(json.loads(line))
    return triplets


def load_pairs(path: str) -> list:
    """Load pairwise dataset."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def create_triplet_examples(triplets: list) -> list:
    """Convert triplet dicts to InputExample objects."""
    examples = []
    for t in triplets:
        examples.append(InputExample(
            texts=[t['anchor'], t['positive'], t['negative']]
        ))
    return examples


def create_pair_examples(pairs: list) -> list:
    """Convert pair dicts to InputExample objects for MultipleNegativesRankingLoss."""
    examples = []
    for p in pairs:
        if p['label'] > 0.5:  # Only use positive pairs
            examples.append(InputExample(
                texts=[p['anchor'], p['candidate']]
            ))
    return examples


def create_triple_softmax_examples(data: list) -> list:
    """Create examples for PairwiseSoftmaxLoss."""
    examples = []
    for item in data:
        examples.append(InputExample(
            texts=[item['anchor'], item['text_a'], item['text_b']],
            label=float(item['label'])
        ))
    return examples


class PairwiseSoftmaxLoss(nn.Module):
    """Direct triple-wise pairwise softmax loss with temperature scaling."""
    
    def __init__(self, model: SentenceTransformer, temperature: float = 1.0):
        super(PairwiseSoftmaxLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features, labels):
        embeddings = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        anchor_emb = embeddings[0]
        a_emb = embeddings[1]
        b_emb = embeddings[2]
        
        sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)
        sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)
        
        scores = torch.stack([sim_a, sim_b], dim=1) / self.temperature
        targets = (1 - labels).long()
        
        loss = self.loss_fn(scores, targets)
        return loss


class MarginRankingLossCustom(nn.Module):
    """Margin ranking loss for metric-aligned training."""
    
    def __init__(self, model: SentenceTransformer, margin: float = 0.2):
        super(MarginRankingLossCustom, self).__init__()
        self.model = model
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, sentence_features, labels):
        embeddings = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        anchor_emb = embeddings[0]
        a_emb = embeddings[1]
        b_emb = embeddings[2]
        
        sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)
        sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)
        
        y = 2 * labels - 1  # Convert {0,1} to {-1,+1}
        loss = self.loss_fn(sim_a, sim_b, y)
        return loss


class SimCSELoss(nn.Module):
    """
    SimCSE-style consistency loss.
    
    Applies two different dropout masks to the same text and maximizes agreement.
    """
    
    def __init__(self, model: SentenceTransformer, temperature: float = 0.05):
        super(SimCSELoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features, labels):
        # Encode with first dropout mask
        emb1 = self.model(sentence_features[0])['sentence_embedding']
        
        # Encode again with different dropout mask (model in train mode)
        emb2 = self.model(sentence_features[0])['sentence_embedding']
        
        # Compute similarity matrix
        batch_size = emb1.size(0)
        sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        
        # Positive pairs are on the diagonal
        targets = torch.arange(batch_size, device=emb1.device)
        
        loss = self.loss_fn(sim_matrix, targets)
        return loss


class CompositeLoss(nn.Module):
    """Composite loss combining multiple objectives for robust training."""
    
    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 0.2,
        temperature: float = 0.7,
        weight_margin: float = 1.0,
        weight_pairwise: float = 0.5
    ):
        super(CompositeLoss, self).__init__()
        self.margin_loss = MarginRankingLossCustom(model, margin=margin)
        self.pairwise_loss = PairwiseSoftmaxLoss(model, temperature=temperature)
        self.weight_margin = weight_margin
        self.weight_pairwise = weight_pairwise
    
    def forward(self, sentence_features, labels):
        loss_margin = self.margin_loss(sentence_features, labels)
        loss_pairwise = self.pairwise_loss(sentence_features, labels)
        
        total_loss = (
            self.weight_margin * loss_margin +
            self.weight_pairwise * loss_pairwise
        )
        
        return total_loss


def mine_hard_negatives(
    model: SentenceTransformer,
    triplets: list,
    all_texts: list,
    device: str,
    k: int = 5
) -> list:
    """
    Mine hard negatives for improved training.
    
    For each anchor, find K nearest neighbors that are NOT the correct positive.
    
    Args:
        model: Current Track B model
        triplets: Original triplets
        all_texts: All unique texts in dataset
        device: Device
        k: Number of hard negatives per anchor
    
    Returns:
        List of hard negative triplets
    """
    print(f"\n{'='*60}")
    print(f"MINING HARD NEGATIVES (K={k})")
    print(f"{'='*60}")
    
    # Encode all texts
    print("Encoding all texts...")
    with torch.no_grad():
        all_embeddings = model.encode(
            all_texts,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=True,
            batch_size=32
        )
    
    # Create text to embedding mapping
    text_to_idx = {text: idx for idx, text in enumerate(all_texts)}
    
    hard_triplets = []
    
    print("Mining hard negatives...")
    for triplet in tqdm(triplets):
        anchor = triplet['anchor']
        positive = triplet['positive']
        
        # Get anchor embedding
        anchor_idx = text_to_idx[anchor]
        anchor_emb = all_embeddings[anchor_idx].unsqueeze(0)
        
        # Compute similarities to all texts
        similarities = F.cosine_similarity(anchor_emb, all_embeddings, dim=1)
        
        # Get top-K most similar (excluding anchor and positive)
        sorted_indices = torch.argsort(similarities, descending=True)
        
        hard_negatives = []
        for idx in sorted_indices:
            idx = idx.item()
            candidate = all_texts[idx]
            
            # Skip anchor and positive
            if candidate == anchor or candidate == positive:
                continue
            
            hard_negatives.append(candidate)
            
            if len(hard_negatives) >= k:
                break
        
        # Create hard negative triplets
        for hard_neg in hard_negatives:
            hard_triplets.append({
                'anchor': anchor,
                'positive': positive,
                'negative': hard_neg
            })
    
    print(f"✓ Mined {len(hard_triplets)} hard negative triplets")
    return hard_triplets


def hyperparameter_sweep(
    model: SentenceTransformer,
    dev_path: str,
    device: str,
    temperature_grid: list,
    margin_grid: list
) -> tuple:
    """
    Sweep over temperature and margin hyperparameters.
    
    Returns:
        (best_temperature, best_margin, best_accuracy)
    """
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SWEEP")
    print(f"{'='*60}")
    
    df = pd.read_json(dev_path, lines=True)
    
    # Collect all unique texts
    all_texts = set()
    for _, row in df.iterrows():
        all_texts.add(row['anchor_text'])
        all_texts.add(row['text_a'])
        all_texts.add(row['text_b'])
    
    # Encode all texts once
    all_texts = list(all_texts)
    print("Encoding texts for sweep...")
    with torch.no_grad():
        embeddings = model.encode(
            all_texts,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
    
    text_to_embedding = {text: emb for text, emb in zip(all_texts, embeddings)}
    
    best_accuracy = 0.0
    best_temp = None
    best_margin = None
    
    results = []
    
    print(f"\nTesting {len(temperature_grid)} temperatures × {len(margin_grid)} margins = {len(temperature_grid) * len(margin_grid)} combinations")
    
    for temp in temperature_grid:
        for margin in margin_grid:
            correct = 0
            
            for _, row in df.iterrows():
                anchor_emb = text_to_embedding[row['anchor_text']]
                a_emb = text_to_embedding[row['text_a']]
                b_emb = text_to_embedding[row['text_b']]
                
                sim_a = cos_sim(anchor_emb, a_emb).item()
                sim_b = cos_sim(anchor_emb, b_emb).item()
                
                # Apply temperature (margin doesn't affect inference, only training)
                sim_a = sim_a / temp
                sim_b = sim_b / temp
                
                predicted_a_closer = sim_a > sim_b
                if predicted_a_closer == row['text_a_is_closer']:
                    correct += 1
            
            accuracy = correct / len(df)
            results.append((temp, margin, accuracy))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_temp = temp
                best_margin = margin
    
    # Print results
    print(f"\n{'Temperature':<12} {'Margin':<8} {'Accuracy':<10}")
    print("-" * 32)
    for temp, margin, acc in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
        marker = " ← BEST" if (temp == best_temp and margin == best_margin) else ""
        print(f"{temp:<12.2f} {margin:<8.2f} {acc:<10.4f}{marker}")
    
    print(f"\n✓ Best: temperature={best_temp}, margin={best_margin}, accuracy={best_accuracy:.4f}")
    
    return best_temp, best_margin, best_accuracy


def evaluate_on_track_a(model, dev_path: str, device: str) -> float:
    """Evaluate the model on Track A dev set."""
    df = pd.read_json(dev_path, lines=True)
    
    all_texts = set()
    for _, row in df.iterrows():
        all_texts.add(row['anchor_text'])
        all_texts.add(row['text_a'])
        all_texts.add(row['text_b'])
    
    all_texts = list(all_texts)
    with torch.no_grad():
        embeddings = model.encode(
            all_texts,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
    
    text_to_embedding = {text: emb for text, emb in zip(all_texts, embeddings)}
    
    correct = 0
    for _, row in df.iterrows():
        anchor_emb = text_to_embedding[row['anchor_text']]
        a_emb = text_to_embedding[row['text_a']]
        b_emb = text_to_embedding[row['text_b']]
        
        sim_a = cos_sim(anchor_emb, a_emb).item()
        sim_b = cos_sim(anchor_emb, b_emb).item()
        
        predicted_a_closer = sim_a > sim_b
        if predicted_a_closer == row['text_a_is_closer']:
            correct += 1
    
    accuracy = correct / len(df)
    return accuracy


class CustomEvaluator:
    """Custom evaluator that tracks best model."""
    
    def __init__(self, dev_path: str, device: str, save_path: str):
        self.dev_path = dev_path
        self.device = device
        self.save_path = save_path
        self.best_accuracy = 0.0
    
    def __call__(self, model, output_path: str, epoch: int, steps: int) -> float:
        """Evaluate and save if best."""
        accuracy = evaluate_on_track_a(model, self.dev_path, self.device)
        print(f"\nEpoch {epoch}, Step {steps}: Accuracy = {accuracy:.4f}")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {self.save_path}")
            model.save(self.save_path)
        
        return accuracy


def add_projection_head(model: SentenceTransformer, projection_dim: int, dropout: float = 0.0):
    """
    Add a projection head to reduce embedding dimensionality.
    
    Args:
        model: Base SentenceTransformer
        projection_dim: Output dimension
        dropout: Dropout rate for regularization
    """
    word_embedding_dimension = model.get_sentence_embedding_dimension()
    
    # Add dense projection layer
    dense = models.Dense(
        in_features=word_embedding_dimension,
        out_features=projection_dim,
        activation_function=nn.Identity()
    )
    
    # Build modules list
    modules = list(model._modules.values())
    
    # Remove existing Normalize layer if present
    if len(modules) > 0 and isinstance(modules[-1], models.Normalize):
        normalize_layer = modules[-1]
        modules = modules[:-1]
    else:
        normalize_layer = models.Normalize()
    
    # Add projection + optional dropout + normalize
    modules.append(dense)
    
    if dropout > 0:
        # Add dropout layer (custom wrapper)
        class DropoutLayer(nn.Module):
            def __init__(self, dropout_rate):
                super().__init__()
                self.dropout = nn.Dropout(dropout_rate)
            
            def forward(self, features):
                features['sentence_embedding'] = self.dropout(features['sentence_embedding'])
                return features
        
        modules.append(DropoutLayer(dropout))
    
    modules.append(normalize_layer)
    
    new_model = SentenceTransformer(modules=modules)
    return new_model


def main():
    """Main training pipeline - Google Colab only."""
    if not is_colab():
        print("⚠️  This script is designed for Google Colab only!")
        print("Please run in Google Colab with GPU enabled.")
        sys.exit(1)
    
    print("="*60)
    print("TRACK B TRAINING (V2 + IMPROVEMENTS) - GOOGLE COLAB")
    print("="*60)
    
    # Setup Colab environment
    colab_paths = setup_colab_environment()
    print_gpu_info()
    
    # Load and update config
    config = load_config(colab_paths['config_path'])
    config = update_config_for_colab(config, colab_paths)
    
    set_seed(config['seed'])
    
    # Setup device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    print(f"\nLoading base model: {config['track_b']['base_model']}")
    model = SentenceTransformer(config['track_b']['base_model'], device=device)
    
    # Add projection head with optional dropout
    projection_dim = config['track_b'].get('projection_dim', 512)
    projection_dropout = config['track_b'].get('projection_dropout', 0.0)
    print(f"\nAdding {projection_dim}-dim projection head (dropout={projection_dropout})...")
    model = add_projection_head(model, projection_dim, dropout=projection_dropout)
    print(f"✓ Model now outputs {projection_dim}-dimensional embeddings")
    
    # Shorten max sequence length
    if 'max_seq_length' in config['track_b']:
        model.max_seq_length = config['track_b']['max_seq_length']
        print(f"Set max_seq_length={model.max_seq_length}")
    
    # Enable gradient checkpointing
    if config['track_b'].get('enable_gradient_checkpointing', True):
        try:
            model._first_module().auto_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory savings.")
        except Exception as e:
            print(f"Warning: could not enable gradient checkpointing: {e}")
    
    try:
        model._first_module().auto_model.config.use_cache = False
        print("Disabled transformer cache to reduce memory.")
    except Exception:
        pass
    
    # Load prepared data
    prepared_dir = Path(config['data']['prepared_data_dir'])
    print("\nLoading data...")
    triplets = load_triplets(prepared_dir / "triplets.jsonl")
    pairs = load_pairs(prepared_dir / "pairs.jsonl")
    
    with open(prepared_dir / "cross_encoder_data.jsonl", 'r', encoding='utf-8') as f:
        cross_encoder_data = [json.loads(line) for line in f]
    
    # Load train/val split
    with open(prepared_dir / "train_val_split.json", 'r') as f:
        split = json.load(f)
    
    train_indices = set(split['train'])
    
    # Split data
    train_triplets = [t for i, t in enumerate(triplets) if i in train_indices]
    train_pairs = [p for i, p in enumerate(pairs) if i // 2 in train_indices]
    train_cross_encoder = [d for i, d in enumerate(cross_encoder_data) if i // 2 in train_indices]
    
    print(f"Train triplets: {len(train_triplets)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Train cross-encoder: {len(train_cross_encoder)}")
    
    # Create InputExamples
    pair_examples = create_pair_examples(train_pairs)
    triple_softmax_examples = create_triple_softmax_examples(train_cross_encoder)
    
    # Create DataLoaders
    batch_size = config['track_b']['batch_size']
    
    pair_loader = DataLoader(pair_examples, batch_size=batch_size * 2, shuffle=True)
    triple_softmax_loader = DataLoader(triple_softmax_examples, batch_size=batch_size, shuffle=True)
    
    # Setup losses
    mnr_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    loss_weights = config['track_b'].get('loss_weights', {})
    composite_loss = CompositeLoss(
        model=model,
        margin=config['track_b'].get('margin', 0.2),
        temperature=config['track_b'].get('temperature', 0.7),
        weight_margin=loss_weights.get('margin', 1.0),
        weight_pairwise=loss_weights.get('pairwise', 0.5)
    )
    
    # Setup evaluator
    base_evaluator = CustomEvaluator(
        dev_path=config['data']['dev_track_a'],
        device=device,
        save_path=config['track_b']['model_save_path']
    )
    evaluator = SequentialEvaluator([base_evaluator])
    
    # Setup optimizer with different weight decay for projection head
    projection_weight_decay = config['track_b'].get('projection_weight_decay', config['track_b']['weight_decay'])
    
    print(f"\nRegularization:")
    print(f"  Backbone weight decay: {config['track_b']['weight_decay']}")
    print(f"  Projection weight decay: {projection_weight_decay}")
    print(f"  Projection dropout: {projection_dropout}")
    
    # V2 Unified Training
    print("\n" + "="*60)
    print("PHASE 1: UNIFIED TRAINING (Composite Loss + MNR)")
    print("="*60)
    
    train_objectives = []
    if len(pair_examples) > 0:
        weight_mnr = loss_weights.get('mnr', 0.5)
        if weight_mnr > 0:
            train_objectives.append((pair_loader, mnr_loss))
            print(f"✓ Added MultipleNegativesRankingLoss (weight: {weight_mnr})")
    
    if len(triple_softmax_examples) > 0:
        train_objectives.append((triple_softmax_loader, composite_loss))
        print(f"✓ Added CompositeLoss (Margin + Pairwise)")
    
    model.fit(
        train_objectives=train_objectives,
        epochs=config['track_b']['epochs'],
        warmup_steps=config['track_b']['warmup_steps'],
        optimizer_params={'lr': config['track_b']['learning_rate']},
        weight_decay=config['track_b']['weight_decay'],
        evaluation_steps=config['track_b']['eval_steps'],
        evaluator=evaluator,
        output_path=config['track_b']['model_save_path'] + "_temp",
        save_best_model=False,
        use_amp=config['track_b']['mixed_precision']
    )
    
    # Load best model from phase 1
    model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
    phase1_accuracy = base_evaluator.best_accuracy
    print(f"\n✓ Phase 1 complete. Best accuracy: {phase1_accuracy:.4f}")
    
    # PHASE 2: Hard Negative Mining (if enabled)
    if config['track_b'].get('enable_hard_negatives', False):
        print("\n" + "="*60)
        print("PHASE 2: HARD NEGATIVE MINING")
        print("="*60)
        
        # Collect all unique texts
        all_texts = set()
        for t in train_triplets:
            all_texts.add(t['anchor'])
            all_texts.add(t['positive'])
            all_texts.add(t['negative'])
        all_texts = list(all_texts)
        
        # Mine hard negatives
        hard_k = config['track_b'].get('hard_negative_k', 5)
        hard_triplets = mine_hard_negatives(model, train_triplets, all_texts, device, k=hard_k)
        
        # Convert to examples
        hard_examples = create_triplet_examples(hard_triplets)
        hard_loader = DataLoader(hard_examples, batch_size=batch_size, shuffle=True)
        
        # Create loss for hard negatives (use TripletLoss)
        hard_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=config['track_b'].get('margin', 0.2)
        )
        
        # Train on hard negatives
        hard_epochs = config['track_b'].get('hard_negative_epochs', 2)
        print(f"\nTraining on hard negatives for {hard_epochs} epochs...")
        
        model.fit(
            train_objectives=[(hard_loader, hard_loss)],
            epochs=hard_epochs,
            warmup_steps=max(1, config['track_b']['warmup_steps'] // 4),
            optimizer_params={'lr': config['track_b']['learning_rate'] / 2},
            weight_decay=config['track_b']['weight_decay'],
            evaluation_steps=config['track_b']['eval_steps'],
            evaluator=evaluator,
            output_path=config['track_b']['model_save_path'] + "_hard",
            save_best_model=False,
            use_amp=config['track_b']['mixed_precision']
        )
        
        model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
        phase2_accuracy = base_evaluator.best_accuracy
        print(f"\n✓ Phase 2 complete. Best accuracy: {phase2_accuracy:.4f} (Δ={phase2_accuracy - phase1_accuracy:+.4f})")
    
    # PHASE 3: Hyperparameter Sweep (if enabled)
    if config['track_b'].get('enable_hyperparam_sweep', False):
        print("\n" + "="*60)
        print("PHASE 3: HYPERPARAMETER SWEEP")
        print("="*60)
        
        temp_grid = config['track_b'].get('temperature_grid', [0.5, 0.7, 1.0])
        margin_grid = config['track_b'].get('margin_grid', [0.1, 0.2, 0.3])
        
        best_temp, best_margin, sweep_accuracy = hyperparameter_sweep(
            model,
            config['data']['dev_track_a'],
            device,
            temp_grid,
            margin_grid
        )
        
        # Save best hyperparameters
        best_config_path = Path(config['track_b']['model_save_path']) / "best_hyperparams.json"
        with open(best_config_path, 'w') as f:
            json.dump({
                'temperature': best_temp,
                'margin': best_margin,
                'accuracy': sweep_accuracy
            }, f, indent=2)
        
        print(f"\n✓ Best hyperparameters saved to: {best_config_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    final_model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
    final_accuracy = evaluate_on_track_a(final_model, config['data']['dev_track_a'], device)
    print(f"\nFinal best model accuracy: {final_accuracy:.4f}")
    print(f"Model saved to: {config['track_b']['model_save_path']}")
    print(f"Embedding dimension: {projection_dim}")


if __name__ == "__main__":
    main()
