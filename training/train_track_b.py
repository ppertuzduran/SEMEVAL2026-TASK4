"""
Training script for Track B: Bi-encoder with projection head (v2).

Google Colab training script.

This script implements the v2 approach from APPROACH.md:
- 512-dim projection head for better generalization on small datasets
- Unified composite loss (MarginRanking + PairwiseSoftmax + MNR)
- Removes obsolete A→B distillation logic
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
        # For MultipleNegativesRankingLoss, we use pairs of (anchor, positive)
        # The loss automatically treats other positives in the batch as negatives
        if p['label'] > 0.5:  # Only use positive pairs
            examples.append(InputExample(
                texts=[p['anchor'], p['candidate']]
            ))
    return examples


def create_triple_softmax_examples(data: list) -> list:
    """
    Create examples for PairwiseSoftmaxLoss.
    Each example contains (anchor, text_a, text_b, label).
    """
    examples = []
    for item in data:
        examples.append(InputExample(
            texts=[item['anchor'], item['text_a'], item['text_b']],
            label=float(item['label'])
        ))
    return examples


class PairwiseSoftmaxLoss(nn.Module):
    """
    Direct triple-wise pairwise softmax loss with temperature scaling.
    
    Directly optimizes the evaluation metric by:
    1. Encoding anchor, text_a, text_b independently
    2. Computing cosine similarities
    3. Applying temperature scaling
    4. Applying softmax cross-entropy
    
    This aligns training with the triple-wise evaluation.
    Temperature < 1.0 sharpens the distribution and improves gradient flow.
    """
    
    def __init__(self, model: SentenceTransformer, temperature: float = 1.0):
        super(PairwiseSoftmaxLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features, labels):
        """
        sentence_features: List of dicts with 'input_ids', 'attention_mask', etc.
                          [anchor_features, text_a_features, text_b_features]
        labels: Tensor of shape (batch_size,) with 1 if A is closer, 0 if B is closer
        """
        # Encode all three texts independently
        embeddings = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        anchor_emb = embeddings[0]  # (batch, embed_dim)
        a_emb = embeddings[1]
        b_emb = embeddings[2]
        
        # Compute cosine similarities
        sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)  # (batch,)
        sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)  # (batch,)
        
        # Stack similarities and apply temperature scaling
        scores = torch.stack([sim_a, sim_b], dim=1)  # (batch, 2)
        scores = scores / self.temperature  # Temperature scaling: < 1.0 sharpens gradients
        
        # Convert labels: 1 → index 0 (A closer), 0 → index 1 (B closer)
        targets = (1 - labels).long()
        
        # Cross-entropy loss
        loss = self.loss_fn(scores, targets)
        return loss


class MarginRankingLossCustom(nn.Module):
    """
    Margin ranking loss for metric-aligned training.
    
    For each triple (anchor, A, B) with label indicating which is closer:
    - Computes cosine similarities
    - Applies margin ranking loss to enforce correct ordering
    """
    
    def __init__(self, model: SentenceTransformer, margin: float = 0.2):
        super(MarginRankingLossCustom, self).__init__()
        self.model = model
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, sentence_features, labels):
        """
        sentence_features: [anchor_features, text_a_features, text_b_features]
        labels: Tensor with 1 if A is closer, 0 if B is closer
        """
        # Encode all three texts
        embeddings = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        anchor_emb = embeddings[0]
        a_emb = embeddings[1]
        b_emb = embeddings[2]
        
        # Compute cosine similarities
        sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)
        sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)
        
        # Convert labels: 1 if A closer (y=+1), 0 if B closer (y=-1)
        y = 2 * labels - 1  # Convert {0,1} to {-1,+1}
        
        # MarginRankingLoss expects: loss(x1, x2, y) where y=+1 means x1 > x2
        loss = self.loss_fn(sim_a, sim_b, y)
        return loss


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple objectives for robust training.
    
    Combines:
    - MarginRankingLoss: metric-aligned ranking
    - PairwiseSoftmaxLoss: direct metric optimization
    - MultipleNegativesRankingLoss: contrastive learning (applied separately)
    """
    
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
        """Compute weighted combination of losses."""
        loss_margin = self.margin_loss(sentence_features, labels)
        loss_pairwise = self.pairwise_loss(sentence_features, labels)
        
        total_loss = (
            self.weight_margin * loss_margin +
            self.weight_pairwise * loss_pairwise
        )
        
        return total_loss


def evaluate_on_track_a(model, dev_path: str, device: str) -> float:
    """
    Evaluate the model on Track A dev set.
    Returns accuracy based on cosine similarity comparison.
    """
    df = pd.read_json(dev_path, lines=True)
    
    # Collect all unique texts
    all_texts = set()
    for _, row in df.iterrows():
        all_texts.add(row['anchor_text'])
        all_texts.add(row['text_a'])
        all_texts.add(row['text_b'])
    
    # Encode all texts
    all_texts = list(all_texts)
    with torch.no_grad():
        embeddings = model.encode(
            all_texts, 
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
    
    # Create lookup
    text_to_embedding = {text: emb for text, emb in zip(all_texts, embeddings)}
    
    # Calculate predictions
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


def add_projection_head(model: SentenceTransformer, projection_dim: int):
    """
    Add a projection head to reduce embedding dimensionality.
    
    This adds: Dense(1024 → projection_dim) + LayerNorm + L2 normalization
    """
    # Get the current pooling dimension
    word_embedding_dimension = model.get_sentence_embedding_dimension()
    
    # Add dense projection layer
    dense = models.Dense(
        in_features=word_embedding_dimension,
        out_features=projection_dim,
        activation_function=nn.Identity()  # No activation, just linear projection
    )
    
    # Add layer normalization for stability
    # Note: SentenceTransformer already has Normalize layer at the end for L2 norm
    
    # Add to model (insert before the final Normalize layer)
    # Remove the last Normalize layer, add Dense, then add Normalize back
    modules = list(model._modules.values())
    
    # Find and remove Normalize layer if it exists
    if len(modules) > 0 and isinstance(modules[-1], models.Normalize):
        normalize_layer = modules[-1]
        modules = modules[:-1]
    else:
        normalize_layer = models.Normalize()
    
    # Rebuild model with projection
    modules.append(dense)
    modules.append(normalize_layer)
    
    # Create new model with updated modules
    new_model = SentenceTransformer(modules=modules)
    
    return new_model


def main():
    """Main training pipeline - Google Colab only."""
    if not is_colab():
        print("⚠️  This script is designed for Google Colab only!")
        print("Please run in Google Colab with GPU enabled.")
        sys.exit(1)
    
    print("="*60)
    print("TRACK B TRAINING (V2) - GOOGLE COLAB")
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
    
    # Add projection head (v2 feature)
    projection_dim = config['track_b'].get('projection_dim', 512)
    print(f"\nAdding {projection_dim}-dim projection head...")
    model = add_projection_head(model, projection_dim)
    print(f"✓ Model now outputs {projection_dim}-dimensional embeddings")
    
    # Shorten max sequence length to save VRAM
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
    print("\nLoading triplets...")
    triplets = load_triplets(prepared_dir / "triplets.jsonl")
    print(f"Loaded {len(triplets)} triplets")
    
    print("Loading pairs...")
    pairs = load_pairs(prepared_dir / "pairs.jsonl")
    print(f"Loaded {len(pairs)} pairs")
    
    # Load cross-encoder data for pairwise softmax loss
    print("Loading cross-encoder data for composite loss...")
    cross_encoder_data = []
    with open(prepared_dir / "cross_encoder_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            cross_encoder_data.append(json.loads(line))
    print(f"Loaded {len(cross_encoder_data)} triple samples")
    
    # Load train/val split
    with open(prepared_dir / "train_val_split.json", 'r') as f:
        split = json.load(f)
    
    train_indices = set(split['train'])
    val_indices = set(split['val'])
    
    # Split data
    train_triplets = [t for i, t in enumerate(triplets) if i in train_indices]
    val_triplets = [t for i, t in enumerate(triplets) if i in val_indices]
    train_pairs = [p for i, p in enumerate(pairs) if i // 2 in train_indices]
    train_cross_encoder = [d for i, d in enumerate(cross_encoder_data) if i // 2 in train_indices]
    
    print(f"\nTrain triplets: {len(train_triplets)}")
    print(f"Val triplets: {len(val_triplets)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Train cross-encoder (for composite loss): {len(train_cross_encoder)}")
    
    # Create InputExamples
    pair_examples = create_pair_examples(train_pairs)
    triple_softmax_examples = create_triple_softmax_examples(train_cross_encoder)
    
    # Create DataLoaders
    batch_size = config['track_b']['batch_size']
    
    pair_loader = DataLoader(
        pair_examples,
        batch_size=batch_size * 2,  # MNR can handle larger batches
        shuffle=True
    )
    
    triple_softmax_loader = DataLoader(
        triple_softmax_examples,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Setup losses
    # 1. MultipleNegativesRankingLoss for contrastive learning
    mnr_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # 2. Composite loss (Margin + Pairwise Softmax)
    loss_weights = config['track_b'].get('loss_weights', {})
    composite_loss = CompositeLoss(
        model=model,
        margin=config['track_b'].get('margin', 0.2),
        temperature=config['track_b'].get('temperature', 0.7),
        weight_margin=loss_weights.get('margin', 1.0),
        weight_pairwise=loss_weights.get('pairwise', 0.5)
    )
    
    # Training parameters
    warmup_steps = config['track_b']['warmup_steps']
    
    # Setup evaluator
    base_evaluator = CustomEvaluator(
        dev_path=config['data']['dev_track_a'],
        device=device,
        save_path=config['track_b']['model_save_path']
    )
    evaluator = SequentialEvaluator([base_evaluator])
    
    # V2 Unified Training with Composite Loss
    print("\n" + "="*60)
    print("V2 UNIFIED TRAINING: Composite Loss (MNR + Margin + Pairwise)")
    print("="*60)
    
    # Combine MNR and Composite loss in multi-objective training
    train_objectives = []
    
    # Add MNR if we have pair data
    if len(pair_examples) > 0:
        weight_mnr = loss_weights.get('mnr', 0.5)
        if weight_mnr > 0:
            train_objectives.append((pair_loader, mnr_loss))
            print(f"✓ Added MultipleNegativesRankingLoss (weight: {weight_mnr})")
    
    # Add Composite loss if we have triple data
    if len(triple_softmax_examples) > 0:
        train_objectives.append((triple_softmax_loader, composite_loss))
        print(f"✓ Added CompositeLoss (Margin + Pairwise)")
    
    if not train_objectives:
        print("❌ No training data available!")
        sys.exit(1)
    
    # Train with all objectives simultaneously
    model.fit(
        train_objectives=train_objectives,
        epochs=config['track_b']['epochs'],
        warmup_steps=warmup_steps,
        optimizer_params={'lr': config['track_b']['learning_rate']},
        weight_decay=config['track_b']['weight_decay'],
        evaluation_steps=config['track_b']['eval_steps'],
        evaluator=evaluator,
        output_path=config['track_b']['model_save_path'] + "_temp",
        save_best_model=False,
        use_amp=config['track_b']['mixed_precision']
    )
    
    # Load best model
    model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    final_accuracy = evaluate_on_track_a(model, config['data']['dev_track_a'], device)
    print(f"\nFinal best model accuracy: {final_accuracy:.4f}")
    print(f"Model saved to: {config['track_b']['model_save_path']}")
    print(f"Embedding dimension: {projection_dim}")


if __name__ == "__main__":
    main()
