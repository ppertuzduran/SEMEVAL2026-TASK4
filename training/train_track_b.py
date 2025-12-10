"""
Training script for Track B: Bi-encoder (embedding model).

Uses sentence-transformers with:
- TripletLoss (margin-based)
- MultipleNegativesRankingLoss
- Mixed precision training
- Evaluation on Track A dev set
"""

import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from tqdm import tqdm
import pandas as pd
from sentence_transformers.util import cos_sim


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


def main():
    """Main training pipeline."""
    # Load config
    config = load_config()
    set_seed(config['seed'])
    
    # Setup device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    print(f"\nLoading base model: {config['track_b']['base_model']}")
    model = SentenceTransformer(config['track_b']['base_model'], device=device)
    
    # Load prepared data
    prepared_dir = Path(config['data']['prepared_data_dir'])
    print("\nLoading triplets...")
    triplets = load_triplets(prepared_dir / "triplets.jsonl")
    print(f"Loaded {len(triplets)} triplets")
    
    print("Loading pairs...")
    pairs = load_pairs(prepared_dir / "pairs.jsonl")
    print(f"Loaded {len(pairs)} pairs")
    
    # Load train/val split
    with open(prepared_dir / "train_val_split.json", 'r') as f:
        split = json.load(f)
    
    train_indices = set(split['train'])
    val_indices = set(split['val'])
    
    # Split triplets and pairs
    train_triplets = [t for i, t in enumerate(triplets) if i in train_indices]
    val_triplets = [t for i, t in enumerate(triplets) if i in val_indices]
    
    # For pairs, we have 2 pairs per original sample
    train_pairs = [p for i, p in enumerate(pairs) if i // 2 in train_indices]
    
    print(f"\nTrain triplets: {len(train_triplets)}")
    print(f"Val triplets: {len(val_triplets)}")
    print(f"Train pairs: {len(train_pairs)}")
    
    # Create InputExamples
    triplet_examples = create_triplet_examples(train_triplets)
    pair_examples = create_pair_examples(train_pairs)
    
    # Create DataLoaders
    batch_size = config['track_b']['batch_size']
    
    triplet_loader = DataLoader(
        triplet_examples,
        batch_size=batch_size,
        shuffle=True
    )
    
    pair_loader = DataLoader(
        pair_examples,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Setup losses
    triplet_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=config['track_b']['triplet_margin']
    )
    
    mnr_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Training parameters
    num_epochs = config['track_b']['epochs']
    warmup_steps = config['track_b']['warmup_steps']
    
    # Setup evaluator
    evaluator = CustomEvaluator(
        dev_path=config['data']['dev_track_a'],
        device=device,
        save_path=config['track_b']['model_save_path']
    )
    
    # Training strategy: Alternate between triplet and MNR loss
    # We'll train with triplet loss, then evaluate
    print("\n" + "="*60)
    print("Starting training with TripletLoss...")
    print("="*60)
    
    model.fit(
        train_objectives=[(triplet_loader, triplet_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': config['track_b']['learning_rate']},
        weight_decay=config['track_b']['weight_decay'],
        evaluation_steps=config['track_b']['eval_steps'],
        evaluator=evaluator,
        output_path=config['track_b']['model_save_path'] + "_temp",
        save_best_model=False,  # We handle this in the evaluator
        use_amp=config['track_b']['mixed_precision']
    )
    
    # If we want to use MNR loss, we can do a second phase
    if config['track_b']['use_multiple_negatives_ranking'] and len(pair_examples) > 0:
        print("\n" + "="*60)
        print("Fine-tuning with MultipleNegativesRankingLoss...")
        print("="*60)
        
        # Load best model from first phase
        model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
        
        # Recreate evaluator with updated model
        evaluator = CustomEvaluator(
            dev_path=config['data']['dev_track_a'],
            device=device,
            save_path=config['track_b']['model_save_path']
        )
        evaluator.best_accuracy = evaluator(model, "", 0, 0)
        
        model.fit(
            train_objectives=[(pair_loader, mnr_loss)],
            epochs=num_epochs // 2,  # Fewer epochs for fine-tuning
            warmup_steps=warmup_steps // 2,
            optimizer_params={'lr': config['track_b']['learning_rate'] / 2},
            weight_decay=config['track_b']['weight_decay'],
            evaluation_steps=config['track_b']['eval_steps'],
            evaluator=evaluator,
            output_path=config['track_b']['model_save_path'] + "_temp2",
            save_best_model=False,
            use_amp=config['track_b']['mixed_precision']
        )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    final_model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
    final_accuracy = evaluate_on_track_a(final_model, config['data']['dev_track_a'], device)
    print(f"\nFinal best model accuracy: {final_accuracy:.4f}")
    print(f"Model saved to: {config['track_b']['model_save_path']}")


if __name__ == "__main__":
    main()

