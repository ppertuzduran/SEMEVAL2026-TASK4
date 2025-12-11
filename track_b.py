"""
Track B system: Embedding-based narrative similarity.

This script embeds texts from Track B file using either:
1. Fine-tuned bi-encoder model (if available)
2. Baseline sentence-transformer model

The embeddings are evaluated using labels from Track A.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import yaml


def evaluate(labeled_data_path, embedding_lookup):
    df = pd.read_json(labeled_data_path, lines=True)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Look up cosine similarities
    df["sim_a"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]), axis=1
    )
    df["sim_b"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]), axis=1
    )

    # Predict and calculate accuracy
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration if available."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


def get_model(use_finetuned: bool = True) -> SentenceTransformer:
    """
    Load the appropriate model.
    
    Args:
        use_finetuned: If True, attempt to load fine-tuned model
        
    Returns:
        SentenceTransformer model
    """
    config = load_config()
    
    if use_finetuned and config:
        model_path = config['track_b']['model_save_path']
        if Path(model_path).exists():
            print(f"Loading fine-tuned model from: {model_path}")
            return SentenceTransformer(model_path)
        else:
            print(f"Fine-tuned model not found at {model_path}")
            print("Falling back to baseline model")
    
    # Fallback to baseline
    baseline_model = "all-MiniLM-L6-v2"
    if config:
        baseline_model = config['track_b']['base_model']
    
    print(f"Loading baseline model: {baseline_model}")
    return SentenceTransformer(baseline_model)


def main():
    """Main inference pipeline."""
    # Load config to get data paths
    config = load_config()
    
    # Determine data paths (from config or default)
    if config and 'data' in config:
        track_b_path = config['data'].get('dev_track_b', 'data/dev_track_b.jsonl')
        track_a_path = config['data'].get('dev_track_a', 'data/dev_track_a.jsonl')
        output_dir = Path(config['data'].get('output_dir', 'output'))
    else:
        track_b_path = 'data/dev_track_b.jsonl'
        track_a_path = 'data/dev_track_a.jsonl'
        output_dir = Path('output')
    
    # Load data
    print(f"Loading data from: {track_b_path}")
    data = pd.read_json(track_b_path, lines=True)
    print(f"✓ Loaded {len(data)} texts from Track B")
    
    # Load model (fine-tuned if available, otherwise baseline)
    model = get_model(use_finetuned=True)
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(data["text"], show_progress_bar=True)
    
    # Create lookup for evaluation
    embedding_lookup = dict(zip(data["text"], embeddings))
    
    # Evaluate on Track A
    print(f"\nEvaluating on Track A dev set: {track_a_path}")
    accuracy = evaluate(track_a_path, embedding_lookup)
    print(f"✓ Accuracy on Track A dev set: {accuracy:.4f}")
    
    # Save embeddings
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "track_b.npy"
    np.save(output_path, embeddings)
    print(f"\n✓ Embeddings saved to: {output_path}")
    print(f"✓ Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
