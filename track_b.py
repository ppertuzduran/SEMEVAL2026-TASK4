"""
Track B Inference: Embedding-based narrative similarity (v11: Qwen3-Embedding compatible).

Google Colab inference script.

This script:
- Runs ONLY in Google Colab with GPU
- Loads fine-tuned bi-encoder from Google Drive (supports both BGE-large and Qwen3-Embedding)
- Loads data from Google Drive
- Generates embeddings and evaluates on Track A
- Saves embeddings (.npy) to Google Drive
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import yaml

# Add training directory to path for colab_utils
sys.path.insert(0, '/content/project/training')
from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, print_gpu_info


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


def main():
    """Main inference pipeline - Google Colab only."""
    import time
    start_time = time.time()
    
    # Check if running in Colab
    if not is_colab():
        print("⚠️  This script is designed for Google Colab only!")
        print("Inference must be run in Colab with GPU enabled.")
        print("\nPlease:")
        print("1. Open Google Colab")
        print("2. Runtime → Change runtime type → T4 GPU")
        print("3. Run this script")
        sys.exit(1)
    
    print("="*60)
    print("TRACK B INFERENCE - GOOGLE COLAB")
    print("="*60)
    
    # Setup Colab environment
    colab_paths = setup_colab_environment()
    print_gpu_info()
    
    # Load and update config with Colab paths
    print(f"\nLoading configuration...")
    config = load_config(colab_paths['config_path'])
    if config is None:
        print("❌ config.yaml not found!")
        sys.exit(1)
    
    config = update_config_for_colab(config, colab_paths)
    
    # Load data from Drive
    print(f"\nLoading Track B data from Drive...")
    track_b_path = config['data']['dev_track_b']
    data = pd.read_json(track_b_path, lines=True)
    print(f"✓ Loaded {len(data)} texts from: {track_b_path}")
    
    # Load fine-tuned model from Drive
    model_path = config['track_b']['model_save_path']
    if not Path(model_path).exists():
        print(f"\n❌ Fine-tuned model not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  !python training/train_track_b.py")
        sys.exit(1)
    
    print(f"\nLoading fine-tuned model from Drive...")
    print(f"  Model: {model_path}")
    model = SentenceTransformer(model_path)
    print(f"✓ Model loaded")
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    inference_start = time.time()
    embeddings = model.encode(
        data["text"],
        show_progress_bar=True,
        batch_size=32,  # T4 can handle larger batches
        convert_to_numpy=True
    )
    inference_time = time.time() - inference_start
    print(f"✓ Embeddings generated in {inference_time:.1f} seconds")
    print(f"  ({len(data)/inference_time:.1f} samples/sec)")
    print(f"  Shape: {embeddings.shape}")
    
    # Verify embedding dimension (should be 512 with v2 projection head)
    expected_dim = 512
    if embeddings.shape[1] != expected_dim:
        print(f"⚠️  Warning: Expected {expected_dim}-dim embeddings, got {embeddings.shape[1]}-dim")
        print(f"  Make sure you're using a Track B model with projection head!")
    else:
        print(f"✓ Verified: {expected_dim}-dim embeddings (v2 projection head)")
    
    # Create lookup for evaluation
    embedding_lookup = dict(zip(data["text"], embeddings))
    
    # Evaluate on Track A data from Drive
    print(f"\nEvaluating on Track A dev set...")
    track_a_path = config['data']['dev_track_a']
    accuracy = evaluate(track_a_path, embedding_lookup)
    print(f"✓ Accuracy: {accuracy:.4f}")
    
    # Save embeddings to Drive
    print(f"\nSaving embeddings to Drive...")
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "track_b.npy"
    np.save(output_path, embeddings)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Embeddings saved to: {output_path}")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
