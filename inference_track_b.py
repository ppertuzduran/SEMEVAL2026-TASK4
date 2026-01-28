"""
Track B Test Inference: Generate embeddings for unlabeled test data.

This script:
- Loads unlabeled test data (list of texts)
- Uses trained bi-encoder model
- Generates embeddings in .npy format for submission
- Works in Google Colab or locally
- NO accuracy calculation (test data is unlabeled)
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import yaml
import argparse

# Add training directory to path for colab_utils
sys.path.insert(0, str(Path(__file__).parent / 'training'))
try:
    from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, print_gpu_info
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    is_colab = lambda: False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration if available."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


def main():
    """Main inference pipeline for test data (unlabeled)."""
    import time
    start_time = time.time()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Track B Test Inference')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: output/track_b_test.npy)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    
    print("="*60)
    print("TRACK B TEST INFERENCE (UNLABELED DATA)")
    print("="*60)
    
    # Setup environment
    if is_colab() and COLAB_AVAILABLE:
        colab_paths = setup_colab_environment()
        print_gpu_info()
        config_path = colab_paths['config_path']
    else:
        config_path = "config.yaml"
    
    # Load config
    print(f"\nLoading configuration...")
    config = load_config(config_path)
    if config is None:
        print("❌ config.yaml not found!")
        sys.exit(1)
    
    # Update config if in Colab
    if is_colab() and COLAB_AVAILABLE:
        config = update_config_for_colab(config, colab_paths)
    
    # Load test data
    print(f"\nLoading test data...")
    test_path = args.test_data
    if not Path(test_path).exists():
        print(f"❌ Test data not found at: {test_path}")
        sys.exit(1)
    
    data = pd.read_json(test_path, lines=True)
    print(f"✓ Loaded {len(data)} texts from: {test_path}")
    
    # Verify required column
    if 'text' not in data.columns:
        print(f"❌ Missing required column: 'text'")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)
    
    # Load fine-tuned model
    model_path = config['track_b']['model_save_path']
    if not Path(model_path).exists():
        print(f"\n❌ Fine-tuned model not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  python training/train_track_b.py")
        sys.exit(1)
    
    print(f"\nLoading fine-tuned model...")
    print(f"  Model: {model_path}")
    model = SentenceTransformer(model_path)
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Generate embeddings
    print(f"\nGenerating embeddings for test data...")
    inference_start = time.time()
    embeddings = model.encode(
        data["text"],
        show_progress_bar=True,
        batch_size=args.batch_size,
        convert_to_numpy=True
    )
    inference_time = time.time() - inference_start
    print(f"✓ Embeddings generated in {inference_time:.1f} seconds")
    print(f"  ({len(data)/inference_time:.1f} samples/sec)")
    print(f"  Shape: {embeddings.shape}")
    
    # Verify embedding dimension
    expected_dim = config['track_b'].get('projection_dim', 512)
    if embeddings.shape[1] != expected_dim:
        print(f"⚠️  Warning: Expected {expected_dim}-dim embeddings, got {embeddings.shape[1]}-dim")
        print(f"  Make sure you're using the correct Track B model!")
    else:
        print(f"✓ Verified: {expected_dim}-dim embeddings")
    
    # Save embeddings
    print(f"\nSaving embeddings...")
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / "track_b_test.npy"
    
    np.save(output_path, embeddings)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Embeddings saved to: {output_path}")
    print(f"✓ Total samples: {len(data)}")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")
    print(f"\n📦 Ready for submission to CodaBench!")


if __name__ == "__main__":
    main()
