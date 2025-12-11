"""
Data preparation for Track A and Track B training.

This script:
- Loads Track A JSONL data
- Creates triplet datasets: (anchor, positive, negative)
- Creates pairwise datasets: (anchor, candidate, label)
- Implements optional data augmentation
- Saves prepared datasets for training

Works in both local and Google Colab environments.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import yaml
from sklearn.model_selection import KFold

# Import Colab utilities
try:
    from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, install_colab_dependencies, print_gpu_info
except ImportError:
    # If running locally without colab_utils in path
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, install_colab_dependencies, print_gpu_info
    except ImportError:
        # Fallback for local execution
        def is_colab(): return False
        def setup_colab_environment(x): return None
        def update_config_for_colab(c, p): return c
        def install_colab_dependencies(): pass
        def print_gpu_info(): pass


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_track_a_data(path: str) -> pd.DataFrame:
    """Load Track A JSONL data."""
    return pd.read_json(path, lines=True)


def create_triplets(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Create triplet dataset from Track A data.
    Each triplet: (anchor, positive, negative)
    - positive is the closer story
    - negative is the farther story
    """
    triplets = []
    for _, row in df.iterrows():
        anchor = row['anchor_text']
        
        if row['text_a_is_closer']:
            positive = row['text_a']
            negative = row['text_b']
        else:
            positive = row['text_b']
            negative = row['text_a']
        
        triplets.append({
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        })
    
    return triplets


def create_pairwise_dataset(df: pd.DataFrame, augment_swap: bool = True) -> List[Dict]:
    """
    Create pairwise dataset from Track A data.
    Each pair: (anchor, candidate, label) where label ∈ {1.0, 0.0}
    
    Args:
        df: Track A dataframe
        augment_swap: If True, create additional samples by swapping A/B
    """
    pairs = []
    
    for _, row in df.iterrows():
        anchor = row['anchor_text']
        text_a = row['text_a']
        text_b = row['text_b']
        a_is_closer = row['text_a_is_closer']
        
        # Original pairs
        pairs.append({
            'anchor': anchor,
            'candidate': text_a,
            'label': 1.0 if a_is_closer else 0.0
        })
        pairs.append({
            'anchor': anchor,
            'candidate': text_b,
            'label': 0.0 if a_is_closer else 1.0
        })
        
        # Augmentation: swap A and B (creates duplicate with inverted perspective)
        if augment_swap:
            # This doubles the dataset size but provides more training signal
            pass  # Already covered by adding both A and B pairs above
    
    return pairs


def create_cross_encoder_dataset(df: pd.DataFrame, augment_swap: bool = True) -> List[Dict]:
    """
    Create dataset for cross-encoder training.
    Each sample: (anchor, text_a, text_b, label)
    where label = 1 if A is closer, 0 if B is closer
    """
    samples = []
    
    for _, row in df.iterrows():
        samples.append({
            'anchor': row['anchor_text'],
            'text_a': row['text_a'],
            'text_b': row['text_b'],
            'label': 1 if row['text_a_is_closer'] else 0
        })
        
        # Augmentation: swap A and B positions
        if augment_swap:
            samples.append({
                'anchor': row['anchor_text'],
                'text_a': row['text_b'],
                'text_b': row['text_a'],
                'label': 0 if row['text_a_is_closer'] else 1
            })
    
    return samples


def create_kfold_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> List[Tuple]:
    """
    Create k-fold cross-validation splits.
    Returns list of (train_indices, val_indices) tuples.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    
    for train_idx, val_idx in kf.split(df):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    return splits


def save_dataset(data: List[Dict], path: str):
    """Save dataset as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """Main data preparation pipeline."""
    # Check if we're using Drive-based setup or cloned repo
    import os
    using_drive_setup = os.path.exists('/content/drive/MyDrive/narrative_similarity/config.yaml')
    
    # Setup Colab environment if running in Drive-based mode
    if is_colab() and using_drive_setup:
        print("="*60)
        print("RUNNING IN GOOGLE COLAB - DRIVE MODE")
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
            print("RUNNING IN GOOGLE COLAB - REPO MODE")
            print("="*60)
        else:
            print("Running locally")
        
        # Just load config normally
        config = load_config()
    
    seed = config['seed']
    random.seed(seed)
    
    # Create output directory
    output_dir = Path(config['data']['prepared_data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Track A data
    print("Loading Track A data...")
    df = load_track_a_data(config['data']['dev_track_a'])
    print(f"Loaded {len(df)} samples")
    
    # Create triplet dataset for Track B (bi-encoder)
    print("\nCreating triplet dataset for Track B...")
    triplets = create_triplets(df)
    save_dataset(triplets, output_dir / "triplets.jsonl")
    print(f"Saved {len(triplets)} triplets")
    
    # Create pairwise dataset for Track B
    print("\nCreating pairwise dataset for Track B...")
    pairs = create_pairwise_dataset(
        df, 
        augment_swap=config['augmentation']['swap_ab']
    )
    save_dataset(pairs, output_dir / "pairs.jsonl")
    print(f"Saved {len(pairs)} pairs")
    
    # Create cross-encoder dataset for Track A
    print("\nCreating cross-encoder dataset for Track A...")
    cross_encoder_data = create_cross_encoder_dataset(
        df,
        augment_swap=config['augmentation']['swap_ab']
    )
    save_dataset(cross_encoder_data, output_dir / "cross_encoder_data.jsonl")
    print(f"Saved {len(cross_encoder_data)} cross-encoder samples")
    
    # Create k-fold splits
    if config['track_a']['use_kfold']:
        print(f"\nCreating {config['track_a']['n_folds']}-fold splits...")
        splits = create_kfold_splits(df, config['track_a']['n_folds'], seed)
        with open(output_dir / "kfold_splits.json", 'w') as f:
            json.dump(splits, f)
        print(f"Saved {len(splits)} fold splits")
    
    # Save a simple train/val split (80/20) for Track B
    print("\nCreating train/val split (80/20) for Track B...")
    train_size = int(0.8 * len(df))
    indices = list(range(len(df)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    with open(output_dir / "train_val_split.json", 'w') as f:
        json.dump({
            'train': train_indices,
            'val': val_indices
        }, f)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    print("\n✓ Data preparation complete!")
    print(f"All prepared data saved to: {output_dir}")


if __name__ == "__main__":
    main()

