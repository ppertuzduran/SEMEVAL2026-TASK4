"""
Script to augment training data for Track B.

This script:
1. Loads prepared training data (triplets, pairs, cross-encoder data)
2. Applies augmentation to generate 2x more training examples
3. Saves augmented dataset to data/prepared/augmented/
4. Generates quality report

Usage (Colab):
    python scripts/augment_training_data.py

Note: This script is Colab-aware and will automatically use Google Drive paths.
"""

import json
import sys
import os
import random
from pathlib import Path
from collections import defaultdict
import yaml
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_augmentation import create_default_pipeline

# Import Colab utilities
try:
    from training.colab_utils import is_colab, setup_colab_environment, update_config_for_colab
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    def is_colab():
        return False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: Path):
    """Save data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word-overlap similarity.
    
    Used to filter out low-quality augmentations.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def augment_triplets(
    triplets: list,
    pipeline,
    n_augmentations: int = 2,
    min_similarity: float = 0.3,
    max_similarity: float = 0.9
) -> list:
    """
    Augment triplet data.
    
    For each triplet, we augment the anchor and positive texts.
    Negatives are kept as-is to maintain hard negative relationships.
    
    Args:
        triplets: Original triplets
        pipeline: Augmentation pipeline
        n_augmentations: Number of augmented versions per triplet
        min_similarity: Minimum similarity to original (filter too different)
        max_similarity: Maximum similarity to original (filter too similar)
        
    Returns:
        Augmented triplets (includes originals)
    """
    print(f"\n{'='*60}")
    print("AUGMENTING TRIPLETS")
    print(f"{'='*60}")
    
    augmented = []
    stats = {
        'total_original': len(triplets),
        'total_augmented': 0,
        'filtered_too_similar': 0,
        'filtered_too_different': 0
    }
    
    # Keep originals
    augmented.extend(triplets)
    
    for triplet in tqdm(triplets, desc="Augmenting triplets"):
        anchor = triplet['anchor']
        positive = triplet['positive']
        negative = triplet['negative']
        
        # Augment anchor
        anchor_augmented = pipeline.augment(anchor, n_augmentations=n_augmentations)
        
        # Augment positive
        positive_augmented = pipeline.augment(positive, n_augmentations=n_augmentations)
        
        # Create augmented triplets
        for aug_anchor, aug_positive in zip(anchor_augmented, positive_augmented):
            # Quality check: filter if too similar or too different
            anchor_sim = calculate_similarity(anchor, aug_anchor)
            positive_sim = calculate_similarity(positive, aug_positive)
            
            if anchor_sim > max_similarity or positive_sim > max_similarity:
                stats['filtered_too_similar'] += 1
                continue
            
            if anchor_sim < min_similarity or positive_sim < min_similarity:
                stats['filtered_too_different'] += 1
                continue
            
            # Add augmented triplet
            augmented.append({
                'anchor': aug_anchor,
                'positive': aug_positive,
                'negative': negative,  # Keep original negative
                'augmented': True
            })
            stats['total_augmented'] += 1
    
    print(f"\n✓ Triplets: {stats['total_original']} → {len(augmented)}")
    print(f"  Added: {stats['total_augmented']} augmented")
    print(f"  Filtered (too similar): {stats['filtered_too_similar']}")
    print(f"  Filtered (too different): {stats['filtered_too_different']}")
    
    return augmented


def augment_pairs(
    pairs: list,
    pipeline,
    n_augmentations: int = 2,
    min_similarity: float = 0.3,
    max_similarity: float = 0.9
) -> list:
    """
    Augment pair data.
    
    Args:
        pairs: Original pairs
        pipeline: Augmentation pipeline
        n_augmentations: Number of augmented versions per pair
        min_similarity: Minimum similarity to original
        max_similarity: Maximum similarity to original
        
    Returns:
        Augmented pairs (includes originals)
    """
    print(f"\n{'='*60}")
    print("AUGMENTING PAIRS")
    print(f"{'='*60}")
    
    augmented = []
    stats = {
        'total_original': len(pairs),
        'total_augmented': 0,
        'filtered_too_similar': 0,
        'filtered_too_different': 0
    }
    
    # Keep originals
    augmented.extend(pairs)
    
    for pair in tqdm(pairs, desc="Augmenting pairs"):
        anchor = pair['anchor']
        candidate = pair['candidate']
        label = pair['label']
        
        # Only augment positive pairs (label > 0.5)
        if label <= 0.5:
            continue
        
        # Augment both texts
        anchor_augmented = pipeline.augment(anchor, n_augmentations=n_augmentations)
        candidate_augmented = pipeline.augment(candidate, n_augmentations=n_augmentations)
        
        for aug_anchor, aug_candidate in zip(anchor_augmented, candidate_augmented):
            # Quality check
            anchor_sim = calculate_similarity(anchor, aug_anchor)
            candidate_sim = calculate_similarity(candidate, aug_candidate)
            
            if anchor_sim > max_similarity or candidate_sim > max_similarity:
                stats['filtered_too_similar'] += 1
                continue
            
            if anchor_sim < min_similarity or candidate_sim < min_similarity:
                stats['filtered_too_different'] += 1
                continue
            
            # Add augmented pair
            augmented.append({
                'anchor': aug_anchor,
                'candidate': aug_candidate,
                'label': label,
                'augmented': True
            })
            stats['total_augmented'] += 1
    
    print(f"\n✓ Pairs: {stats['total_original']} → {len(augmented)}")
    print(f"  Added: {stats['total_augmented']} augmented")
    print(f"  Filtered (too similar): {stats['filtered_too_similar']}")
    print(f"  Filtered (too different): {stats['filtered_too_different']}")
    
    return augmented


def augment_cross_encoder_data(
    data: list,
    pipeline,
    n_augmentations: int = 2,
    min_similarity: float = 0.3,
    max_similarity: float = 0.9
) -> list:
    """
    Augment cross-encoder data.
    
    Args:
        data: Original cross-encoder data
        pipeline: Augmentation pipeline
        n_augmentations: Number of augmented versions
        min_similarity: Minimum similarity to original
        max_similarity: Maximum similarity to original
        
    Returns:
        Augmented data (includes originals)
    """
    print(f"\n{'='*60}")
    print("AUGMENTING CROSS-ENCODER DATA")
    print(f"{'='*60}")
    
    augmented = []
    stats = {
        'total_original': len(data),
        'total_augmented': 0,
        'filtered_too_similar': 0,
        'filtered_too_different': 0
    }
    
    # Keep originals
    augmented.extend(data)
    
    for item in tqdm(data, desc="Augmenting cross-encoder"):
        anchor = item['anchor']
        text_a = item['text_a']
        text_b = item['text_b']
        label = item['label']
        
        # Augment all three texts
        anchor_augmented = pipeline.augment(anchor, n_augmentations=n_augmentations)
        text_a_augmented = pipeline.augment(text_a, n_augmentations=n_augmentations)
        text_b_augmented = pipeline.augment(text_b, n_augmentations=n_augmentations)
        
        for aug_anchor, aug_a, aug_b in zip(anchor_augmented, text_a_augmented, text_b_augmented):
            # Quality check
            anchor_sim = calculate_similarity(anchor, aug_anchor)
            a_sim = calculate_similarity(text_a, aug_a)
            b_sim = calculate_similarity(text_b, aug_b)
            
            avg_sim = (anchor_sim + a_sim + b_sim) / 3
            
            if avg_sim > max_similarity:
                stats['filtered_too_similar'] += 1
                continue
            
            if avg_sim < min_similarity:
                stats['filtered_too_different'] += 1
                continue
            
            # Add augmented item
            augmented.append({
                'anchor': aug_anchor,
                'text_a': aug_a,
                'text_b': aug_b,
                'label': label,
                'augmented': True
            })
            stats['total_augmented'] += 1
    
    print(f"\n✓ Cross-encoder: {stats['total_original']} → {len(augmented)}")
    print(f"  Added: {stats['total_augmented']} augmented")
    print(f"  Filtered (too similar): {stats['filtered_too_similar']}")
    print(f"  Filtered (too different): {stats['filtered_too_different']}")
    
    return augmented


def save_samples(augmented_data: dict, output_dir: Path):
    """Save sample augmentations for manual inspection."""
    samples_file = output_dir / "augmentation_samples.txt"
    
    with open(samples_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AUGMENTATION SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        # Sample triplets
        f.write("TRIPLET SAMPLES\n")
        f.write("-"*80 + "\n")
        triplets = augmented_data['triplets']
        augmented_triplets = [t for t in triplets if t.get('augmented', False)]
        
        for i, triplet in enumerate(random.sample(augmented_triplets, min(5, len(augmented_triplets)))):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  Anchor: {triplet['anchor'][:100]}...\n")
            f.write(f"  Positive: {triplet['positive'][:100]}...\n")
            f.write(f"  Negative: {triplet['negative'][:100]}...\n")
        
        # Sample pairs
        f.write("\n" + "="*80 + "\n")
        f.write("PAIR SAMPLES\n")
        f.write("-"*80 + "\n")
        pairs = augmented_data['pairs']
        augmented_pairs = [p for p in pairs if p.get('augmented', False)]
        
        for i, pair in enumerate(random.sample(augmented_pairs, min(5, len(augmented_pairs)))):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  Anchor: {pair['anchor'][:100]}...\n")
            f.write(f"  Candidate: {pair['candidate'][:100]}...\n")
            f.write(f"  Label: {pair['label']}\n")
    
    print(f"\n✓ Samples saved to: {samples_file}")


def main():
    """Main augmentation pipeline."""
    print("="*60)
    print("DATA AUGMENTATION FOR TRACK B")
    print("="*60)
    
    # Setup Colab environment if running in Colab
    if is_colab():
        print("\n🔧 Running in Google Colab - setting up environment...")
        colab_paths = setup_colab_environment()
        
        # Load and update config for Colab
        config = load_config(colab_paths['config_path'])
        config = update_config_for_colab(config, colab_paths)
    else:
        # Local execution
        print("\n💻 Running locally")
        config = load_config()
    
    # Setup paths - augmented data is a peer to prepared, not a child
    prepared_dir = Path(config['data']['prepared_data_dir'])
    
    # Augmented data goes in data/augmented/, not data/prepared/augmented/
    data_root = prepared_dir.parent  # Get data/ directory
    output_dir = data_root / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput directory: {prepared_dir}")
    print(f"Output directory: {output_dir}")
    print(f"  (Augmented data is stored as a peer to 'prepared', not inside it)")
    
    # Load data
    print("\nLoading prepared data...")
    triplets = load_jsonl(prepared_dir / "triplets.jsonl")
    pairs = load_jsonl(prepared_dir / "pairs.jsonl")
    cross_encoder_data = load_jsonl(prepared_dir / "cross_encoder_data.jsonl")
    
    print(f"  Triplets: {len(triplets)}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Cross-encoder: {len(cross_encoder_data)}")
    
    # Create augmentation pipeline
    pipeline = create_default_pipeline()
    
    # Augmentation parameters
    n_augmentations = 2  # 2x augmentation
    min_similarity = 0.3
    max_similarity = 0.9
    
    print(f"\nAugmentation settings:")
    print(f"  n_augmentations: {n_augmentations}")
    print(f"  min_similarity: {min_similarity}")
    print(f"  max_similarity: {max_similarity}")
    
    # Augment data
    augmented_triplets = augment_triplets(
        triplets, pipeline, n_augmentations, min_similarity, max_similarity
    )
    
    augmented_pairs = augment_pairs(
        pairs, pipeline, n_augmentations, min_similarity, max_similarity
    )
    
    augmented_cross_encoder = augment_cross_encoder_data(
        cross_encoder_data, pipeline, n_augmentations, min_similarity, max_similarity
    )
    
    # Save augmented data
    print(f"\n{'='*60}")
    print("SAVING AUGMENTED DATA")
    print(f"{'='*60}")
    
    save_jsonl(augmented_triplets, output_dir / "triplets.jsonl")
    save_jsonl(augmented_pairs, output_dir / "pairs.jsonl")
    save_jsonl(augmented_cross_encoder, output_dir / "cross_encoder_data.jsonl")
    
    # Generate new train/val split that includes augmented data
    # Load original split
    with open(prepared_dir / "train_val_split.json", 'r') as f:
        original_split = json.load(f)
    
    # The augmented data includes all original data at the beginning
    # followed by augmented samples. We need to update the split to include
    # all indices for the augmented dataset.
    
    # Original triplets are at indices 0 to len(triplets)-1
    # Augmented triplets start at len(triplets)
    # Total triplets: len(augmented_triplets)
    
    # For the new split:
    # - Training indices: all indices from 0 to len(augmented_triplets)-1 that correspond
    #   to original training indices OR augmented data
    # - Validation indices: same as original (these are indices in the original dataset)
    
    # Since augmented data is appended after originals, we need to:
    # 1. Keep the same validation indices (they reference original data)
    # 2. Expand training indices to include all augmented samples
    
    original_train_indices = set(original_split['train'])
    original_val_indices = set(original_split['val'])
    
    # New training indices include:
    # - Original training indices (0 to len(triplets)-1 if in original train set)
    # - All augmented indices (len(triplets) to len(augmented_triplets)-1)
    new_train_indices = []
    
    # Add original training indices
    for idx in range(len(triplets)):
        if idx in original_train_indices:
            new_train_indices.append(idx)
    
    # Add all augmented indices (they are all for training)
    for idx in range(len(triplets), len(augmented_triplets)):
        new_train_indices.append(idx)
    
    # Validation indices stay the same (they reference original data only)
    new_val_indices = list(original_val_indices)
    
    # Save new split
    new_split = {
        'train': new_train_indices,
        'val': new_val_indices
    }
    
    with open(output_dir / "train_val_split.json", 'w') as f:
        json.dump(new_split, f, indent=2)
    
    print(f"\n✓ Saved augmented triplets: {len(augmented_triplets)}")
    print(f"✓ Saved augmented pairs: {len(augmented_pairs)}")
    print(f"✓ Saved augmented cross-encoder: {len(augmented_cross_encoder)}")
    print(f"✓ Generated new train/val split:")
    print(f"    Train indices: {len(new_train_indices)} (includes {len(new_train_indices) - len(original_train_indices)} augmented)")
    print(f"    Val indices: {len(new_val_indices)} (original validation set)")
    
    # Save samples for inspection
    save_samples({
        'triplets': augmented_triplets,
        'pairs': augmented_pairs,
        'cross_encoder': augmented_cross_encoder
    }, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nDataset size increase:")
    print(f"  Triplets: {len(triplets)} → {len(augmented_triplets)} ({len(augmented_triplets)/len(triplets):.1f}x)")
    print(f"  Pairs: {len(pairs)} → {len(augmented_pairs)} ({len(augmented_pairs)/len(pairs):.1f}x)")
    print(f"  Cross-encoder: {len(cross_encoder_data)} → {len(augmented_cross_encoder)} ({len(augmented_cross_encoder)/len(cross_encoder_data):.1f}x)")
    
    print(f"\n✓ Augmented data saved to: {output_dir}")
    print(f"✓ Review samples in: {output_dir / 'augmentation_samples.txt'}")
    print(f"\nNext step: Enable augmented data in config.yaml:")
    print(f"  augmentation:")
    print(f"    use_augmented_data: true")
    print(f"\nThe training script will automatically use data/augmented/ when enabled.")



if __name__ == "__main__":
    main()
