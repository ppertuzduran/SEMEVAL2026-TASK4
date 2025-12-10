"""
Local evaluation script for Track A and Track B.

This script evaluates model predictions against ground truth labels
and computes various metrics.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration if available."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


def evaluate_track_a(predictions_path: str, labels_path: str) -> Dict:
    """
    Evaluate Track A predictions.
    
    Args:
        predictions_path: Path to predictions JSONL
        labels_path: Path to ground truth JSONL
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load predictions and labels
    pred_df = pd.read_json(predictions_path, lines=True)
    label_df = pd.read_json(labels_path, lines=True)
    
    # Ensure same order
    if 'id' in pred_df.columns and 'id' in label_df.columns:
        pred_df = pred_df.sort_values('id').reset_index(drop=True)
        label_df = label_df.sort_values('id').reset_index(drop=True)
    
    # Get predictions and labels
    y_pred = pred_df['text_a_is_closer'].values
    y_true = label_df['text_a_is_closer'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true),
        'correct_predictions': int((y_pred == y_true).sum())
    }
    
    return metrics


def evaluate_track_b_with_track_a(embeddings_path: str, labels_path: str) -> Dict:
    """
    Evaluate Track B embeddings using Track A labels.
    
    Args:
        embeddings_path: Path to embeddings .npy file
        labels_path: Path to Track A ground truth JSONL
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sentence_transformers.util import cos_sim
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Load Track B data (to get text-to-embedding mapping)
    config = load_config()
    if config:
        track_b_path = config['data']['dev_track_b']
    else:
        track_b_path = "data/dev_track_b.jsonl"
    
    track_b_df = pd.read_json(track_b_path, lines=True)
    
    # Create text to embedding lookup
    text_to_embedding = {
        text: embeddings[i] 
        for i, text in enumerate(track_b_df['text'])
    }
    
    # Load Track A labels
    label_df = pd.read_json(labels_path, lines=True)
    
    # Calculate predictions
    predictions = []
    for _, row in label_df.iterrows():
        anchor_emb = text_to_embedding[row['anchor_text']]
        a_emb = text_to_embedding[row['text_a']]
        b_emb = text_to_embedding[row['text_b']]
        
        sim_a = cos_sim(anchor_emb, a_emb).item()
        sim_b = cos_sim(anchor_emb, b_emb).item()
        
        predictions.append(sim_a > sim_b)
    
    # Calculate metrics
    y_true = label_df['text_a_is_closer'].values
    y_pred = np.array(predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_samples': len(y_true),
        'correct_predictions': int((y_pred == y_true).sum()),
        'embedding_dim': embeddings.shape[1]
    }
    
    return metrics


def print_metrics(metrics: Dict, title: str):
    """Print metrics in a nice format."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1_score']:.4f}")
    print(f"Correct:    {metrics['correct_predictions']}/{metrics['total_samples']}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              False  True")
        print(f"Actual False  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       True   {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    if 'embedding_dim' in metrics:
        print(f"Embedding dim: {metrics['embedding_dim']}")


def compare_models(metrics_list: list, names: list):
    """Compare multiple models."""
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Accuracy':<10} {'F1 Score':<10}")
    print(f"{'-'*60}")
    
    for name, metrics in zip(names, metrics_list):
        print(f"{name:<30} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f}")


def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Track A and Track B models")
    parser.add_argument(
        '--track',
        type=str,
        choices=['a', 'b', 'both'],
        default='both',
        help='Which track to evaluate'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        default='output/track_a.jsonl',
        help='Path to Track A predictions (JSONL)'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default='output/track_b.npy',
        help='Path to Track B embeddings (.npy)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='data/dev_track_a.jsonl',
        help='Path to ground truth labels'
    )
    parser.add_argument(
        '--save',
        type=str,
        default='output/evaluation_results.json',
        help='Path to save evaluation results'
    )
    
    args = parser.parse_args()
    
    results = {}
    
    # Evaluate Track A
    if args.track in ['a', 'both']:
        if Path(args.predictions).exists():
            print(f"Evaluating Track A predictions from: {args.predictions}")
            track_a_metrics = evaluate_track_a(args.predictions, args.labels)
            print_metrics(track_a_metrics, "Track A: Cross-Encoder Results")
            results['track_a'] = track_a_metrics
        else:
            print(f"Track A predictions not found at: {args.predictions}")
    
    # Evaluate Track B
    if args.track in ['b', 'both']:
        if Path(args.embeddings).exists():
            print(f"\nEvaluating Track B embeddings from: {args.embeddings}")
            track_b_metrics = evaluate_track_b_with_track_a(args.embeddings, args.labels)
            print_metrics(track_b_metrics, "Track B: Bi-Encoder Results")
            results['track_b'] = track_b_metrics
        else:
            print(f"Track B embeddings not found at: {args.embeddings}")
    
    # Save results
    if results:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

