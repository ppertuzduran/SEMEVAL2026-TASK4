"""
Evaluation script for Track A and Track B.

Google Colab script.

Evaluates model predictions from Google Drive against ground truth labels
and computes various metrics.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml

# Add training directory to path for colab_utils
sys.path.insert(0, '/content/project/training')
from colab_utils import is_colab, setup_colab_environment, update_config_for_colab


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
    
    # Check if scores are available in predictions
    has_scores = 'score' in pred_df.columns or 'probability' in pred_df.columns
    score_field = 'score' if 'score' in pred_df.columns else 'probability' if 'probability' in pred_df.columns else None
    
    # Track incorrect predictions
    incorrect_predictions = []
    for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
        if pred != true:
            row = label_df.iloc[idx]
            pred_row = pred_df.iloc[idx]
            item = {
                'index': int(idx),
                'anchor': row['anchor_text'],
                'text_a': row['text_a'],
                'text_b': row['text_b'],
                'predicted': bool(pred),
                'actual': bool(true)
            }
            # Add score if available
            if score_field:
                item['score'] = float(pred_row[score_field])
            incorrect_predictions.append(item)
    
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
        'correct_predictions': int((y_pred == y_true).sum()),
        # Diagnostic information
        'incorrect_predictions_count': len(incorrect_predictions),
        'incorrect_predictions': incorrect_predictions[:10],  # First 10
        'has_scores': has_scores
    }
    
    return metrics


def evaluate_track_b_with_track_a(embeddings_path: str, labels_path: str, track_b_path: str = None, uncertainty_threshold: float = 0.05) -> Dict:
    """
    Evaluate Track B embeddings using Track A labels.
    
    Args:
        embeddings_path: Path to embeddings .npy file
        labels_path: Path to Track A ground truth JSONL
        track_b_path: Path to Track B data JSONL (for text-to-embedding mapping)
        uncertainty_threshold: Threshold for uncertain predictions (default: 0.05)
        
    Returns:
        Dictionary of evaluation metrics including failure diagnostics
    """
    from sentence_transformers.util import cos_sim
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Load Track B data (to get text-to-embedding mapping)
    if track_b_path is None:
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
    
    # Track failures and diagnostics
    missing_embeddings = []
    uncertain_predictions = []
    incorrect_predictions = []
    predictions = []
    similarity_scores = []
    
    # Calculate predictions
    for idx, row in label_df.iterrows():
        # Check for missing embeddings
        missing_texts = []
        if row['anchor_text'] not in text_to_embedding:
            missing_texts.append(('anchor', row['anchor_text']))
        if row['text_a'] not in text_to_embedding:
            missing_texts.append(('text_a', row['text_a']))
        if row['text_b'] not in text_to_embedding:
            missing_texts.append(('text_b', row['text_b']))
        
        if missing_texts:
            missing_embeddings.append({
                'index': int(idx),
                'missing_texts': missing_texts,
                'anchor': row['anchor_text'],
                'text_a': row['text_a'],
                'text_b': row['text_b']
            })
            # Skip this sample
            predictions.append(False)  # Default prediction
            similarity_scores.append({'sim_a': None, 'sim_b': None, 'diff': None})
            continue
        
        # Get embeddings
        anchor_emb = text_to_embedding[row['anchor_text']]
        a_emb = text_to_embedding[row['text_a']]
        b_emb = text_to_embedding[row['text_b']]
        
        # Calculate similarities
        sim_a = cos_sim(anchor_emb, a_emb).item()
        sim_b = cos_sim(anchor_emb, b_emb).item()
        sim_diff = abs(sim_a - sim_b)
        
        # Make prediction
        pred = sim_a > sim_b
        predictions.append(pred)
        
        # Store similarity scores
        similarity_scores.append({
            'sim_a': float(sim_a),
            'sim_b': float(sim_b),
            'diff': float(sim_diff)
        })
        
        # Track uncertain predictions (small difference)
        if sim_diff < uncertainty_threshold:
            uncertain_predictions.append({
                'index': int(idx),
                'anchor': row['anchor_text'],
                'text_a': row['text_a'],
                'text_b': row['text_b'],
                'sim_a': float(sim_a),
                'sim_b': float(sim_b),
                'diff': float(sim_diff),
                'predicted': pred,
                'actual': bool(row['text_a_is_closer'])
            })
        
        # Track incorrect predictions
        if pred != row['text_a_is_closer']:
            incorrect_predictions.append({
                'index': int(idx),
                'anchor': row['anchor_text'],
                'text_a': row['text_a'],
                'text_b': row['text_b'],
                'sim_a': float(sim_a),
                'sim_b': float(sim_b),
                'diff': float(sim_diff),
                'predicted': pred,
                'actual': bool(row['text_a_is_closer'])
            })
    
    # Calculate metrics
    y_true = label_df['text_a_is_closer'].values
    y_pred = np.array(predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate similarity statistics
    valid_scores = [s for s in similarity_scores if s['diff'] is not None]
    avg_sim_diff = np.mean([s['diff'] for s in valid_scores]) if valid_scores else 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_samples': len(y_true),
        'correct_predictions': int((y_pred == y_true).sum()),
        'embedding_dim': embeddings.shape[1],
        # Diagnostic information
        'missing_embeddings_count': len(missing_embeddings),
        'missing_embeddings': missing_embeddings[:10],  # First 10 for brevity
        'uncertain_predictions_count': len(uncertain_predictions),
        'uncertain_predictions': uncertain_predictions[:10],  # First 10
        'incorrect_predictions_count': len(incorrect_predictions),
        'incorrect_predictions': incorrect_predictions[:10],  # First 10
        'avg_similarity_diff': float(avg_sim_diff),
        'uncertainty_threshold': uncertainty_threshold
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
    
    # Print Track A diagnostic information
    if 'incorrect_predictions_count' in metrics and 'missing_embeddings_count' not in metrics:
        print(f"\n{'-'*60}")
        print("TRACK A DIAGNOSTICS")
        print(f"{'-'*60}")
        
        # Incorrect predictions
        print(f"\n✗ Incorrect Predictions: {metrics['incorrect_predictions_count']}")
        if metrics['incorrect_predictions_count'] > 0:
            print(f"\n   Sample failures (showing first 5):")
            for i, item in enumerate(metrics['incorrect_predictions'][:5], 1):
                print(f"\n   [{i}] Index {item['index']}")
                print(f"       Predicted: {'A' if item['predicted'] else 'B'}, Actual: {'A' if item['actual'] else 'B'}")
                if 'score' in item:
                    print(f"       Score: {item['score']:.4f}")
                print(f"       Anchor: {item['anchor']}")
                print(f"       Text A: {item['text_a']}")
                print(f"       Text B: {item['text_b']}")
            if metrics['incorrect_predictions_count'] > 5:
                print(f"\n   ... and {metrics['incorrect_predictions_count'] - 5} more")
    
    # Print Track B diagnostic information
    if 'missing_embeddings_count' in metrics:
        print(f"\n{'-'*60}")
        print("TRACK B DIAGNOSTICS")
        print(f"{'-'*60}")
        
        # Missing embeddings
        if metrics['missing_embeddings_count'] > 0:
            print(f"\n⚠️  Missing Embeddings: {metrics['missing_embeddings_count']}")
            print(f"   Stories where embedding lookup failed:")
            for item in metrics['missing_embeddings'][:5]:  # Show first 5
                missing_fields = ', '.join([f[0] for f in item['missing_texts']])
                print(f"   - Index {item['index']}: Missing {missing_fields}")
            if metrics['missing_embeddings_count'] > 5:
                print(f"   ... and {metrics['missing_embeddings_count'] - 5} more")
        else:
            print(f"\n✓ No missing embeddings")
        
        # Uncertain predictions
        print(f"\n🤔 Uncertain Predictions: {metrics['uncertain_predictions_count']}")
        print(f"   (similarity difference < {metrics['uncertainty_threshold']:.3f})")
        if metrics['uncertain_predictions_count'] > 0:
            print(f"   Avg similarity diff: {metrics.get('avg_similarity_diff', 0):.4f}")
            print(f"\n   Top uncertain cases (showing first 3):")
            for i, item in enumerate(metrics['uncertain_predictions'][:3], 1):
                status = "✓" if item['predicted'] == item['actual'] else "✗"
                print(f"\n   [{i}] {status} Index {item['index']}")
                print(f"       Similarities: sim_a={item['sim_a']:.4f}, sim_b={item['sim_b']:.4f}, diff={item['diff']:.4f}")
                print(f"       Predicted: {'A' if item['predicted'] else 'B'}, Actual: {'A' if item['actual'] else 'B'}")
                print(f"       Anchor: {item['anchor']}")
                print(f"       Text A: {item['text_a']}")
                print(f"       Text B: {item['text_b']}")
        
        # Incorrect predictions
        print(f"\n✗ Incorrect Predictions: {metrics['incorrect_predictions_count']}")
        if metrics['incorrect_predictions_count'] > 0:
            print(f"\n   Sample failures (showing first 3):")
            for i, item in enumerate(metrics['incorrect_predictions'][:3], 1):
                print(f"\n   [{i}] Index {item['index']}")
                print(f"       Similarities: sim_a={item['sim_a']:.4f}, sim_b={item['sim_b']:.4f}, diff={item['diff']:.4f}")
                print(f"       Predicted: {'A' if item['predicted'] else 'B'}, Actual: {'A' if item['actual'] else 'B'}")
                print(f"       Anchor: {item['anchor']}")
                print(f"       Text A: {item['text_a']}")
                print(f"       Text B: {item['text_b']}")
            if metrics['incorrect_predictions_count'] > 3:
                print(f"\n   ... and {metrics['incorrect_predictions_count'] - 3} more")


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
    """Main evaluation pipeline - Google Colab only."""
    import argparse
    
    # Check if running in Colab
    if not is_colab():
        print("⚠️  This script is designed for Google Colab only!")
        print("Evaluation must be run in Colab where models and data are stored.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Evaluate Track A and Track B models")
    parser.add_argument(
        '--track',
        type=str,
        choices=['a', 'b', 'both'],
        default='both',
        help='Which track to evaluate'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATION - GOOGLE COLAB")
    print("="*60)
    
    # Setup Colab environment
    colab_paths = setup_colab_environment()
    
    # Load config with Drive paths
    config = load_config(colab_paths['config_path'])
    config = update_config_for_colab(config, colab_paths)
    
    # Get paths from config
    labels_path = config['data']['dev_track_a']
    track_b_path = config['data']['dev_track_b']
    predictions_path = Path(config['data']['output_dir']) / 'track_a.jsonl'
    embeddings_path = Path(config['data']['output_dir']) / 'track_b.npy'
    results_path = Path(config['data']['output_dir']) / 'evaluation_results.json'
    
    results = {}
    
    # Evaluate Track A
    if args.track in ['a', 'both']:
        if predictions_path.exists():
            print(f"\nEvaluating Track A predictions...")
            print(f"  Predictions: {predictions_path}")
            print(f"  Labels: {labels_path}")
            track_a_metrics = evaluate_track_a(str(predictions_path), labels_path)
            print_metrics(track_a_metrics, "Track A: Cross-Encoder Results")
            results['track_a'] = track_a_metrics
        else:
            print(f"\n⚠ Track A predictions not found at: {predictions_path}")
            print("  Run: !python track_a.py first")
    
    # Evaluate Track B
    if args.track in ['b', 'both']:
        if embeddings_path.exists():
            print(f"\nEvaluating Track B embeddings...")
            print(f"  Embeddings: {embeddings_path}")
            print(f"  Track B data: {track_b_path}")
            print(f"  Labels: {labels_path}")
            track_b_metrics = evaluate_track_b_with_track_a(
                str(embeddings_path), 
                labels_path,
                track_b_path
            )
            print_metrics(track_b_metrics, "Track B: Bi-Encoder Results")
            results['track_b'] = track_b_metrics
        else:
            print(f"\n⚠ Track B embeddings not found at: {embeddings_path}")
            print("  Run: !python track_b.py first")
    
    # Save results to Drive
    if results:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Results saved to: {results_path}")
        print(f"{'='*60}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

