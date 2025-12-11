"""
Track A system: Cross-encoder for narrative similarity prediction.

This script uses a fine-tuned cross-encoder model to determine which of two stories
is more narratively similar to an anchor story.
"""

import random
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration if available."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


class CrossEncoderPredictor:
    """Cross-encoder model for narrative similarity prediction."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    @torch.no_grad()
    def predict(self, anchor: str, text_a: str, text_b: str) -> bool:
        """
        Predict which text is closer to the anchor using pairwise scoring.
        
        NEW APPROACH: Score (anchor, A) and (anchor, B) separately, compare scores.
        
        Args:
            anchor: The anchor story
            text_a: First candidate story
            text_b: Second candidate story
            
        Returns:
            True if text_a is predicted to be closer, False otherwise
        """
        # Pairwise scoring: encode pairs separately
        # Pair A: [CLS] anchor [SEP] text_a [SEP]
        text_a_pair = f"{anchor} {self.tokenizer.sep_token} {text_a}"
        inputs_a = self.tokenizer(
            text_a_pair,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs_a = {k: v.to(self.device) for k, v in inputs_a.items()}
        
        # Pair B: [CLS] anchor [SEP] text_b [SEP]
        text_b_pair = f"{anchor} {self.tokenizer.sep_token} {text_b}"
        inputs_b = self.tokenizer(
            text_b_pair,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs_b = {k: v.to(self.device) for k, v in inputs_b.items()}
        
        # Get scores (use positive class logit as score)
        outputs_a = self.model(**inputs_a)
        score_a = outputs_a.logits[0, 1].item()
        
        outputs_b = self.model(**inputs_b)
        score_b = outputs_b.logits[0, 1].item()
        
        # Predict: A is closer if score_a > score_b
        return score_a > score_b
    
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 4) -> List[bool]:
        """
        Predict for a batch of samples using pairwise scoring.
        
        NEW APPROACH: Process (anchor, A) and (anchor, B) pairs separately.
        
        Args:
            df: DataFrame with columns 'anchor_text', 'text_a', 'text_b'
            batch_size: Batch size for inference (note: processes 2x pairs internally)
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Predicting"):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare pair A texts: (anchor, text_a)
            texts_a = []
            texts_b = []
            for _, row in batch.iterrows():
                text_a = f"{row['anchor_text']} {self.tokenizer.sep_token} {row['text_a']}"
                text_b = f"{row['anchor_text']} {self.tokenizer.sep_token} {row['text_b']}"
                texts_a.append(text_a)
                texts_b.append(text_b)
            
            # Tokenize batch A
            inputs_a = self.tokenizer(
                texts_a,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs_a = {k: v.to(self.device) for k, v in inputs_a.items()}
            
            # Get scores for pair A
            outputs_a = self.model(**inputs_a)
            scores_a = outputs_a.logits[:, 1].detach().cpu().numpy()  # Positive class logit
            
            # Clear memory
            if self.device == 'cuda':
                del inputs_a, outputs_a
                torch.cuda.empty_cache()
            
            # Tokenize batch B
            inputs_b = self.tokenizer(
                texts_b,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs_b = {k: v.to(self.device) for k, v in inputs_b.items()}
            
            # Get scores for pair B
            outputs_b = self.model(**inputs_b)
            scores_b = outputs_b.logits[:, 1].detach().cpu().numpy()  # Positive class logit
            
            # Predict: A is closer if score_a > score_b
            batch_predictions = scores_a > scores_b
            predictions.extend([bool(p) for p in batch_predictions])
            
            # Clear CUDA cache to free memory
            if self.device == 'cuda':
                del inputs_b, outputs_b, scores_a, scores_b
                torch.cuda.empty_cache()
        
        return predictions


class EnsemblePredictor:
    """
    Ensemble of multiple cross-encoder models.
    
    As recommended in APPROACH.md section 3.4:
    - Loads all k-fold models
    - Aggregates predictions via averaging or voting
    - Typically yields 1-3% accuracy improvement
    """
    
    def __init__(self, model_paths: List[str], device: str = None, method: str = "average"):
        """
        Initialize ensemble with multiple models.
        
        Args:
            model_paths: Paths to fold models
            device: Device to use
            method: "average" (average logits) or "vote" (majority vote)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.models = []
        
        print(f"Initializing ensemble with {len(model_paths)} models (method: {method})")
        for i, path in enumerate(model_paths):
            print(f"  Loading fold {i}: {path}")
            predictor = CrossEncoderPredictor(path, device=self.device)
            self.models.append(predictor)
    
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 4) -> List[bool]:
        """
        Predict using ensemble.
        
        Method options:
        - "average": Average logits across models (recommended)
        - "vote": Majority vote on predictions
        """
        if self.method == "average":
            return self._predict_average(df, batch_size)
        else:
            return self._predict_vote(df, batch_size)
    
    def _predict_average(self, df: pd.DataFrame, batch_size: int) -> List[bool]:
        """Average logits across models (better calibration)."""
        all_scores_a = []
        all_scores_b = []
        
        print(f"\n{'='*60}")
        print(f"Running ensemble inference on {len(df)} samples...")
        print(f"{'='*60}")
        
        for i, predictor in enumerate(self.models):
            print(f"\n[{i+1}/{len(self.models)}] Processing fold {i} model...")
            # Get raw scores for this model
            scores_a, scores_b = self._get_raw_scores(predictor, df, batch_size)
            all_scores_a.append(scores_a)
            all_scores_b.append(scores_b)
            
            # Clear memory between models
            if predictor.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Average scores across models
        print(f"\nAveraging scores from {len(self.models)} models...")
        avg_scores_a = torch.tensor(all_scores_a).mean(dim=0)
        avg_scores_b = torch.tensor(all_scores_b).mean(dim=0)
        
        # Predict based on averaged scores
        predictions = (avg_scores_a > avg_scores_b).cpu().numpy()
        print(f"✓ Ensemble inference complete!")
        return [bool(p) for p in predictions]
    
    def _predict_vote(self, df: pd.DataFrame, batch_size: int) -> List[bool]:
        """Majority vote on predictions."""
        all_predictions = []
        
        print(f"\n{'='*60}")
        print(f"Running ensemble inference (vote) on {len(df)} samples...")
        print(f"{'='*60}")
        
        for i, predictor in enumerate(self.models):
            print(f"\n[{i+1}/{len(self.models)}] Processing fold {i} model...")
            preds = predictor.predict_batch(df, batch_size)
            all_predictions.append(preds)
            
            # Clear memory between models
            if predictor.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Majority vote
        print(f"\nComputing majority vote from {len(self.models)} models...")
        all_predictions = torch.tensor(all_predictions, dtype=torch.float)
        ensemble_predictions = (all_predictions.mean(dim=0) > 0.5).cpu().numpy()
        print(f"✓ Ensemble inference complete!")
        
        return [bool(p) for p in ensemble_predictions]
    
    def _get_raw_scores(self, predictor: CrossEncoderPredictor, df: pd.DataFrame, batch_size: int) -> tuple:
        """Get raw scores (logits) from a predictor."""
        scores_a_list = []
        scores_b_list = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="  Processing batches", leave=False):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare pair texts
            texts_a = []
            texts_b = []
            for _, row in batch.iterrows():
                text_a = f"{row['anchor_text']} {predictor.tokenizer.sep_token} {row['text_a']}"
                text_b = f"{row['anchor_text']} {predictor.tokenizer.sep_token} {row['text_b']}"
                texts_a.append(text_a)
                texts_b.append(text_b)
            
            # Tokenize and score batch A
            inputs_a = predictor.tokenizer(
                texts_a,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs_a = {k: v.to(predictor.device) for k, v in inputs_a.items()}
            
            with torch.no_grad():
                outputs_a = predictor.model(**inputs_a)
                scores_a = outputs_a.logits[:, 1].cpu()  # Positive class logit
            
            # Clear memory
            if predictor.device == 'cuda':
                del inputs_a, outputs_a
                torch.cuda.empty_cache()
            
            # Tokenize and score batch B
            inputs_b = predictor.tokenizer(
                texts_b,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs_b = {k: v.to(predictor.device) for k, v in inputs_b.items()}
            
            with torch.no_grad():
                outputs_b = predictor.model(**inputs_b)
                scores_b = outputs_b.logits[:, 1].cpu()  # Positive class logit
            
            # Clear memory
            if predictor.device == 'cuda':
                del inputs_b, outputs_b
                torch.cuda.empty_cache()
            
            scores_a_list.append(scores_a)
            scores_b_list.append(scores_b)
        
        return torch.cat(scores_a_list), torch.cat(scores_b_list)


def get_predictor(config: dict):
    """
    Get the appropriate predictor based on configuration.
    
    As per APPROACH.md section 3.4:
    - If use_ensemble=true, loads all k-fold models and ensembles them
    - Otherwise uses single model or fold 0 as fallback

    Returns:
        Predictor instance or None if no model found
    """
    model_path = config['track_a']['model_save_path']
    
    # Check if we should use ensemble (RECOMMENDED in APPROACH.md)
    if config['track_a'].get('use_ensemble', False):
        # Find all fold models
        model_dir = Path(model_path).parent
        fold_models = sorted(model_dir.glob(f"{Path(model_path).name}_fold*"))
        
        if fold_models:
            ensemble_method = config['track_a'].get('ensemble_method', 'average')
            print(f"\n{'='*60}")
            print(f"ENSEMBLE MODE: {len(fold_models)} models (method: {ensemble_method})")
            print(f"{'='*60}")
            return EnsemblePredictor(
                [str(p) for p in fold_models],
                method=ensemble_method
            )
        else:
            print("Warning: use_ensemble=true but no fold models found!")
            print("Falling back to single model...")
    
    # Single model
    if Path(model_path).exists():
        print(f"Using single model: {model_path}")
        return CrossEncoderPredictor(model_path)
    
    # Check for fold 0 model as fallback
    fold0_path = model_path + "_fold0"
    if Path(fold0_path).exists():
        print(f"Using fold 0 model: {fold0_path}")
        return CrossEncoderPredictor(fold0_path)
    
    return None


def predict_with_baseline(df: pd.DataFrame, baseline: str = "random") -> pd.DataFrame:
    """
    Fallback baseline prediction.
    
    Args:
        df: Input dataframe
        baseline: 'random' or 'openai'
        
    Returns:
        DataFrame with predictions
    """
    if baseline == "random":
        print("Using random baseline")
    elif baseline == "openai":
        print("OpenAI baseline not implemented in this version")
        print("Falling back to random baseline")
    
    # Apply random baseline
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )
    
    return df


def main():
    """Main inference pipeline."""
    import time
    import os
    start_time = time.time()
    
    print("="*60)
    print("TRACK A INFERENCE - Narrative Similarity Prediction")
    print("="*60)
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠ Running on CPU (slower)")
    
    # Load config first to get data paths
    config = load_config()
    
    # Determine data path (from config or default)
    if config and 'data' in config:
        track_a_path = config['data'].get('dev_track_a', 'data/dev_track_a.jsonl')
        output_dir = Path(config['data'].get('output_dir', 'output'))
    else:
        track_a_path = 'data/dev_track_a.jsonl'
        output_dir = Path('output')
    
    # Check if file exists, try alternative paths
    if not os.path.exists(track_a_path):
        print(f"\n⚠️  File not found: {track_a_path}")
        # Try Google Drive path
        alt_path = '/content/drive/MyDrive/narrative_similarity/data/dev_track_a.jsonl'
        if os.path.exists(alt_path):
            print(f"✓ Found file at: {alt_path}")
            track_a_path = alt_path
        else:
            print(f"❌ File not found at alternative path: {alt_path}")
            print("\nPlease ensure data files are in one of these locations:")
            print("  - data/dev_track_a.jsonl (local)")
            print("  - /content/drive/MyDrive/narrative_similarity/data/dev_track_a.jsonl (Colab)")
            print("\nOr run Cell 7 to update config.yaml with correct paths.")
            return
    
    # Load data
    print(f"\nLoading data from: {track_a_path}")
    df = pd.read_json(track_a_path, lines=True)
    print(f"✓ Loaded {len(df)} samples")
    
    # Get predictor
    print(f"\nInitializing model...")
    predictor = None
    if config:
        predictor = get_predictor(config)
    
    if predictor:
        # Use fine-tuned model (reduced batch size for 6GB GPU)
        print(f"\nStarting inference...")
        inference_start = time.time()
        predictions = predictor.predict_batch(df, batch_size=4)
        inference_time = time.time() - inference_start
        print(f"\n✓ Inference completed in {inference_time:.1f} seconds")
        print(f"  ({len(df)/inference_time:.1f} samples/sec)")
        df["predicted_text_a_is_closer"] = predictions
    else:
        # Fallback to baseline
        print("\n⚠ Fine-tuned model not found!")
        print("Falling back to random baseline")
        df = predict_with_baseline(df, baseline="random")
    
    # Calculate accuracy
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    print(f"\nAccuracy: {accuracy:.4f}")

    # Prepare output
    df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
    del df["predicted_text_a_is_closer"]

    # Save results
    print(f"\nSaving results...")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "track_a.jsonl"
    
    with open(output_path, 'w') as f:
        f.write(df.to_json(orient='records', lines=True))
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved to: {output_path}")
    print(f"✓ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
