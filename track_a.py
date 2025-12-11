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
    """Ensemble of multiple cross-encoder models."""
    
    def __init__(self, model_paths: List[str], device: str = None):
        """Initialize ensemble with multiple models."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        
        for path in model_paths:
            print(f"Loading model: {path}")
            predictor = CrossEncoderPredictor(path, device=self.device)
            self.models.append(predictor)
    
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 8) -> List[bool]:
        """Predict using ensemble (majority vote)."""
        all_predictions = []
        
        for i, predictor in enumerate(self.models):
            print(f"\nModel {i+1}/{len(self.models)} predictions:")
            preds = predictor.predict_batch(df, batch_size)
            all_predictions.append(preds)
            
            # Clear memory between models
            if predictor.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Majority vote
        all_predictions = torch.tensor(all_predictions, dtype=torch.float)
        ensemble_predictions = (all_predictions.mean(dim=0) > 0.5).cpu().numpy()
        
        return [bool(p) for p in ensemble_predictions]


def get_predictor(config: dict):
    """
    Get the appropriate predictor based on configuration.

    Returns:
        Predictor instance or None if no model found
    """
    model_path = config['track_a']['model_save_path']
    
    # Check if we should use ensemble
    if config['track_a'].get('use_ensemble', False):
        # Find all fold models
        model_dir = Path(model_path).parent
        fold_models = sorted(model_dir.glob(f"{Path(model_path).name}_fold*"))
        
        if fold_models:
            print(f"Using ensemble of {len(fold_models)} models")
            return EnsemblePredictor([str(p) for p in fold_models])
    
    # Single model
    if Path(model_path).exists():
        return CrossEncoderPredictor(model_path)
    
    # Check for fold 0 model as fallback
    fold0_path = model_path + "_fold0"
    if Path(fold0_path).exists():
        print("Using fold 0 model")
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
        df["predicted_text_a_is_closer"] = df.apply(
            lambda row: random.choice([True, False]), axis=1
        )
    elif baseline == "openai":
        print("OpenAI baseline not implemented in this version")
        print("Falling back to random baseline")
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )
    
    return df


def main():
    """Main inference pipeline."""
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    df = pd.read_json("data/dev_track_a.jsonl", lines=True)
    print(f"Loaded {len(df)} samples")
    
    # Load config
    config = load_config()
    
    # Get predictor
    predictor = None
    if config:
        predictor = get_predictor(config)
    
    if predictor:
        # Use fine-tuned model (reduced batch size for 6GB GPU)
        predictions = predictor.predict_batch(df, batch_size=4)
        df["predicted_text_a_is_closer"] = predictions
    else:
        # Fallback to baseline
        print("\nFine-tuned model not found!")
        print("Falling back to random baseline")
        df = predict_with_baseline(df, baseline="random")
    
    # Calculate accuracy
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    print(f"\nAccuracy: {accuracy:.4f}")

    # Prepare output
df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "track_a.jsonl"
    
    with open(output_path, 'w') as f:
        f.write(df.to_json(orient='records', lines=True))
    
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
