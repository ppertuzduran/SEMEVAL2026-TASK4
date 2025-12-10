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
        Predict which text is closer to the anchor.
        
        Args:
            anchor: The anchor story
            text_a: First candidate story
            text_b: Second candidate story
            
        Returns:
            True if text_a is predicted to be closer, False otherwise
        """
        # Format: [CLS] anchor [SEP] text_a [SEP] text_b [SEP]
        text = f"{anchor} {self.tokenizer.sep_token} {text_a} {self.tokenizer.sep_token} {text_b}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        
        return bool(prediction)
    
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 8) -> List[bool]:
        """
        Predict for a batch of samples.
        
        Args:
            df: DataFrame with columns 'anchor_text', 'text_a', 'text_b'
            batch_size: Batch size for inference
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Predicting"):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare texts
            texts = []
            for _, row in batch.iterrows():
                text = f"{row['anchor_text']} {self.tokenizer.sep_token} {row['text_a']} {self.tokenizer.sep_token} {row['text_b']}"
                texts.append(text)
            
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            outputs = self.model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            predictions.extend([bool(p) for p in batch_predictions])
            
            # Clear CUDA cache to free memory
            if self.device == 'cuda':
                del inputs, outputs, logits, batch_predictions
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
