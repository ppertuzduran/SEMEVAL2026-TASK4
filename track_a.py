"""
Track A Inference: Cross-encoder for narrative similarity prediction.

Google Colab inference script.

This script:
- Runs ONLY in Google Colab with GPU
- Loads models from Google Drive
- Loads data from Google Drive
- Uses ensemble inference for best accuracy
- Saves predictions to Google Drive
"""

import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
from tqdm import tqdm

# Add training directory to path for colab_utils
sys.path.insert(0, '/content/project/training')
from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, print_gpu_info


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration if available."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


class CrossEncoderPredictor:
    """Cross-encoder model for narrative similarity prediction."""
    
    def __init__(self, model_path: str, device: str = None, verbose: bool = True, max_length: int = 512, temperature: float = 1.0):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use (cuda/cpu). Auto-detected if None.
            verbose: Print loading messages
            max_length: Max sequence length per pair
            temperature: Temperature for scaling logits (must match training config)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.temperature = temperature
        if verbose:
            print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        if verbose:
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
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs_a = {k: v.to(self.device) for k, v in inputs_a.items()}
        
        # Pair B: [CLS] anchor [SEP] text_b [SEP]
        text_b_pair = f"{anchor} {self.tokenizer.sep_token} {text_b}"
        inputs_b = self.tokenizer(
            text_b_pair,
            max_length=self.max_length,
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
        
        # Apply temperature scaling (must match training config)
        score_a = score_a / self.temperature
        score_b = score_b / self.temperature
        
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
        
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(df), batch_size), 
                     desc="Predicting", 
                     total=total_batches,
                     unit="batch",
):
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
                max_length=self.max_length,
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
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs_b = {k: v.to(self.device) for k, v in inputs_b.items()}
            
            # Get scores for pair B
            outputs_b = self.model(**inputs_b)
            scores_b = outputs_b.logits[:, 1].detach().cpu().numpy()  # Positive class logit
            
            # Apply temperature scaling (must match training config)
            scores_a = scores_a / self.temperature
            scores_b = scores_b / self.temperature
            
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
    
    def __init__(self, model_paths: List[str], device: str = None, method: str = "average", max_length: int = 512, temperature: float = 1.0):
        """
        Initialize ensemble with multiple models.
        
        Args:
            model_paths: Paths to fold models
            device: Device to use
            method: "average" (average logits) or "vote" (majority vote)
            max_length: Max sequence length per pair
            temperature: Temperature for scaling logits (must match training config)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.models = []
        
        print(f"Initializing ensemble with {len(model_paths)} models (method: {method})")
        for path in tqdm(model_paths, desc="Loading fold models", unit="model"):
            predictor = CrossEncoderPredictor(path, device=self.device, verbose=False, max_length=max_length, temperature=temperature)
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
        print(f"{'='*60}\n")
        
        for predictor in tqdm(self.models, desc="Processing fold models", unit="fold"):
            # Get raw scores for this model
            scores_a, scores_b = self._get_raw_scores(predictor, df, batch_size)
            all_scores_a.append(scores_a)
            all_scores_b.append(scores_b)
            
            # Clear memory between models
            if predictor.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Average scores across models
        print(f"\nAveraging scores from {len(self.models)} models...")
        # Stack tensors and average (all_scores_a/b are lists of tensors)
        avg_scores_a = torch.stack(all_scores_a).mean(dim=0)
        avg_scores_b = torch.stack(all_scores_b).mean(dim=0)
        
        # Predict based on averaged scores
        predictions = (avg_scores_a > avg_scores_b).cpu().numpy()
        print(f"✓ Ensemble inference complete!")
        return [bool(p) for p in predictions]
    
    def _predict_vote(self, df: pd.DataFrame, batch_size: int) -> List[bool]:
        """Majority vote on predictions."""
        all_predictions = []
        
        print(f"\n{'='*60}")
        print(f"Running ensemble inference (vote) on {len(df)} samples...")
        print(f"{'='*60}\n")
        
        for predictor in tqdm(self.models, desc="Processing fold models", unit="fold"):
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
        
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(df), batch_size), 
                     desc="  Scoring pairs", 
                     total=total_batches,
                     unit="batch",
                     leave=False,
):
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
                max_length=predictor.max_length,
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
                max_length=predictor.max_length,
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
    max_length = config['track_a'].get('max_length', 512)
    temperature = config['track_a'].get('temperature', 1.0)  # Get temperature from config
    
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
                method=ensemble_method,
                max_length=max_length,
                temperature=temperature
            )
        else:
            print("Warning: use_ensemble=true but no fold models found!")
            print("Falling back to single model...")
    
    # Single model
    if Path(model_path).exists():
        print(f"Using single model: {model_path}")
        return CrossEncoderPredictor(model_path, max_length=max_length, temperature=temperature)
    
    # Check for fold 0 model as fallback
    fold0_path = model_path + "_fold0"
    if Path(fold0_path).exists():
        print(f"Using fold 0 model: {fold0_path}")
        return CrossEncoderPredictor(fold0_path, max_length=max_length, temperature=temperature)
    
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
    print("TRACK A INFERENCE - GOOGLE COLAB")
    print("="*60)
    
    # Setup Colab environment
    colab_paths = setup_colab_environment()
    print_gpu_info()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load and update config with Colab paths
    print(f"\nLoading configuration...")
    config = load_config(colab_paths['config_path'])
    if config is None:
        print("❌ config.yaml not found!")
        sys.exit(1)
    
    config = update_config_for_colab(config, colab_paths)
    
    # Load data from Drive
    print(f"\nLoading data from Drive...")
    data_path = config['data']['dev_track_a']
    df = pd.read_json(data_path, lines=True)
    print(f"✓ Loaded {len(df)} samples from: {data_path}")
    
    # Get predictor from Drive models
    print(f"\nInitializing model from Drive...")
    print("This may take a moment for ensemble mode (loading 5 models)...\n")
    predictor = get_predictor(config)
    
    if predictor:
        # Use fine-tuned model with T4 GPU
        print(f"\nStarting inference...")
        inference_start = time.time()
        # T4 has 16GB VRAM, can use larger batch size than local RTX 4050
        batch_size = 8  # Increased from 4 for T4 GPU
        predictions = predictor.predict_batch(df, batch_size=batch_size)
        inference_time = time.time() - inference_start
        print(f"\n✓ Inference completed in {inference_time:.1f} seconds")
        print(f"  ({len(df)/inference_time:.1f} samples/sec)")
        df["predicted_text_a_is_closer"] = predictions
    else:
        # No model found
        print("\n❌ Fine-tuned model not found in Drive!")
        print(f"Expected model at: {config['track_a']['model_save_path']}")
        print("\nPlease train the model first:")
        print("  !python training/train_track_a.py")
        sys.exit(1)
    
    # Calculate accuracy
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    print(f"\nAccuracy: {accuracy:.4f}")

    # Prepare output
    df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
    del df["predicted_text_a_is_closer"]

    # Save results to Drive with LF line endings (Unix format for submission)
    print(f"\nSaving results to Drive...")
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "track_a.jsonl"

    # Ensure Unix line endings for submission compatibility
    jsonl_str = df.to_json(orient='records', lines=True)
    jsonl_str = jsonl_str.replace('\r\n', '\n').replace('\r', '\n')  # Normalize to LF

    with open(output_path, 'w', newline='\n') as f:
        f.write(jsonl_str)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved to: {output_path}")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
