"""
Track A Inference: MLP Head over Track B embeddings (v11: Qwen3-Embedding compatible).

Google Colab inference script.

This script:
- Runs ONLY in Google Colab with GPU
- Loads Track B model and MLP head(s) from Google Drive (supports both BGE-large and Qwen3-Embedding)
- Loads data from Google Drive
- Uses MLP head ensemble for best accuracy
- Saves predictions to Google Drive
"""

import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import yaml
from tqdm import tqdm
import json

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


class MLPHead(nn.Module):
    """
    Lightweight MLP head for pairwise ranking over embeddings.
    
    Architecture matches training script.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        super(MLPHead, self).__init__()
        self.embedding_dim = embedding_dim
        input_dim = 4 * embedding_dim  # [e1, e2, |e1-e2|, e1⊙e2]
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def construct_pairwise_features(self, anchor_emb, candidate_emb):
        """Construct pairwise features: [e_anchor, e_cand, |e_anchor - e_cand|, e_anchor ⊙ e_cand]"""
        diff = torch.abs(anchor_emb - candidate_emb)
        prod = anchor_emb * candidate_emb
        features = torch.cat([anchor_emb, candidate_emb, diff, prod], dim=1)
        return features
    
    def forward(self, anchor_emb, candidate_emb):
        """Forward pass through MLP head."""
        features = self.construct_pairwise_features(anchor_emb, candidate_emb)
        score = self.mlp(features)
        return score.squeeze(-1)  # (batch,)


class MLPHeadPredictor:
    """MLP head predictor using Track B embeddings."""
    
    def __init__(
        self,
        track_b_model: SentenceTransformer,
        mlp_head: MLPHead,
        device: str = None,
        temperature: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            track_b_model: Track B bi-encoder model
            mlp_head: Trained MLP head
            device: Device to use (cuda/cpu)
            temperature: Temperature for scaling logits
            verbose: Print loading messages
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.track_b_model = track_b_model
        self.mlp_head = mlp_head.to(self.device)
        self.mlp_head.eval()
        
        if verbose:
            print(f"✓ MLP Head predictor loaded on device: {self.device}")
    
    @torch.no_grad()
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 8) -> List[bool]:
        """
        Predict for a batch of samples using MLP head over Track B embeddings.
        
        Args:
            df: DataFrame with columns 'anchor_text', 'text_a', 'text_b'
            batch_size: Batch size for inference
            
        Returns:
            List of predictions
        """
        predictions = []
        
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(df), batch_size), 
                     desc="Predicting", 
                     total=total_batches,
                     unit="batch"):
            batch = df.iloc[i:i+batch_size]
            
            # Get texts
            anchor_texts = batch['anchor_text'].tolist()
            text_a_list = batch['text_a'].tolist()
            text_b_list = batch['text_b'].tolist()
            
            # Generate embeddings with Track B
            anchor_emb = self.track_b_model.encode(
                anchor_texts, convert_to_tensor=True, device=self.device, show_progress_bar=False
            )
            a_emb = self.track_b_model.encode(
                text_a_list, convert_to_tensor=True, device=self.device, show_progress_bar=False
            )
            b_emb = self.track_b_model.encode(
                text_b_list, convert_to_tensor=True, device=self.device, show_progress_bar=False
            )
            
            # Get scores from MLP head
            score_a = self.mlp_head(anchor_emb, a_emb)
            score_b = self.mlp_head(anchor_emb, b_emb)
            
            # Apply temperature scaling
            score_a = score_a / self.temperature
            score_b = score_b / self.temperature
            
            # Predict: A is closer if score_a > score_b
            batch_predictions = score_a > score_b
            predictions.extend([bool(p) for p in batch_predictions.cpu().numpy()])
            
            # Clear CUDA cache to free memory
            if self.device == 'cuda':
                del anchor_emb, a_emb, b_emb, score_a, score_b
                torch.cuda.empty_cache()
        
        return predictions


class EnsemblePredictor:
    """
    Ensemble of multiple MLP heads.
    
    Loads all k-fold MLP heads and aggregates predictions via averaging.
    """
    
    def __init__(
        self,
        track_b_model: SentenceTransformer,
        mlp_head_paths: List[str],
        device: str = None,
        method: str = "average"
    ):
        """
        Initialize ensemble with multiple MLP heads.
        
        Args:
            track_b_model: Track B bi-encoder (shared across all heads)
            mlp_head_paths: Paths to fold MLP head directories
            device: Device to use
            method: "average" (average scores) or "vote" (majority vote)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.track_b_model = track_b_model
        self.predictors = []
        
        print(f"Initializing ensemble with {len(mlp_head_paths)} MLP heads (method: {method})")
        for path in tqdm(mlp_head_paths, desc="Loading fold models", unit="model"):
            # Load config
            config_path = Path(path) / "config.json"
            with open(config_path, 'r') as f:
                head_config = json.load(f)
            
            # Create MLP head
            mlp_head = MLPHead(
                embedding_dim=head_config['embedding_dim'],
                hidden_dim=head_config['mlp_hidden_dim'],
                dropout=head_config.get('mlp_dropout', 0.2)
            )
            
            # Load weights
            state_dict = torch.load(Path(path) / "mlp_head.pt", map_location=self.device)
            mlp_head.load_state_dict(state_dict)
            
            # Create predictor
            predictor = MLPHeadPredictor(
                track_b_model=self.track_b_model,
                mlp_head=mlp_head,
                device=self.device,
                temperature=head_config.get('temperature', 1.0),
                verbose=False
            )
            self.predictors.append(predictor)
        
        print(f"✓ Loaded {len(self.predictors)} MLP heads")
    
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 8) -> List[bool]:
        """
        Predict using ensemble.
        
        Method options:
        - "average": Average scores across models (recommended)
        - "vote": Majority vote on predictions
        """
        if self.method == "average":
            return self._predict_average(df, batch_size)
        else:
            return self._predict_vote(df, batch_size)
    
    def _predict_average(self, df: pd.DataFrame, batch_size: int) -> List[bool]:
        """Average scores across models (better calibration)."""
        all_scores_a = []
        all_scores_b = []
        
        print(f"\n{'='*60}")
        print(f"Running ensemble inference on {len(df)} samples...")
        print(f"{'='*60}\n")
        
        for predictor in tqdm(self.predictors, desc="Processing fold models", unit="fold"):
            # Get raw scores for this model
            scores_a, scores_b = self._get_raw_scores(predictor, df, batch_size)
            all_scores_a.append(scores_a)
            all_scores_b.append(scores_b)
            
            # Clear memory between models
            if predictor.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Average scores across models
        print(f"\nAveraging scores from {len(self.predictors)} models...")
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
        
        for predictor in tqdm(self.predictors, desc="Processing fold models", unit="fold"):
            preds = predictor.predict_batch(df, batch_size)
            all_predictions.append(preds)
            
            # Clear memory between models
            if predictor.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Majority vote
        print(f"\nComputing majority vote from {len(self.predictors)} models...")
        all_predictions = torch.tensor(all_predictions, dtype=torch.float)
        ensemble_predictions = (all_predictions.mean(dim=0) > 0.5).cpu().numpy()
        print(f"✓ Ensemble inference complete!")
        
        return [bool(p) for p in ensemble_predictions]
    
    def _get_raw_scores(self, predictor: MLPHeadPredictor, df: pd.DataFrame, batch_size: int) -> tuple:
        """Get raw scores from a predictor."""
        scores_a_list = []
        scores_b_list = []
        
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(df), batch_size), 
                     desc="  Scoring pairs", 
                     total=total_batches,
                     unit="batch",
                     leave=False):
            batch = df.iloc[i:i+batch_size]
            
            # Get texts
            anchor_texts = batch['anchor_text'].tolist()
            text_a_list = batch['text_a'].tolist()
            text_b_list = batch['text_b'].tolist()
            
            # Generate embeddings with Track B
            anchor_emb = predictor.track_b_model.encode(
                anchor_texts, convert_to_tensor=True, device=predictor.device, show_progress_bar=False
            )
            a_emb = predictor.track_b_model.encode(
                text_a_list, convert_to_tensor=True, device=predictor.device, show_progress_bar=False
            )
            b_emb = predictor.track_b_model.encode(
                text_b_list, convert_to_tensor=True, device=predictor.device, show_progress_bar=False
            )
            
            # Get scores from MLP head
            with torch.no_grad():
                score_a = predictor.mlp_head(anchor_emb, a_emb)
                score_b = predictor.mlp_head(anchor_emb, b_emb)
            
            # Apply temperature scaling
            score_a = score_a / predictor.temperature
            score_b = score_b / predictor.temperature
            
            scores_a_list.append(score_a.cpu())
            scores_b_list.append(score_b.cpu())
            
            # Clear memory
            if predictor.device == 'cuda':
                del anchor_emb, a_emb, b_emb, score_a, score_b
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        all_scores_a = torch.cat(scores_a_list)
        all_scores_b = torch.cat(scores_b_list)
        
        return all_scores_a, all_scores_b


def get_predictor(config: dict, track_b_model: SentenceTransformer):
    """
    Get the appropriate predictor based on configuration.
    
    Returns:
        Predictor instance or None if no model found
    """
    model_path = config['track_a']['model_save_path']
    
    # Check if we should use ensemble (RECOMMENDED)
    if config['track_a'].get('use_ensemble', False):
        # Find all fold models
        model_dir = Path(model_path).parent
        fold_models = sorted(model_dir.glob(f"{Path(model_path).name}_fold*"))
        
        if fold_models:
            ensemble_method = config['track_a'].get('ensemble_method', 'average')
            print(f"\n{'='*60}")
            print(f"ENSEMBLE MODE: {len(fold_models)} MLP heads (method: {ensemble_method})")
            print(f"{'='*60}")
            return EnsemblePredictor(
                track_b_model=track_b_model,
                mlp_head_paths=[str(p) for p in fold_models],
                method=ensemble_method
            )
        else:
            print("Warning: use_ensemble=true but no fold models found!")
            print("Falling back to single model...")
    
    # Single model
    if Path(model_path).exists():
        print(f"Using single MLP head: {model_path}")
        # Load config
        config_path = Path(model_path) / "config.json"
        with open(config_path, 'r') as f:
            head_config = json.load(f)
        
        # Create MLP head
        mlp_head = MLPHead(
            embedding_dim=head_config['embedding_dim'],
            hidden_dim=head_config['mlp_hidden_dim'],
            dropout=head_config.get('mlp_dropout', 0.2)
        )
        
        # Load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(Path(model_path) / "mlp_head.pt", map_location=device)
        mlp_head.load_state_dict(state_dict)
        
        return MLPHeadPredictor(
            track_b_model=track_b_model,
            mlp_head=mlp_head,
            temperature=head_config.get('temperature', 1.0)
        )
    
    # Check for fold 0 model as fallback
    fold0_path = model_path + "_fold0"
    if Path(fold0_path).exists():
        print(f"Using fold 0 MLP head: {fold0_path}")
        # Load config
        config_path = Path(fold0_path) / "config.json"
        with open(config_path, 'r') as f:
            head_config = json.load(f)
        
        # Create MLP head
        mlp_head = MLPHead(
            embedding_dim=head_config['embedding_dim'],
            hidden_dim=head_config['mlp_hidden_dim'],
            dropout=head_config.get('mlp_dropout', 0.2)
        )
        
        # Load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(Path(fold0_path) / "mlp_head.pt", map_location=device)
        mlp_head.load_state_dict(state_dict)
        
        return MLPHeadPredictor(
            track_b_model=track_b_model,
            mlp_head=mlp_head,
            temperature=head_config.get('temperature', 1.0)
        )
    
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
    print("TRACK A INFERENCE (V2: MLP HEAD) - GOOGLE COLAB")
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
    
    # Load Track B model first (required for MLP head)
    track_b_path = config['track_a']['track_b_model_path']
    if not Path(track_b_path).exists():
        print(f"❌ Track B model not found at: {track_b_path}")
        print("\nPlease train Track B first:")
        print("  !python training/train_track_b.py")
        sys.exit(1)
    
    print(f"\nLoading Track B model from Drive...")
    print(f"  Model: {track_b_path}")
    track_b_model = SentenceTransformer(track_b_path)
    print(f"✓ Track B model loaded")
    print(f"  Embedding dimension: {track_b_model.get_sentence_embedding_dimension()}")
    
    # Load data from Drive
    print(f"\nLoading data from Drive...")
    data_path = config['data']['dev_track_a']
    df = pd.read_json(data_path, lines=True)
    print(f"✓ Loaded {len(df)} samples from: {data_path}")
    
    # Get predictor from Drive models
    print(f"\nInitializing MLP head(s) from Drive...")
    print("This may take a moment for ensemble mode (loading 5 heads)...\n")
    predictor = get_predictor(config, track_b_model)
    
    if predictor:
        # Use MLP head with T4 GPU
        print(f"\nStarting inference...")
        inference_start = time.time()
        # T4 has 16GB VRAM, can use larger batch size
        batch_size = 16  # Larger than cross-encoder since MLP head is tiny
        predictions = predictor.predict_batch(df, batch_size=batch_size)
        inference_time = time.time() - inference_start
        print(f"\n✓ Inference completed in {inference_time:.1f} seconds")
        print(f"  ({len(df)/inference_time:.1f} samples/sec)")
        df["predicted_text_a_is_closer"] = predictions
    else:
        # No model found
        print("\n❌ MLP head not found in Drive!")
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
