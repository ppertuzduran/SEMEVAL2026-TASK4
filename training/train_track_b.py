"""
Training script for Track B: Bi-encoder (embedding model).

Google Colab training script.

This script implements recommendations from APPROACH.md:
- Multi-phase training curriculum (section 4.2):
  * Phase 1: Pairwise Softmax Loss (direct metric optimization)
  * Phase 2: TripletLoss (margin-based metric learning)
  * Phase 3: MultipleNegativesRankingLoss (contrastive learning)
- Temperature scaling for gradient sharpening (section 4.1.2)
- Support for BGE-base and BGE-large models (section 4.1)
- L2 normalization of embeddings (section 4.1.2)
- Mixed precision training (FP16)
- Saves models to Google Drive
"""

import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SequentialEvaluator
from tqdm import tqdm
import pandas as pd
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from colab_utils import is_colab, setup_colab_environment, update_config_for_colab, install_colab_dependencies, print_gpu_info


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_triplets(path: str) -> list:
    """Load triplet dataset."""
    triplets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            triplets.append(json.loads(line))
    return triplets


def load_pairs(path: str) -> list:
    """Load pairwise dataset."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def create_triplet_examples(triplets: list) -> list:
    """Convert triplet dicts to InputExample objects."""
    examples = []
    for t in triplets:
        examples.append(InputExample(
            texts=[t['anchor'], t['positive'], t['negative']]
        ))
    return examples


def create_pair_examples(pairs: list) -> list:
    """Convert pair dicts to InputExample objects for MultipleNegativesRankingLoss."""
    examples = []
    for p in pairs:
        # For MultipleNegativesRankingLoss, we use pairs of (anchor, positive)
        # The loss automatically treats other positives in the batch as negatives
        if p['label'] > 0.5:  # Only use positive pairs
            examples.append(InputExample(
                texts=[p['anchor'], p['candidate']]
            ))
    return examples


class PairwiseSoftmaxLoss(nn.Module):
    """
    NEW: Direct triple-wise pairwise softmax loss with temperature scaling.
    
    Directly optimizes the evaluation metric by:
    1. Encoding anchor, text_a, text_b independently
    2. Computing cosine similarities
    3. Applying temperature scaling
    4. Applying softmax cross-entropy
    
    This aligns training with the triple-wise evaluation.
    Temperature < 1.0 sharpens the distribution and improves gradient flow.
    """
    
    def __init__(self, model: SentenceTransformer, temperature: float = 1.0):
        super(PairwiseSoftmaxLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, sentence_features, labels):
        """
        sentence_features: List of dicts with 'input_ids', 'attention_mask', etc.
                          [anchor_features, text_a_features, text_b_features]
        labels: Tensor of shape (batch_size,) with 1 if A is closer, 0 if B is closer
        """
        # Encode all three texts independently
        embeddings = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        anchor_emb = embeddings[0]  # (batch, embed_dim)
        a_emb = embeddings[1]
        b_emb = embeddings[2]
        
        # Compute cosine similarities
        sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)  # (batch,)
        sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)  # (batch,)
        
        # Stack similarities and apply temperature scaling
        scores = torch.stack([sim_a, sim_b], dim=1)  # (batch, 2)
        scores = scores / self.temperature  # Temperature scaling: < 1.0 sharpens gradients
        
        # Convert labels: 1 → index 0 (A closer), 0 → index 1 (B closer)
        targets = (1 - labels).long()
        
        # Cross-entropy loss
        loss = self.loss_fn(scores, targets)
        return loss


class DistillDataset(Dataset):
    """Dataset for teacher → student distillation (cross-encoder to bi-encoder)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_triple_softmax_examples(data: list) -> list:
    """
    Create examples for PairwiseSoftmaxLoss.
    Each example contains (anchor, text_a, text_b, label).
    """
    examples = []
    for item in data:
        examples.append(InputExample(
            texts=[item['anchor'], item['text_a'], item['text_b']],
            label=float(item['label'])
        ))
    return examples


def evaluate_on_track_a(model, dev_path: str, device: str) -> float:
    """
    Evaluate the model on Track A dev set.
    Returns accuracy based on cosine similarity comparison.
    """
    df = pd.read_json(dev_path, lines=True)
    
    # Collect all unique texts
    all_texts = set()
    for _, row in df.iterrows():
        all_texts.add(row['anchor_text'])
        all_texts.add(row['text_a'])
        all_texts.add(row['text_b'])
    
    # Encode all texts
    all_texts = list(all_texts)
    with torch.no_grad():
        embeddings = model.encode(
            all_texts, 
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
    
    # Create lookup
    text_to_embedding = {text: emb for text, emb in zip(all_texts, embeddings)}
    
    # Calculate predictions
    correct = 0
    for _, row in df.iterrows():
        anchor_emb = text_to_embedding[row['anchor_text']]
        a_emb = text_to_embedding[row['text_a']]
        b_emb = text_to_embedding[row['text_b']]
        
        sim_a = cos_sim(anchor_emb, a_emb).item()
        sim_b = cos_sim(anchor_emb, b_emb).item()
        
        predicted_a_closer = sim_a > sim_b
        if predicted_a_closer == row['text_a_is_closer']:
            correct += 1
    
    accuracy = correct / len(df)
    return accuracy


class CustomEvaluator:
    """Custom evaluator that tracks best model."""
    
    def __init__(self, dev_path: str, device: str, save_path: str):
        self.dev_path = dev_path
        self.device = device
        self.save_path = save_path
        self.best_accuracy = 0.0
    
    def __call__(self, model, output_path: str, epoch: int, steps: int) -> float:
        """Evaluate and save if best."""
        accuracy = evaluate_on_track_a(model, self.dev_path, self.device)
        print(f"\nEpoch {epoch}, Step {steps}: Accuracy = {accuracy:.4f}")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {self.save_path}")
            model.save(self.save_path)
        
        return accuracy


def main():
    """Main training pipeline - Google Colab only."""
    if not is_colab():
        print("⚠️  This script is designed for Google Colab only!")
        print("Please run in Google Colab with GPU enabled.")
        sys.exit(1)
    
    print("="*60)
    print("TRACK B TRAINING - GOOGLE COLAB")
    print("="*60)
    
    # Setup Colab environment
    colab_paths = setup_colab_environment()
    print_gpu_info()
    
    # Load and update config
    config = load_config(colab_paths['config_path'])
    config = update_config_for_colab(config, colab_paths)
    
    set_seed(config['seed'])
    
    # Setup device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model (shared backbone for both tracks)
    run_distill_only = config['track_b'].get('run_distill_only', False)
    if run_distill_only:
        print(f"\nLoading pre-trained model for distillation only: {config['track_b']['model_save_path']}")
        model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
    else:
        print(f"\nLoading base model: {config['track_b']['base_model']}")
        model = SentenceTransformer(config['track_b']['base_model'], device=device)
    # Shorten max sequence length to save VRAM (affects tokenizer inside ST)
    if 'max_seq_length' in config['track_b']:
        model.max_seq_length = config['track_b']['max_seq_length']
        print(f"Set SentenceTransformer max_seq_length={model.max_seq_length}")
    if config['track_b'].get('enable_gradient_checkpointing', True):
        try:
            model._first_module().auto_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory savings.")
        except Exception as e:
            print(f"Warning: could not enable gradient checkpointing: {e}")
    try:
        model._first_module().auto_model.config.use_cache = False
        print("Disabled transformer cache to reduce memory.")
    except Exception:
        pass
    
    # Load prepared data
    prepared_dir = Path(config['data']['prepared_data_dir'])
    print("\nLoading triplets...")
    triplets = load_triplets(prepared_dir / "triplets.jsonl")
    print(f"Loaded {len(triplets)} triplets")
    
    print("Loading pairs...")
    pairs = load_pairs(prepared_dir / "pairs.jsonl")
    print(f"Loaded {len(pairs)} pairs")
    
    # Load cross-encoder data for pairwise softmax loss
    print("Loading cross-encoder data for pairwise softmax...")
    cross_encoder_data = []
    with open(prepared_dir / "cross_encoder_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            cross_encoder_data.append(json.loads(line))
    print(f"Loaded {len(cross_encoder_data)} triple samples")
    
    # Load train/val split
    with open(prepared_dir / "train_val_split.json", 'r') as f:
        split = json.load(f)
    
    train_indices = set(split['train'])
    val_indices = set(split['val'])
    
    # Split data
    train_triplets = [t for i, t in enumerate(triplets) if i in train_indices]
    val_triplets = [t for i, t in enumerate(triplets) if i in val_indices]
    train_pairs = [p for i, p in enumerate(pairs) if i // 2 in train_indices]
    train_cross_encoder = [d for i, d in enumerate(cross_encoder_data) if i // 2 in train_indices]
    
    print(f"\nTrain triplets: {len(train_triplets)}")
    print(f"Val triplets: {len(val_triplets)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Train cross-encoder (for pairwise softmax): {len(train_cross_encoder)}")
    
    # Create InputExamples
    triplet_examples = create_triplet_examples(train_triplets)
    pair_examples = create_pair_examples(train_pairs)
    triple_softmax_examples = create_triple_softmax_examples(train_cross_encoder)
    
    # Create DataLoaders with per-phase batch sizes to avoid OOM on large models
    batch_size = config['track_b']['batch_size']
    batch_size_mnr = config['track_b'].get('batch_size_mnr', batch_size)
    batch_size_pairwise = config['track_b'].get('batch_size_pairwise', batch_size)
    batch_size_triplet = config['track_b'].get('batch_size_triplet', batch_size)

    triplet_loader = DataLoader(
        triplet_examples,
        batch_size=batch_size_triplet,
        shuffle=True
    )

    pair_loader = DataLoader(
        pair_examples,
        batch_size=batch_size_mnr,
        shuffle=True
    )

    triple_softmax_loader = DataLoader(
        triple_softmax_examples,
        batch_size=batch_size_pairwise,
        shuffle=True
    )
    
    # Setup losses
    triplet_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=config['track_b']['triplet_margin']
    )
    
    mnr_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    pairwise_softmax_loss = PairwiseSoftmaxLoss(
        model=model,
        temperature=config['track_b'].get('temperature', 1.0)
    )
    
    # Training parameters per phase
    warmup_steps = config['track_b']['warmup_steps']
    
    # Setup evaluator
    base_evaluator = CustomEvaluator(
        dev_path=config['data']['dev_track_a'],
        device=device,
        save_path=config['track_b']['model_save_path']
    )
    evaluator = SequentialEvaluator([base_evaluator])
    
    # Phase 1: MultipleNegativesRankingLoss (global structure)
    if not run_distill_only and config['track_b'].get('use_multiple_negatives_ranking', True) and len(pair_examples) > 0:
        print("\n" + "="*60)
        print("Phase 1: MultipleNegativesRankingLoss (global structure)...")
        print("="*60)
        model.fit(
            train_objectives=[(pair_loader, mnr_loss)],
            epochs=config['track_b']['epochs_mnr'],
            warmup_steps=warmup_steps,
            optimizer_params={'lr': config['track_b']['learning_rate']},
            weight_decay=config['track_b']['weight_decay'],
            evaluation_steps=config['track_b']['eval_steps'],
            evaluator=evaluator,
            output_path=config['track_b']['model_save_path'] + "_mnr",
            save_best_model=False,
            use_amp=config['track_b']['mixed_precision']
        )
        model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
        base_evaluator.best_accuracy = base_evaluator(model, "", 0, 0)
    
    # Phase 2: PairwiseSoftmaxLoss (metric-aligned)
    if not run_distill_only and config['track_b'].get('use_pairwise_softmax', True) and len(triple_softmax_examples) > 0:
        print("\n" + "="*60)
        print("Phase 2: PairwiseSoftmaxLoss (metric aligned)...")
        print("="*60)
        model.fit(
            train_objectives=[(triple_softmax_loader, pairwise_softmax_loss)],
            epochs=config['track_b']['epochs_pairwise'],
            warmup_steps=warmup_steps,
            optimizer_params={'lr': config['track_b']['learning_rate']},
            weight_decay=config['track_b']['weight_decay'],
            evaluation_steps=config['track_b']['eval_steps'],
            evaluator=evaluator,
            output_path=config['track_b']['model_save_path'] + "_pairwise",
            save_best_model=False,
            use_amp=config['track_b']['mixed_precision']
        )
        model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
        base_evaluator.best_accuracy = base_evaluator(model, "", 0, 0)
        
    # Phase 3: TripletLoss (fine-grained discrimination)
    if not run_distill_only and len(triplet_examples) > 0:
        print("\n" + "="*60)
        print("Phase 3: TripletLoss (fine-grained)...")
        print("="*60)
        model.fit(
            train_objectives=[(triplet_loader, triplet_loss)],
            epochs=config['track_b']['epochs_triplet'],
            warmup_steps=max(1, warmup_steps // 2),
            optimizer_params={'lr': config['track_b']['learning_rate'] / 2},
            weight_decay=config['track_b']['weight_decay'],
            evaluation_steps=config['track_b']['eval_steps'],
            evaluator=evaluator,
            output_path=config['track_b']['model_save_path'] + "_triplet",
            save_best_model=False,
            use_amp=config['track_b']['mixed_precision']
        )
        model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
        base_evaluator.best_accuracy = base_evaluator(model, "", 0, 0)

    # Optional Phase 4: Distillation from Track A cross-encoder (teacher → student)
    if run_distill_only or config['track_b'].get('distill_from_teacher', False):
        teacher_paths = config['track_b'].get('teacher_model_paths', [config['track_b'].get('teacher_model_path')])
        print(f"Checking teacher models: {teacher_paths}")
        valid_teachers = [p for p in teacher_paths if Path(p).exists()]
        if valid_teachers:
            print(f"✓ Found {len(valid_teachers)} teacher models")
        else:
            print("✗ No teacher models found")
        if valid_teachers:
            print("\n" + "="*60)
            print(f"Phase 4: Distillation from {len(valid_teachers)} Track A cross-encoders (ensemble teacher)...")
            print("="*60)
            # Load all teacher models
            teachers = []
            teacher_tokenizer = None
            for path in valid_teachers:
                tokenizer = AutoTokenizer.from_pretrained(path)
                teacher_model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
                teacher_model.eval()
                teachers.append(teacher_model)
                if teacher_tokenizer is None:
                    teacher_tokenizer = tokenizer

            distill_bs = config['track_b'].get('batch_size_distill', batch_size)
            distill_dataset = DistillDataset(train_cross_encoder)
            distill_loader = DataLoader(
                distill_dataset,
                batch_size=distill_bs,
                shuffle=True
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=config['track_b']['learning_rate'])
            kl_loss = nn.KLDivLoss(reduction='batchmean')
            ce_loss = nn.CrossEntropyLoss()
            model.train()

            for epoch in range(1):
                pbar = tqdm(distill_loader, desc="Distill (A→B)")
                for batch in pbar:
                    anchor = batch['anchor']
                    text_a = batch['text_a']
                    text_b = batch['text_b']
                    labels = batch['label'].to(device, dtype=torch.float)

                    # Teacher logits (ensemble average)
                    with torch.no_grad():
                        inputs_a = teacher_tokenizer(
                            [f"{a} {teacher_tokenizer.sep_token} {b}" for a, b in zip(anchor, text_a)],
                            max_length=512,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
                        inputs_b = teacher_tokenizer(
                            [f"{a} {teacher_tokenizer.sep_token} {b}" for a, b in zip(anchor, text_b)],
                            max_length=512,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        inputs_b = {k: v.to(device) for k, v in inputs_b.items()}

                        # Average logits across all teacher models
                        t_a_list = []
                        t_b_list = []
                        for teacher_model in teachers:
                            t_a = teacher_model(**inputs_a).logits[:, 1]
                            t_b = teacher_model(**inputs_b).logits[:, 1]
                            t_a_list.append(t_a)
                            t_b_list.append(t_b)
                        t_a_avg = torch.stack(t_a_list).mean(dim=0)
                        t_b_avg = torch.stack(t_b_list).mean(dim=0)

                        teacher_scores = torch.stack([t_a_avg, t_b_avg], dim=1) / config['track_b']['distill_temperature']
                        teacher_probs = F.softmax(teacher_scores, dim=1)

                    # Student embeddings
                    features_anchor = model.tokenize(list(anchor))
                    features_a = model.tokenize(list(text_a))
                    features_b = model.tokenize(list(text_b))
                    # Move tokenized features to device
                    features_anchor = {k: v.to(device) for k, v in features_anchor.items()}
                    features_a = {k: v.to(device) for k, v in features_a.items()}
                    features_b = {k: v.to(device) for k, v in features_b.items()}
                    anchor_emb = model(features_anchor)['sentence_embedding']
                    a_emb = model(features_a)['sentence_embedding']
                    b_emb = model(features_b)['sentence_embedding']
                    sim_a = F.cosine_similarity(anchor_emb, a_emb, dim=1)
                    sim_b = F.cosine_similarity(anchor_emb, b_emb, dim=1)
                    student_scores = torch.stack([sim_a, sim_b], dim=1) / config['track_b']['distill_temperature']
                    student_log_probs = F.log_softmax(student_scores, dim=1)

                    # Loss = supervised CE + KL distill
                    targets = (1 - labels.long()).to(device)
                    loss_ce = ce_loss(student_scores, targets)
                    loss_kl = kl_loss(student_log_probs, teacher_probs)
                    loss = loss_ce + config['track_b']['distill_weight'] * loss_kl

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['track_b']['gradient_clip'])
                    optimizer.step()

                    pbar.set_postfix({"loss": loss.item()})

            # Save distilled model
            model.save(config['track_b']['model_save_path'])
            base_evaluator.best_accuracy = base_evaluator(model, "", 0, 0)
        else:
            print("Distillation skipped: teacher model not found.")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    final_model = SentenceTransformer(config['track_b']['model_save_path'], device=device)
    final_accuracy = evaluate_on_track_a(final_model, config['data']['dev_track_a'], device)
    print(f"\nFinal best model accuracy: {final_accuracy:.4f}")
    print(f"Model saved to: {config['track_b']['model_save_path']}")


if __name__ == "__main__":
    main()

