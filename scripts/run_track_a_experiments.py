"""
Experiment runner for Track A improvements (v10).

This script runs incremental experiments to test different hyperparameters:
1. Baseline (current config)
2. + Distillation sweep
3. + Head size × dropout sweep
4. + Learning rate sweep
5. + Optional joint fine-tuning (unfreeze Track B)

Each experiment builds on the previous one, keeping successful improvements.
Results are saved to experiments_track_a.json for analysis.
"""

import json
import sys
from pathlib import Path
import yaml
import subprocess
import shutil
from datetime import datetime

# Hyperparameter search grids (v10)
HIDDEN_DIMS = [512, 1024]
DROPOUTS = [0.1, 0.2, 0.3]
LEARNING_RATES = [5e-5, 1e-4, 2e-4]
DISTILL_WEIGHTS = [0.0, 0.3, 0.5]
FREEZE_OPTIONS = [True, False]


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str = "config.yaml"):
    """Save configuration."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_experiment(exp_name: str, config: dict, config_path: str) -> dict:
    """
    Run a single experiment.
    
    Returns:
        dict with experiment results
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}\n")
    
    # Save config for this experiment
    save_config(config, config_path)
    
    # Run training
    import time
    start_time = time.time()
    print(f"🚀 Starting training at {datetime.now().strftime('%H:%M:%S')}...")
    print(f"📝 This may take 5-15 minutes depending on your GPU.")
    print(f"⏳ Training in progress...\n")
    
    result = subprocess.run(
        [sys.executable, "training/train_track_a.py"],
        capture_output=True,
        text=True
    )
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    if result.returncode != 0:
        print(f"\n❌ Training failed after {minutes}m {seconds}s!")
        print(result.stderr)
        return {
            'name': exp_name,
            'status': 'failed',
            'error': result.stderr,
            'duration_seconds': elapsed_time
        }
    
    # Extract accuracy from output
    output_lines = result.stdout.split('\n')
    final_accuracy = None
    
    for line in output_lines:
        if "Mean Accuracy:" in line or "Final validation accuracy:" in line:
            try:
                # Extract number from "Mean Accuracy: 0.9500 ± 0.0100" or "Final validation accuracy: 0.9500"
                parts = line.split(':')[-1].strip().split()
                final_accuracy = float(parts[0])
            except:
                pass
    
    if final_accuracy is None:
        print(f"\n⚠️  Could not extract accuracy from output")
        final_accuracy = 0.0
    
    print(f"\n✅ Experiment complete: {exp_name}")
    print(f"   Accuracy: {final_accuracy:.4f}")
    print(f"   Duration: {minutes}m {seconds}s")
    
    return {
        'name': exp_name,
        'status': 'success',
        'accuracy': final_accuracy,
        'config': config['track_a'].copy(),
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': elapsed_time
    }


def main():
    """Run incremental experiments for Track A."""
    print("="*70)
    print("TRACK A INCREMENTAL EXPERIMENTS")
    print("="*70)
    
    config_path = "config.yaml"
    results_path = "experiments_track_a.json"
    
    # Load base config
    base_config = load_config(config_path)
    
    # Backup original config
    backup_path = "config.yaml.backup"
    shutil.copy(config_path, backup_path)
    print(f"\n✓ Backed up config to: {backup_path}")
    
    experiments = []
    
    # Experiment 1: Baseline (current settings)
    print(f"\n{'='*70}")
    print("BASELINE: Current configuration")
    print(f"{'='*70}")
    print("\nSettings:")
    print(f"  mlp_hidden_dim: {base_config['track_a'].get('mlp_hidden_dim', 512)}")
    print(f"  mlp_dropout: {base_config['track_a'].get('mlp_dropout', 0.2)}")
    print(f"  learning_rate: {base_config['track_a'].get('learning_rate', 1e-4)}")
    print(f"  distill_weight: {base_config['track_a'].get('distill_weight', 0.0)}")
    print(f"  freeze_track_b: {base_config['track_a'].get('freeze_track_b', True)}")
    
    baseline_result = run_experiment("Baseline", base_config, config_path)
    experiments.append(baseline_result)
    
    baseline_accuracy = baseline_result.get('accuracy', 0.0)
    print(f"\n📊 Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Experiment 2: Distillation Sweep
    print(f"\n{'='*70}")
    print("IMPROVEMENT 1: Distillation Weight Sweep")
    print(f"{'='*70}")
    print(f"\nSearching over {len(DISTILL_WEIGHTS)} distillation weights")
    
    best_distill_result = None
    best_distill_config = None
    
    for dw in DISTILL_WEIGHTS:
        cfg = load_config(backup_path)
        cfg['track_a']['distill_weight'] = dw
        cfg['track_a']['distill_temperature'] = 2.0
        
        result = run_experiment(f'+ Distill (weight={dw})', cfg, config_path)
        experiments.append(result)
        
        if best_distill_result is None or result.get('accuracy', 0.0) > best_distill_result.get('accuracy', 0.0):
            best_distill_result = result
            best_distill_config = cfg
    
    distill_accuracy = best_distill_result.get('accuracy', 0.0)
    delta = distill_accuracy - baseline_accuracy
    print(f"\n📊 Best Distillation Accuracy: {distill_accuracy:.4f} (Δ={delta:+.4f})")
    
    if distill_accuracy >= baseline_accuracy:
        print("✓ Keeping best distillation settings")
        base_config = best_distill_config
        baseline_accuracy = distill_accuracy
    else:
        print("✗ Reverting distillation (no improvement)")
    
    # Experiment 3: Head Size × Dropout Sweep
    print(f"\n{'='*70}")
    print("IMPROVEMENT 2: Head Size × Dropout Sweep")
    print(f"{'='*70}")
    print(f"\nSearching over {len(HIDDEN_DIMS)} hidden dims × {len(DROPOUTS)} dropouts = {len(HIDDEN_DIMS) * len(DROPOUTS)} combinations")
    
    best_head_result = None
    best_head_config = None
    
    for hd in HIDDEN_DIMS:
        for d in DROPOUTS:
            cfg = base_config.copy()
            cfg['track_a']['mlp_hidden_dim'] = hd
            cfg['track_a']['mlp_dropout'] = d
            
            result = run_experiment(f'+ Head (dim={hd}, drop={d})', cfg, config_path)
            experiments.append(result)
            
            if best_head_result is None or result.get('accuracy', 0.0) > best_head_result.get('accuracy', 0.0):
                best_head_result = result
                best_head_config = cfg
    
    head_accuracy = best_head_result.get('accuracy', 0.0)
    delta = head_accuracy - baseline_accuracy
    print(f"\n📊 Best Head Accuracy: {head_accuracy:.4f} (Δ={delta:+.4f})")
    
    if head_accuracy >= baseline_accuracy:
        print("✓ Keeping best head settings")
        base_config = best_head_config
        baseline_accuracy = head_accuracy
    else:
        print("✗ Reverting head settings (no improvement)")
    
    # Experiment 4: Learning Rate Sweep
    print(f"\n{'='*70}")
    print("IMPROVEMENT 3: Learning Rate Sweep")
    print(f"{'='*70}")
    print(f"\nSearching over {len(LEARNING_RATES)} learning rates")
    
    best_lr_result = None
    best_lr_config = None
    
    for lr in LEARNING_RATES:
        cfg = base_config.copy()
        cfg['track_a']['learning_rate'] = lr
        
        result = run_experiment(f'+ LR (lr={lr})', cfg, config_path)
        experiments.append(result)
        
        if best_lr_result is None or result.get('accuracy', 0.0) > best_lr_result.get('accuracy', 0.0):
            best_lr_result = result
            best_lr_config = cfg
    
    lr_accuracy = best_lr_result.get('accuracy', 0.0)
    delta = lr_accuracy - baseline_accuracy
    print(f"\n📊 Best LR Accuracy: {lr_accuracy:.4f} (Δ={delta:+.4f})")
    
    if lr_accuracy >= baseline_accuracy:
        print("✓ Keeping best learning rate")
        base_config = best_lr_config
        baseline_accuracy = lr_accuracy
    else:
        print("✗ Reverting learning rate (no improvement)")
    
    # Experiment 5: Optional Joint Fine-tuning
    print(f"\n{'='*70}")
    print("IMPROVEMENT 4: Joint Fine-tuning (Optional)")
    print(f"{'='*70}")
    print("\nTesting unfreezing Track B for light joint fine-tuning")
    
    cfg = base_config.copy()
    cfg['track_a']['freeze_track_b'] = False
    cfg['track_a']['epochs'] = 3  # Reduce epochs for joint training
    
    result = run_experiment('+ Joint Fine-tuning', cfg, config_path)
    experiments.append(result)
    
    joint_accuracy = result.get('accuracy', 0.0)
    delta = joint_accuracy - baseline_accuracy
    print(f"\n📊 Joint Fine-tuning Accuracy: {joint_accuracy:.4f} (Δ={delta:+.4f})")
    
    if joint_accuracy >= baseline_accuracy:
        print("✓ Keeping joint fine-tuning")
        base_config = cfg
        baseline_accuracy = joint_accuracy
    else:
        print("✗ Reverting to frozen Track B (no improvement)")
        # Restore epochs if we're reverting
        base_config['track_a']['epochs'] = load_config(backup_path)['track_a'].get('epochs', 10)
    
    # Save final best config
    save_config(base_config, config_path)
    
    # Save experiment results
    with open(results_path, 'w') as f:
        json.dump({
            'experiments': experiments,
            'best_config': base_config['track_a'],
            'final_accuracy': baseline_accuracy
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Experiment':<40} {'Accuracy':<12} {'Delta':<10} {'Status':<10}")
    print("-" * 70)
    
    for i, exp in enumerate(experiments):
        if exp['status'] == 'success':
            acc = exp['accuracy']
            delta = acc - experiments[0]['accuracy'] if i > 0 else 0.0
            status = "✓" if acc >= experiments[max(0, i-1)]['accuracy'] else "✗"
            print(f"{exp['name']:<40} {acc:<12.4f} {delta:+<10.4f} {status:<10}")
        else:
            print(f"{exp['name']:<40} {'FAILED':<12} {'-':<10} {'✗':<10}")
    
    print(f"\n{'='*70}")
    print(f"FINAL BEST ACCURACY: {baseline_accuracy:.4f}")
    print(f"{'='*70}")
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Best config saved to: {config_path}")
    print(f"✓ Original config backed up to: {backup_path}")


if __name__ == "__main__":
    main()
