"""
Experiment runner for Track B improvements.

This script runs incremental experiments to test different improvements:
1. Baseline (current config)
2. + Better regularization (projection dropout + weight decay)
3. + Hyperparameter sweep
4. + Hard negative mining
5. + SimCSE (if still < target accuracy)

Each experiment builds on the previous one, keeping successful improvements.
Results are saved to experiments.json for analysis.
"""

import json
import sys
from pathlib import Path
import yaml
import subprocess
import shutil
from datetime import datetime


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
    print(f"Running training with updated config...")
    result = subprocess.run(
        [sys.executable, "training/train_track_b.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Training failed!")
        print(result.stderr)
        return {
            'name': exp_name,
            'status': 'failed',
            'error': result.stderr
        }
    
    # Extract accuracy from output
    output_lines = result.stdout.split('\n')
    final_accuracy = None
    
    for line in output_lines:
        if "Final best model accuracy:" in line:
            try:
                final_accuracy = float(line.split(':')[-1].strip())
            except:
                pass
    
    if final_accuracy is None:
        print(f"⚠️  Could not extract accuracy from output")
        final_accuracy = 0.0
    
    print(f"\n✓ Experiment complete: {exp_name}")
    print(f"  Accuracy: {final_accuracy:.4f}")
    
    return {
        'name': exp_name,
        'status': 'success',
        'accuracy': final_accuracy,
        'config': config['track_b'].copy(),
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Run incremental experiments."""
    print("="*70)
    print("TRACK B INCREMENTAL EXPERIMENTS")
    print("="*70)
    
    config_path = "config.yaml"
    results_path = "experiments_track_b.json"
    
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
    print(f"  projection_dropout: {base_config['track_b'].get('projection_dropout', 0.0)}")
    print(f"  projection_weight_decay: {base_config['track_b'].get('projection_weight_decay', base_config['track_b']['weight_decay'])}")
    print(f"  enable_hard_negatives: {base_config['track_b'].get('enable_hard_negatives', False)}")
    print(f"  enable_hyperparam_sweep: {base_config['track_b'].get('enable_hyperparam_sweep', False)}")
    print(f"  enable_simcse: {base_config['track_b'].get('enable_simcse', False)}")
    
    baseline_result = run_experiment("Baseline", base_config, config_path)
    experiments.append(baseline_result)
    
    baseline_accuracy = baseline_result.get('accuracy', 0.0)
    print(f"\n📊 Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Experiment 2: + Better Regularization
    config_reg = load_config(backup_path)
    config_reg['track_b']['projection_dropout'] = 0.15
    config_reg['track_b']['projection_weight_decay'] = 0.08
    
    print(f"\n{'='*70}")
    print("IMPROVEMENT 1: Better Regularization")
    print(f"{'='*70}")
    print("\nChanges:")
    print(f"  projection_dropout: 0.0 → 0.15")
    print(f"  projection_weight_decay: 0.01 → 0.08")
    
    reg_result = run_experiment("+ Regularization", config_reg, config_path)
    experiments.append(reg_result)
    
    reg_accuracy = reg_result.get('accuracy', 0.0)
    delta = reg_accuracy - baseline_accuracy
    print(f"\n📊 Accuracy: {reg_accuracy:.4f} (Δ={delta:+.4f})")
    
    # Keep if improvement
    if reg_accuracy >= baseline_accuracy:
        print("✓ Keeping regularization improvements")
        base_config = config_reg
        baseline_accuracy = reg_accuracy
    else:
        print("✗ Reverting regularization (no improvement)")
    
    # Experiment 3: + Hyperparameter Sweep
    config_sweep = base_config.copy()
    config_sweep['track_b']['enable_hyperparam_sweep'] = True
    
    print(f"\n{'='*70}")
    print("IMPROVEMENT 2: Hyperparameter Sweep")
    print(f"{'='*70}")
    print("\nChanges:")
    print(f"  enable_hyperparam_sweep: False → True")
    print(f"  Testing: {len(config_sweep['track_b']['temperature_grid'])} temps × {len(config_sweep['track_b']['margin_grid'])} margins")
    
    sweep_result = run_experiment("+ Hyperparam Sweep", config_sweep, config_path)
    experiments.append(sweep_result)
    
    sweep_accuracy = sweep_result.get('accuracy', 0.0)
    delta = sweep_accuracy - baseline_accuracy
    print(f"\n📊 Accuracy: {sweep_accuracy:.4f} (Δ={delta:+.4f})")
    
    if sweep_accuracy >= baseline_accuracy:
        print("✓ Keeping hyperparameter sweep")
        base_config = config_sweep
        baseline_accuracy = sweep_accuracy
    else:
        print("✗ Reverting sweep (no improvement)")
    
    # Experiment 4: + Hard Negative Mining
    config_hard = base_config.copy()
    config_hard['track_b']['enable_hard_negatives'] = True
    config_hard['track_b']['hard_negative_k'] = 5
    config_hard['track_b']['hard_negative_epochs'] = 2
    
    print(f"\n{'='*70}")
    print("IMPROVEMENT 3: Hard Negative Mining")
    print(f"{'='*70}")
    print("\nChanges:")
    print(f"  enable_hard_negatives: False → True")
    print(f"  hard_negative_k: 5")
    print(f"  hard_negative_epochs: 2")
    
    hard_result = run_experiment("+ Hard Negatives", config_hard, config_path)
    experiments.append(hard_result)
    
    hard_accuracy = hard_result.get('accuracy', 0.0)
    delta = hard_accuracy - baseline_accuracy
    print(f"\n📊 Accuracy: {hard_accuracy:.4f} (Δ={delta:+.4f})")
    
    if hard_accuracy >= baseline_accuracy:
        print("✓ Keeping hard negative mining")
        base_config = config_hard
        baseline_accuracy = hard_accuracy
    else:
        print("✗ Reverting hard negatives (no improvement)")
    
    # Experiment 5: + SimCSE
    config_simcse = base_config.copy()
    config_simcse['track_b']['enable_simcse'] = True
    
    print(f"\n{'='*70}")
    print("IMPROVEMENT 4: SimCSE Consistency Loss")
    print(f"{'='*70}")
    print("\nChanges:")
    print(f"  enable_simcse: False → True")
    print(f"  loss_weights.simcse: 0.1")
    
    simcse_result = run_experiment("+ SimCSE", config_simcse, config_path)
    experiments.append(simcse_result)
    
    simcse_accuracy = simcse_result.get('accuracy', 0.0)
    delta = simcse_accuracy - baseline_accuracy
    print(f"\n📊 Accuracy: {simcse_accuracy:.4f} (Δ={delta:+.4f})")
    
    if simcse_accuracy >= baseline_accuracy:
        print("✓ Keeping SimCSE")
        base_config = config_simcse
        baseline_accuracy = simcse_accuracy
    else:
        print("✗ Reverting SimCSE (no improvement)")
    
    # Save final best config
    save_config(base_config, config_path)
    
    # Save experiment results
    with open(results_path, 'w') as f:
        json.dump({
            'experiments': experiments,
            'best_config': base_config['track_b'],
            'final_accuracy': baseline_accuracy
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Experiment':<30} {'Accuracy':<12} {'Delta':<10} {'Status':<10}")
    print("-" * 70)
    
    for i, exp in enumerate(experiments):
        if exp['status'] == 'success':
            acc = exp['accuracy']
            delta = acc - experiments[0]['accuracy'] if i > 0 else 0.0
            status = "✓" if acc >= experiments[max(0, i-1)]['accuracy'] else "✗"
            print(f"{exp['name']:<30} {acc:<12.4f} {delta:+<10.4f} {status:<10}")
        else:
            print(f"{exp['name']:<30} {'FAILED':<12} {'-':<10} {'✗':<10}")
    
    print(f"\n{'='*70}")
    print(f"FINAL BEST ACCURACY: {baseline_accuracy:.4f}")
    print(f"{'='*70}")
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Best config saved to: {config_path}")
    print(f"✓ Original config backed up to: {backup_path}")


if __name__ == "__main__":
    main()
