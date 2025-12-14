"""
Lightweight experiment runner for Track B (Colab-friendly).

This version uses SMALLER GRIDS to reduce GPU usage:
- Regularization: 3 combinations (instead of 9)
- Hard negatives: 2 combinations (instead of 6)
- Runs faster and uses less GPU quota

Estimated time: 30-45 minutes (vs 60-90 minutes for full version)
"""

import json
import sys
from pathlib import Path
import yaml
import subprocess
import shutil
from datetime import datetime

# SMALLER hyperparameter search grids (Colab-friendly)
REG_DROPOUTS = [0.0, 0.1]              # Reduced from [0.0, 0.1, 0.2]
REG_WEIGHT_DECAYS = [0.05]             # Reduced from [0.03, 0.06, 0.1]
HARD_K = [5]                           # Reduced from [3, 5, 7]
HARD_EPOCHS = [1, 2]                   # Keep same


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
    print(f"📝 This may take 10-20 minutes depending on your GPU.")
    print(f"⏳ Training in progress...\n")
    
    result = subprocess.run(
        [sys.executable, "training/train_track_b.py"],
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
        if "Final best model accuracy:" in line:
            try:
                final_accuracy = float(line.split(':')[-1].strip())
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
        'config': config['track_b'].copy(),
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': elapsed_time
    }


def main():
    """Run incremental experiments (lightweight version)."""
    import time
    overall_start = time.time()
    
    print("="*70)
    print("TRACK B INCREMENTAL EXPERIMENTS (LIGHTWEIGHT)")
    print("="*70)
    print(f"\n🕒 Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 Expected total time: 30-45 minutes (reduced grids)")
    print(f"📦 Results will be saved to: experiments_track_b_lite.json\n")
    
    config_path = "config.yaml"
    results_path = "experiments_track_b_lite.json"
    
    # Load base config
    base_config = load_config(config_path)
    
    # Backup original config
    backup_path = "config.yaml.backup"
    shutil.copy(config_path, backup_path)
    print(f"\n✅ Backed up config to: {backup_path}")

    
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
    
    # Experiment 2: + Better Regularization (REDUCED Grid Search)
    print(f"\n{'='*70}")
    print("IMPROVEMENT 1: Better Regularization (Lightweight Grid)")
    print(f"{'='*70}")
    print(f"\nSearching over {len(REG_DROPOUTS)} dropouts × {len(REG_WEIGHT_DECAYS)} weight_decays = {len(REG_DROPOUTS) * len(REG_WEIGHT_DECAYS)} combinations")
    print(f"(Full version tests 9 combinations)")
    
    best_reg_result = None
    best_reg_config = None
    
    for d in REG_DROPOUTS:
        for wd in REG_WEIGHT_DECAYS:
            cfg = load_config(backup_path)
            cfg['track_b']['projection_dropout'] = d
            cfg['track_b']['projection_weight_decay'] = wd
            
            result = run_experiment(f'+ Reg (drop={d}, wd={wd})', cfg, config_path)
            experiments.append(result)
            
            if best_reg_result is None or result.get('accuracy', 0.0) > best_reg_result.get('accuracy', 0.0):
                best_reg_result = result
                best_reg_config = cfg
    
    reg_accuracy = best_reg_result.get('accuracy', 0.0)
    delta = reg_accuracy - baseline_accuracy
    print(f"\n📊 Best Regularization Accuracy: {reg_accuracy:.4f} (Δ={delta:+.4f})")
    
    # Keep if improvement
    if reg_accuracy >= baseline_accuracy:
        print("✓ Keeping best regularization settings")
        base_config = best_reg_config
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
    
    # Experiment 4: + Hard Negative Mining (REDUCED Grid Search)
    print(f"\n{'='*70}")
    print("IMPROVEMENT 3: Hard Negative Mining (Lightweight Grid)")
    print(f"{'='*70}")
    print(f"\nSearching over {len(HARD_K)} k values × {len(HARD_EPOCHS)} epochs = {len(HARD_K) * len(HARD_EPOCHS)} combinations")
    print(f"(Full version tests 6 combinations)")
    
    best_hard_result = None
    best_hard_config = None
    
    for k in HARD_K:
        for e in HARD_EPOCHS:
            cfg = load_config(backup_path)
            # Apply all previous improvements
            if base_config != load_config(backup_path):
                cfg = base_config.copy()
            cfg['track_b']['enable_hard_negatives'] = True
            cfg['track_b']['hard_negative_k'] = k
            cfg['track_b']['hard_negative_epochs'] = e
            
            result = run_experiment(f'+ HardNeg (k={k}, e={e})', cfg, config_path)
            experiments.append(result)
            
            if best_hard_result is None or result.get('accuracy', 0.0) > best_hard_result.get('accuracy', 0.0):
                best_hard_result = result
                best_hard_config = cfg
    
    hard_accuracy = best_hard_result.get('accuracy', 0.0)
    delta = hard_accuracy - baseline_accuracy
    print(f"\n📊 Best Hard Negative Accuracy: {hard_accuracy:.4f} (Δ={delta:+.4f})")
    
    if hard_accuracy >= baseline_accuracy:
        print("✓ Keeping best hard negative settings")
        base_config = best_hard_config
        baseline_accuracy = hard_accuracy
    else:
        print("✗ Reverting hard negatives (no improvement)")
    
    # SKIP SimCSE to save GPU quota
    print(f"\n{'='*70}")
    print("SKIPPING: SimCSE (to save GPU quota)")
    print(f"{'='*70}")
    print("\nSimCSE experiment skipped in lightweight version.")
    print("If needed, enable manually in config.yaml: enable_simcse: true")
    
    # Save final best config
    save_config(base_config, config_path)
    
    # Save experiment results
    overall_elapsed = time.time() - overall_start
    overall_minutes = int(overall_elapsed // 60)
    overall_seconds = int(overall_elapsed % 60)
    
    with open(results_path, 'w') as f:
        json.dump({
            'experiments': experiments,
            'best_config': base_config['track_b'],
            'final_accuracy': baseline_accuracy,
            'total_duration_seconds': overall_elapsed,
            'version': 'lightweight'
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY (LIGHTWEIGHT)")
    print(f"{'='*70}\n")
    
    print(f"{'Experiment':<30} {'Accuracy':<12} {'Delta':<10} {'Duration':<12} {'Status':<10}")
    print("-" * 80)
    
    for i, exp in enumerate(experiments):
        if exp['status'] == 'success':
            acc = exp['accuracy']
            delta = acc - experiments[0]['accuracy'] if i > 0 else 0.0
            duration = exp.get('duration_seconds', 0)
            dur_min = int(duration // 60)
            dur_sec = int(duration % 60)
            status = "✓" if acc >= experiments[max(0, i-1)]['accuracy'] else "✗"
            print(f"{exp['name']:<30} {acc:<12.4f} {delta:+<10.4f} {dur_min}m {dur_sec}s{'':<6} {status:<10}")
        else:
            print(f"{exp['name']:<30} {'FAILED':<12} {'-':<10} {'-':<12} {'✗':<10}")
    
    print(f"\n{'='*70}")
    print(f"FINAL BEST ACCURACY: {baseline_accuracy:.4f}")
    print(f"TOTAL TIME: {overall_minutes}m {overall_seconds}s")
    print(f"{'='*70}")
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Best config saved to: {config_path}")
    print(f"✅ Original config backed up to: {backup_path}")
    print(f"\n💡 TIP: For full grid search, use run_track_b_experiments.py")


if __name__ == "__main__":
    main()
