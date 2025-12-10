"""
Helper script to run the entire training and inference pipeline.

Usage:
    python scripts/run_full_pipeline.py --stage all
    python scripts/run_full_pipeline.py --stage prepare
    python scripts/run_full_pipeline.py --stage train
    python scripts/run_full_pipeline.py --stage inference
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        sys.exit(1)


def check_prerequisites():
    """Check if required files and directories exist."""
    required_files = [
        "config.yaml",
        "data/dev_track_a.jsonl",
        "data/dev_track_b.jsonl",
        "training/prepare_data.py",
        "training/train_track_a.py",
        "training/train_track_b.py",
        "track_a.py",
        "track_b.py"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("⚠ Missing required files:")
        for file in missing:
            print(f"  - {file}")
        return False
    
    return True


def stage_prepare():
    """Run data preparation."""
    return run_command(
        [sys.executable, "training/prepare_data.py"],
        "Data Preparation"
    )


def stage_train_track_b():
    """Train Track B model."""
    return run_command(
        [sys.executable, "training/train_track_b.py"],
        "Track B Training"
    )


def stage_train_track_a():
    """Train Track A model."""
    return run_command(
        [sys.executable, "training/train_track_a.py"],
        "Track A Training"
    )


def stage_inference_track_b():
    """Run Track B inference."""
    return run_command(
        [sys.executable, "track_b.py"],
        "Track B Inference"
    )


def stage_inference_track_a():
    """Run Track A inference."""
    return run_command(
        [sys.executable, "track_a.py"],
        "Track A Inference"
    )


def stage_evaluate():
    """Run evaluation."""
    return run_command(
        [sys.executable, "scripts/eval_local.py", "--track", "both"],
        "Evaluation"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the full training and inference pipeline"
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', 'prepare', 'train', 'train_a', 'train_b', 'inference', 'eval'],
        default='all',
        help='Which stage to run'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip prerequisite checks'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_check:
        print("Checking prerequisites...")
        if not check_prerequisites():
            print("\n✗ Prerequisites check failed!")
            print("Please ensure all required files are present.")
            sys.exit(1)
        print("✓ Prerequisites check passed!\n")
    
    success = True
    
    if args.stage == 'all':
        # Run full pipeline
        print("\n" + "="*60)
        print("FULL PIPELINE EXECUTION")
        print("="*60)
        
        stages = [
            ("Prepare Data", stage_prepare),
            ("Train Track B", stage_train_track_b),
            ("Train Track A", stage_train_track_a),
            ("Track B Inference", stage_inference_track_b),
            ("Track A Inference", stage_inference_track_a),
            ("Evaluation", stage_evaluate)
        ]
        
        for i, (name, func) in enumerate(stages, 1):
            print(f"\n[Stage {i}/{len(stages)}] {name}")
            if not func():
                success = False
                print(f"\n✗ Pipeline failed at stage: {name}")
                break
        
        if success:
            print("\n" + "="*60)
            print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nCheck the output/ directory for results:")
            print("  - output/track_a.jsonl")
            print("  - output/track_b.npy")
            print("  - output/evaluation_results.json")
    
    elif args.stage == 'prepare':
        success = stage_prepare()
    
    elif args.stage == 'train':
        success = stage_train_track_b() and stage_train_track_a()
    
    elif args.stage == 'train_a':
        success = stage_train_track_a()
    
    elif args.stage == 'train_b':
        success = stage_train_track_b()
    
    elif args.stage == 'inference':
        success = stage_inference_track_b() and stage_inference_track_a()
    
    elif args.stage == 'eval':
        success = stage_evaluate()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

