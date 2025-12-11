"""
Utilities for Google Colab training with Drive integration.
"""

import os
import sys
from pathlib import Path


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_environment(project_name="narrative_similarity"):
    """
    Setup Google Colab environment with Drive mounting.
    
    Args:
        project_name: Name of project folder in Google Drive
        
    Returns:
        dict: Paths configuration for Colab
    """
    if not is_colab():
        return None
    
    print("🔧 Setting up Google Colab environment...")
    
    # Mount Google Drive
    from google.colab import drive
    drive_mount = '/content/drive'
    
    if not os.path.exists(f"{drive_mount}/MyDrive"):
        print("📁 Mounting Google Drive...")
        drive.mount(drive_mount)
        print("✓ Drive mounted successfully!")
    else:
        print("✓ Drive already mounted")
    
    # Setup paths
    drive_project = Path(f"{drive_mount}/MyDrive/{project_name}")
    
    # Create necessary directories in Drive
    dirs_to_create = [
        drive_project / "data",
        drive_project / "data/prepared",
        drive_project / "models",
        drive_project / "output",
        drive_project / "logs"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Project directory: {drive_project}")
    
    # Check for required files
    config_file = drive_project / "config.yaml"
    data_files = [
        drive_project / "data/dev_track_a.jsonl",
        drive_project / "data/dev_track_b.jsonl"
    ]
    
    missing_files = []
    if not config_file.exists():
        missing_files.append("config.yaml")
    for data_file in data_files:
        if not data_file.exists():
            missing_files.append(str(data_file.relative_to(drive_project)))
    
    if missing_files:
        print("\n⚠️  Missing required files in Google Drive:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease upload these files to: {drive_project}/")
        print("\nRequired structure:")
        print(f"{project_name}/")
        print("  ├── config.yaml")
        print("  └── data/")
        print("      ├── dev_track_a.jsonl")
        print("      └── dev_track_b.jsonl")
        sys.exit(1)
    
    print("✓ All required files found")
    
    # Return paths configuration
    paths = {
        'project_root': str(drive_project),
        'data_dir': str(drive_project / "data"),
        'models_dir': str(drive_project / "models"),
        'output_dir': str(drive_project / "output"),
        'config_path': str(config_file),
        'prepared_data_dir': str(drive_project / "data/prepared")
    }
    
    # Change to project directory
    os.chdir(drive_project)
    print(f"✓ Changed to project directory: {os.getcwd()}")
    
    return paths


def update_config_for_colab(config, colab_paths):
    """
    Update config paths for Colab environment.
    
    Args:
        config: Original config dict
        colab_paths: Paths dict from setup_colab_environment
        
    Returns:
        Updated config dict
    """
    if colab_paths is None:
        return config
    
    # Update data paths
    config['data']['dev_track_a'] = f"{colab_paths['data_dir']}/dev_track_a.jsonl"
    config['data']['dev_track_b'] = f"{colab_paths['data_dir']}/dev_track_b.jsonl"
    config['data']['output_dir'] = colab_paths['output_dir']
    config['data']['prepared_data_dir'] = colab_paths['prepared_data_dir']
    
    # Update model save paths
    config['track_a']['model_save_path'] = f"{colab_paths['models_dir']}/track_a_cross_encoder"
    config['track_b']['model_save_path'] = f"{colab_paths['models_dir']}/track_b_embedder"
    
    return config


def install_colab_dependencies():
    """Install required packages in Colab."""
    if not is_colab():
        return
    
    print("📦 Installing dependencies...")
    os.system("pip install -q transformers sentence-transformers datasets scikit-learn pyyaml accelerate")
    print("✓ Dependencies installed")


def print_gpu_info():
    """Print GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("\n⚠️  No GPU available - using CPU")
    except ImportError:
        print("\n⚠️  PyTorch not installed yet")

