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


def setup_colab_environment(project_name="narrative_similarity", use_repo=True, repo_url=None):
    """
    Setup Google Colab environment with Drive mounting.
    
    NEW: Clone from repo (scripts) + Drive (data + models)
    
    Args:
        project_name: Name of project folder in Google Drive (for data/models)
        use_repo: If True, clone scripts from GitHub repo
        repo_url: GitHub repo URL (if None, will ask user to provide)
        
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
    
    # Setup Drive paths for data and models
    drive_project = Path(f"{drive_mount}/MyDrive/{project_name}")
    
    # Create necessary directories in Drive (only for data/models)
    dirs_to_create = [
        drive_project / "data",
        drive_project / "data/prepared",
        drive_project / "models",
        drive_project / "output"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Drive directory: {drive_project}")
    
    # Check for required data files in Drive
    data_files = [
        drive_project / "data/dev_track_a.jsonl",
        drive_project / "data/dev_track_b.jsonl"
    ]
    
    missing_files = []
    for data_file in data_files:
        if not data_file.exists():
            missing_files.append(str(data_file.relative_to(drive_project)))
    
    if missing_files:
        print("\n⚠️  Missing required DATA files in Google Drive:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease upload data to: {drive_project}/data/")
        print("\nRequired structure:")
        print(f"{project_name}/")
        print("  └── data/")
        print("      ├── dev_track_a.jsonl")
        print("      └── dev_track_b.jsonl")
        sys.exit(1)
    
    print("✓ Data files found in Drive")
    
    # Clone repository if requested
    if use_repo:
        if repo_url is None:
            print("\n⚠️  No repo URL provided!")
            print("Please provide repo URL in setup_colab_environment(repo_url='...')")
            print("Or set use_repo=False to use files from Drive")
            sys.exit(1)
        
        # Clone to /content/project
        project_dir = Path("/content/project")
        if project_dir.exists():
            print("✓ Repository already cloned")
        else:
            print(f"📥 Cloning repository from: {repo_url}")
            os.system(f"git clone {repo_url} /content/project")
            print("✓ Repository cloned successfully!")
        
        # Change to cloned repo directory
        os.chdir(project_dir)
        config_path = project_dir / "config.yaml"
        
        if not config_path.exists():
            print(f"\n⚠️  config.yaml not found in repository!")
            sys.exit(1)
        
        print(f"✓ Working directory: {os.getcwd()}")
        print(f"✓ Config loaded from: {config_path}")
    else:
        # Old behavior: use everything from Drive
        config_path = drive_project / "config.yaml"
        if not config_path.exists():
            print(f"\n⚠️  config.yaml not found in Drive!")
            sys.exit(1)
        os.chdir(drive_project)
        project_dir = drive_project
    
    # Return paths configuration
    paths = {
        'project_root': str(project_dir),
        'data_dir': str(drive_project / "data"),  # Always from Drive
        'models_dir': str(drive_project / "models"),  # Always to Drive
        'output_dir': str(drive_project / "output"),  # Always to Drive
        'config_path': str(config_path),  # From repo or Drive
        'prepared_data_dir': str(drive_project / "data/prepared")  # Always to Drive
    }
    
    print(f"✓ Setup complete!")
    print(f"  - Scripts: {paths['project_root']}")
    print(f"  - Data: {paths['data_dir']}")
    print(f"  - Models: {paths['models_dir']}")
    
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

