#!/usr/bin/env python3
"""
Smart City Computer Vision - Setup and Environment Check

This script helps set up the environment and checks dependencies
for the Smart City Computer Vision project.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
        
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False


def install_requirements():
    """Install packages from requirements.txt"""
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def check_directory_structure():
    """Check if the project directory structure is correct"""
    print("📁 Checking project structure...")
    
    project_root = Path(__file__).parent
    expected_dirs = [
        'garbage-detection',
        'helmet-detection', 
        'traffic-detection',
        'models',
        'utils',
        'notebooks'
    ]
    
    expected_files = [
        'requirements.txt',
        'demo.py',
        'garbage_train.py',
        'helmet_train.py',
        'traffic_train.py',
        'README.md'
    ]
    
    all_good = True
    
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ - OK")
        else:
            print(f"❌ {dir_name}/ - Missing")
            all_good = False
    
    for file_name in expected_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ {file_name} - OK")
        else:
            print(f"❌ {file_name} - Missing")
            all_good = False
    
    return all_good


def check_dataset_structure():
    """Check dataset directory structure"""
    print("\n📊 Checking dataset structure...")
    
    project_root = Path(__file__).parent
    models = ['garbage', 'helmet', 'traffic']
    
    for model in models:
        print(f"\n{model.title()} Detection:")
        data_dir = project_root / f'{model}-detection' / 'data'
        
        if not data_dir.exists():
            print(f"  ❌ Data directory missing: {data_dir}")
            continue
        
        splits = ['train', 'valid', 'test']
        for split in splits:
            split_dir = data_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
                label_count = len(list(labels_dir.glob('*.txt')))
                print(f"  ✅ {split}: {image_count} images, {label_count} labels")
            else:
                print(f"  ⚠️  {split}: Directory structure needs setup")


def check_gpu_support():
    """Check for GPU support"""
    print("\n🖥️ Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available - {gpu_count} GPU(s)")
            print(f"   Primary GPU: {gpu_name}")
        else:
            print("⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU")


def create_sample_data_structure():
    """Create sample data directory structure"""
    print("\n📁 Creating sample data structure...")
    
    project_root = Path(__file__).parent
    models = ['garbage', 'helmet', 'traffic']
    
    for model in models:
        data_dir = project_root / f'{model}-detection' / 'data'
        
        splits = ['train', 'valid', 'test']
        for split in splits:
            images_dir = data_dir / split / 'images'
            labels_dir = data_dir / split / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("✅ Sample data structure created!")
    print("📝 Add your YOLO format datasets to the respective directories.")


def main():
    """Main setup function"""
    print("🏙️ Smart City Computer Vision - Setup & Environment Check")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return False
    
    print("\n📦 Checking required packages...")
    
    # Core packages to check
    packages = [
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('pillow', 'PIL'),
        ('pyyaml', 'yaml')
    ]
    
    missing_packages = []
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        response = input("Install missing packages? (y/n): ").lower()
        if response == 'y':
            if not install_requirements():
                print("\n❌ Setup failed: Could not install requirements")
                return False
        else:
            print("\n⚠️  Skipping package installation")
    
    print("\n" + "=" * 60)
    
    # Check project structure
    if check_directory_structure():
        print("✅ Project structure - OK")
    else:
        print("❌ Project structure - Issues found")
    
    # Check datasets
    check_dataset_structure()
    
    # Check GPU
    check_gpu_support()
    
    # Create sample structure if needed
    response = input("\nCreate sample data directory structure? (y/n): ").lower()
    if response == 'y':
        create_sample_data_structure()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Add your YOLO datasets to the respective data directories")
    print("2. Train models using: python garbage_train.py, python helmet_train.py, python traffic_train.py")
    print("3. Run inference using: python demo.py --model [model_type] --source [input_source]")
    print("4. Check README.md for detailed instructions")
    
    print("\n🚀 Ready to build your Smart City Computer Vision system!")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)