#!/usr/bin/env python3
"""
Helmet Detection Model Training Script

This script trains a YOLOv8 model for helmet detection and safety compliance.
The model can detect helmets, people without helmets, and persons in construction/industrial sites.

Usage:
    python helmet_train.py

Requirements:
    - Dataset should be in YOLO format in ./helmet-detection/data/
    - Expected structure: train/valid/test folders with images/ and labels/ subfolders
"""

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import (
    create_output_dirs,
    validate_dataset_structure,
    count_dataset_files,
    print_training_summary,
    save_results_plot
)


def train_helmet_detection_model():
    """
    Train YOLOv8 model for helmet detection.
    """
    # Configuration parameters
    MODEL_TYPE = 'helmet'
    EPOCHS = 150  # More epochs for safety-critical application
    IMG_SIZE = 640
    BATCH_SIZE = 16
    PATIENCE = 75
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATASET_PATH = PROJECT_ROOT / 'helmet-detection' / 'data'
    CONFIG_PATH = PROJECT_ROOT / 'helmet-detection' / 'helmet_config.yaml'
    OUTPUT_DIR = PROJECT_ROOT / 'models' / 'helmet'
    
    # Print training summary
    print_training_summary(MODEL_TYPE, EPOCHS, IMG_SIZE, BATCH_SIZE)
    
    # Validate dataset structure
    print("🔍 Validating dataset structure...")
    if not validate_dataset_structure(str(DATASET_PATH)):
        print("❌ Dataset validation failed. Please check your dataset structure.")
        print("Expected structure:")
        print("  helmet-detection/data/")
        print("    ├── train/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    ├── valid/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    └── test/")
        print("        ├── images/")
        print("        └── labels/")
        return False
    
    # Count dataset files
    file_counts = count_dataset_files(str(DATASET_PATH))
    print(f"📊 Dataset statistics:")
    for split, count in file_counts.items():
        print(f"  {split}: {count} images")
    
    if sum(file_counts.values()) == 0:
        print("❌ No images found in dataset. Please add your data first.")
        return False
    
    # Create output directories
    print("📁 Creating output directories...")
    create_output_dirs(str(OUTPUT_DIR))
    
    try:
        # Initialize YOLOv8 model
        print("🚀 Initializing YOLOv8 model...")
        model = YOLO('yolov8s.pt')  # Using YOLOv8 small for better accuracy on safety tasks
        
        # Start training
        print("🏋️ Starting training...")
        print(f"Config file: {CONFIG_PATH}")
        
        results = model.train(
            data=str(CONFIG_PATH),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            save=True,
            project=str(OUTPUT_DIR / 'runs'),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=15,
            resume=False,
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,
            profile=False,
            # Data augmentation parameters - tuned for helmet detection
            hsv_h=0.01,   # Lower hue variation for better color consistency
            hsv_s=0.6,    # HSV saturation augmentation  
            hsv_v=0.3,    # HSV value augmentation
            degrees=5.0,  # Small rotation for realistic helmet angles
            translate=0.15, # Translation fraction
            scale=0.6,    # Scaling factor
            shear=2.0,    # Small shear for perspective variation
            perspective=0.0001, # Small perspective augmentation
            flipud=0.0,   # No vertical flip for helmets
            fliplr=0.5,   # Horizontal flip probability
            mosaic=1.0,   # Mosaic probability
            mixup=0.1,    # Small mixup for better generalization
            copy_paste=0.05 # Small copy-paste for data diversity
        )
        
        # Find the best model weights
        train_dir = OUTPUT_DIR / 'runs' / 'train'
        best_weights = train_dir / 'weights' / 'best.pt'
        last_weights = train_dir / 'weights' / 'last.pt'
        
        # Copy best weights to models directory
        if best_weights.exists():
            shutil.copy2(str(best_weights), str(OUTPUT_DIR / 'helmet_best.pt'))
            print(f"✅ Best model saved to: {OUTPUT_DIR / 'helmet_best.pt'}")
        
        if last_weights.exists():
            shutil.copy2(str(last_weights), str(OUTPUT_DIR / 'helmet_last.pt'))
            print(f"✅ Last model saved to: {OUTPUT_DIR / 'helmet_last.pt'}")
        
        # Save training results plot
        results_plot = train_dir / 'results.png'
        if results_plot.exists():
            save_results_plot(str(results_plot), str(OUTPUT_DIR / 'helmet_training_results.png'))
        
        # Validate the trained model
        print("🔍 Validating trained model...")
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
            val_results = val_model.val(data=str(CONFIG_PATH))
            print(f"📈 Validation mAP50: {val_results.box.map50:.4f}")
            print(f"📈 Validation mAP50-95: {val_results.box.map:.4f}")
        
        print("="*60)
        print("🎉 HELMET DETECTION MODEL TRAINING COMPLETED!")
        print("="*60)
        print(f"📁 Model files saved in: {OUTPUT_DIR}")
        print(f"📊 Training logs saved in: {train_dir}")
        print("🚀 Ready for inference! Use demo.py to test the model.")
        print("⚠️ This model is crucial for workplace safety compliance!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    print("⛑️ Starting Helmet Detection Model Training")
    print("="*60)
    
    success = train_helmet_detection_model()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)