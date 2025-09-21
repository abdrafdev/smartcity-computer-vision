#!/usr/bin/env python3
"""
Garbage Detection Model Training Script

This script trains a YOLOv8 model for garbage detection using the provided dataset.
The model can detect different types of garbage: cardboard, glass, metal, paper, plastic, and trash.

Usage:
    python garbage_train.py

Requirements:
    - Dataset should be in YOLO format in ./garbage-detection/data/
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


def train_garbage_detection_model():
    """
    Train YOLOv8 model for garbage detection.
    """
    # Configuration parameters
    MODEL_TYPE = 'garbage'
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 16
    PATIENCE = 50
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATASET_PATH = PROJECT_ROOT / 'garbage-detection' / 'data'
    CONFIG_PATH = PROJECT_ROOT / 'garbage-detection' / 'garbage_config.yaml'
    OUTPUT_DIR = PROJECT_ROOT / 'models' / 'garbage'
    
    # Print training summary
    print_training_summary(MODEL_TYPE, EPOCHS, IMG_SIZE, BATCH_SIZE)
    
    # Validate dataset structure
    print("🔍 Validating dataset structure...")
    if not validate_dataset_structure(str(DATASET_PATH)):
        print("❌ Dataset validation failed. Please check your dataset structure.")
        print("Expected structure:")
        print("  garbage-detection/data/")
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
        model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for faster training
        
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
            close_mosaic=10,
            resume=False,
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,
            profile=False,
            # Data augmentation parameters
            hsv_h=0.015,  # HSV hue augmentation
            hsv_s=0.7,    # HSV saturation augmentation  
            hsv_v=0.4,    # HSV value augmentation
            degrees=0.0,  # Rotation degrees
            translate=0.1, # Translation fraction
            scale=0.5,    # Scaling factor
            shear=0.0,    # Shear degrees
            perspective=0.0, # Perspective augmentation
            flipud=0.0,   # Flip up-down probability
            fliplr=0.5,   # Flip left-right probability
            mosaic=1.0,   # Mosaic probability
            mixup=0.0,    # Mixup probability
            copy_paste=0.0 # Copy-paste probability
        )
        
        # Find the best model weights
        train_dir = OUTPUT_DIR / 'runs' / 'train'
        best_weights = train_dir / 'weights' / 'best.pt'
        last_weights = train_dir / 'weights' / 'last.pt'
        
        # Copy best weights to models directory
        if best_weights.exists():
            shutil.copy2(str(best_weights), str(OUTPUT_DIR / 'garbage_best.pt'))
            print(f"✅ Best model saved to: {OUTPUT_DIR / 'garbage_best.pt'}")
        
        if last_weights.exists():
            shutil.copy2(str(last_weights), str(OUTPUT_DIR / 'garbage_last.pt'))
            print(f"✅ Last model saved to: {OUTPUT_DIR / 'garbage_last.pt'}")
        
        # Save training results plot
        results_plot = train_dir / 'results.png'
        if results_plot.exists():
            save_results_plot(str(results_plot), str(OUTPUT_DIR / 'garbage_training_results.png'))
        
        # Validate the trained model
        print("🔍 Validating trained model...")
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
            val_results = val_model.val(data=str(CONFIG_PATH))
            print(f"📈 Validation mAP50: {val_results.box.map50:.4f}")
            print(f"📈 Validation mAP50-95: {val_results.box.map:.4f}")
        
        print("="*60)
        print("🎉 GARBAGE DETECTION MODEL TRAINING COMPLETED!")
        print("="*60)
        print(f"📁 Model files saved in: {OUTPUT_DIR}")
        print(f"📊 Training logs saved in: {train_dir}")
        print("🚀 Ready for inference! Use demo.py to test the model.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🗑️ Starting Garbage Detection Model Training")
    print("="*60)
    
    success = train_garbage_detection_model()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)