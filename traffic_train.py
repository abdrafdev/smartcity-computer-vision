#!/usr/bin/env python3
"""
Traffic Violation Detection Model Training Script

This script trains a YOLOv8 model for traffic violation detection in smart city applications.
The model can detect vehicles, traffic signs, people, and traffic violations.

Usage:
    python traffic_train.py

Requirements:
    - Dataset should be in YOLO format in ./traffic-detection/data/
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


def train_traffic_detection_model():
    """
    Train YOLOv8 model for traffic violation detection.
    """
    # Configuration parameters
    MODEL_TYPE = 'traffic'
    EPOCHS = 200  # More epochs for complex traffic scenarios
    IMG_SIZE = 640
    BATCH_SIZE = 12  # Smaller batch size for complex multi-class detection
    PATIENCE = 100
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATASET_PATH = PROJECT_ROOT / 'dataset' / 'Traffic Violations'
    CONFIG_PATH = PROJECT_ROOT / 'traffic-detection' / 'traffic_config.yaml'
    OUTPUT_DIR = PROJECT_ROOT / 'models' / 'traffic'
    
    # Print training summary
    print_training_summary(MODEL_TYPE, EPOCHS, IMG_SIZE, BATCH_SIZE)
    
    # Validate dataset structure
    print("🔍 Validating dataset structure...")
    if not validate_dataset_structure(str(DATASET_PATH)):
        print("❌ Dataset validation failed. Please check your dataset structure.")
        print("Expected structure:")
        print("  dataset/Traffic Violations/")
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
        model = YOLO('yolov8m.pt')  # Using YOLOv8 medium for complex traffic detection
        
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
            cos_lr=True,
            close_mosaic=20,
            resume=False,
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,
            profile=False,
            # Data augmentation parameters - tuned for traffic detection
            hsv_h=0.02,   # Small hue variation for traffic colors
            hsv_s=0.8,    # Higher saturation for vivid traffic elements
            hsv_v=0.5,    # HSV value augmentation
            degrees=10.0, # Rotation for camera angle variations
            translate=0.2, # Translation fraction for moving vehicles
            scale=0.7,    # Scaling factor for distance variations
            shear=3.0,    # Shear for perspective changes
            perspective=0.0002, # Perspective augmentation
            flipud=0.0,   # No vertical flip for traffic scenes
            fliplr=0.5,   # Horizontal flip probability
            mosaic=1.0,   # Mosaic probability for complex scenes
            mixup=0.2,    # Mixup for better generalization
            copy_paste=0.1, # Copy-paste for varied traffic scenarios
            # Multi-scale training
            rect=True,    # Rectangular training for varied aspect ratios
            # Loss function weights
            cls=0.5,      # Classification loss weight
            box=7.5,      # Box regression loss weight  
            dfl=1.5       # Distribution focal loss weight
        )
        
        # Find the best model weights
        train_dir = OUTPUT_DIR / 'runs' / 'train'
        best_weights = train_dir / 'weights' / 'best.pt'
        last_weights = train_dir / 'weights' / 'last.pt'
        
        # Copy best weights to models directory
        if best_weights.exists():
            shutil.copy2(str(best_weights), str(OUTPUT_DIR / 'traffic_best.pt'))
            print(f"✅ Best model saved to: {OUTPUT_DIR / 'traffic_best.pt'}")
        
        if last_weights.exists():
            shutil.copy2(str(last_weights), str(OUTPUT_DIR / 'traffic_last.pt'))
            print(f"✅ Last model saved to: {OUTPUT_DIR / 'traffic_last.pt'}")
        
        # Save training results plot
        results_plot = train_dir / 'results.png'
        if results_plot.exists():
            save_results_plot(str(results_plot), str(OUTPUT_DIR / 'traffic_training_results.png'))
        
        # Validate the trained model
        print("🔍 Validating trained model...")
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
            val_results = val_model.val(data=str(CONFIG_PATH))
            print(f"📈 Validation mAP50: {val_results.box.map50:.4f}")
            print(f"📈 Validation mAP50-95: {val_results.box.map:.4f}")
            
            # Print per-class metrics
            if hasattr(val_results.box, 'mp') and val_results.box.mp is not None:
                class_names = ['car', 'motorcycle', 'bus', 'truck', 'traffic-light', 
                             'stop-sign', 'person', 'bicycle', 'violation']
                print("\n📊 Per-class Precision:")
                for i, class_name in enumerate(class_names):
                    if i < len(val_results.box.mp):
                        print(f"  {class_name}: {val_results.box.mp[i]:.4f}")
        
        print("="*60)
        print("🎉 TRAFFIC DETECTION MODEL TRAINING COMPLETED!")
        print("="*60)
        print(f"📁 Model files saved in: {OUTPUT_DIR}")
        print(f"📊 Training logs saved in: {train_dir}")
        print("🚀 Ready for inference! Use demo.py to test the model.")
        print("🚦 This model helps enforce traffic safety in smart cities!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚦 Starting Traffic Violation Detection Model Training")
    print("="*60)
    
    success = train_traffic_detection_model()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)