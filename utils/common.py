"""
Common utility functions for Smart City Computer Vision project.
Provides shared functionality for training and inference scripts.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from ultralytics import YOLO


def load_model(model_path: str) -> YOLO:
    """
    Load a trained YOLO model from the specified path.
    
    Args:
        model_path (str): Path to the model weights file
        
    Returns:
        YOLO: Loaded YOLO model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return YOLO(model_path)


def create_output_dirs(base_path: str) -> Dict[str, str]:
    """
    Create output directories for training results.
    
    Args:
        base_path (str): Base path for output directories
        
    Returns:
        Dict[str, str]: Dictionary with paths to created directories
    """
    dirs = {
        'runs': os.path.join(base_path, 'runs'),
        'weights': os.path.join(base_path, 'weights'),
        'results': os.path.join(base_path, 'results')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def draw_predictions(image: np.ndarray, results, class_names: Dict[int, str]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image (np.ndarray): Input image
        results: YOLO prediction results
        class_names (Dict[int, str]): Mapping of class IDs to names
        
    Returns:
        np.ndarray: Image with drawn predictions
    """
    annotated_image = image.copy()
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = class_names.get(class_id, f"Class_{class_id}")
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), 
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, 
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated_image


def get_class_names(model_type: str) -> Dict[int, str]:
    """
    Get class names for different model types.
    
    Args:
        model_type (str): Type of model ('garbage', 'helmet', 'traffic')
        
    Returns:
        Dict[int, str]: Dictionary mapping class IDs to names
    """
    class_mappings = {
        'garbage': {
            0: 'cardboard',
            1: 'glass', 
            2: 'metal',
            3: 'paper',
            4: 'plastic',
            5: 'trash'
        },
        'helmet': {
            0: 'helmet',
            1: 'no-helmet',
            2: 'person'
        },
        'traffic': {
            0: 'car',
            1: 'motorcycle',
            2: 'bus', 
            3: 'truck',
            4: 'traffic-light',
            5: 'stop-sign',
            6: 'person',
            7: 'bicycle',
            8: 'violation'
        }
    }
    
    return class_mappings.get(model_type, {})


def save_results_plot(results_path: str, output_path: str):
    """
    Save training results plot.
    
    Args:
        results_path (str): Path to results.png from training
        output_path (str): Path to save the plot
    """
    if os.path.exists(results_path):
        img = cv2.imread(results_path)
        cv2.imwrite(output_path, img)
        print(f"Results plot saved to: {output_path}")


def validate_dataset_structure(dataset_path: str) -> bool:
    """
    Validate that the dataset has the correct YOLO structure.
    
    Args:
        dataset_path (str): Path to dataset directory
        
    Returns:
        bool: True if structure is valid
    """
    required_dirs = ['train', 'valid', 'test']
    
    for split in required_dirs:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: Missing directory {split_path}")
            return False
            
        # Check for images and labels subdirectories
        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Warning: Missing images or labels directory in {split_path}")
            print("Expected structure: {split}/images/ and {split}/labels/")
    
    return True


def count_dataset_files(dataset_path: str) -> Dict[str, int]:
    """
    Count the number of images in each dataset split.
    
    Args:
        dataset_path (str): Path to dataset directory
        
    Returns:
        Dict[str, int]: Count of images per split
    """
    counts = {}
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        images_path = os.path.join(dataset_path, split, 'images')
        if os.path.exists(images_path):
            image_files = [f for f in os.listdir(images_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            counts[split] = len(image_files)
        else:
            counts[split] = 0
    
    return counts


def print_training_summary(model_type: str, epochs: int, img_size: int, batch_size: int):
    """
    Print a summary of training parameters.
    
    Args:
        model_type (str): Type of model being trained
        epochs (int): Number of training epochs
        img_size (int): Image size for training
        batch_size (int): Batch size for training
    """
    print("="*60)
    print(f"TRAINING {model_type.upper()} DETECTION MODEL")
    print("="*60)
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print("="*60)