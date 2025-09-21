#!/usr/bin/env python3
"""
Smart City Computer Vision - Global Configuration

This module contains global configuration settings for all detection models
and shared parameters across the smart city computer vision project.
"""

import os
from pathlib import Path

# Project Information
PROJECT_NAME = "Smart City Computer Vision"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Abdul Rafay"
PROJECT_EMAIL = "abdrafdev@gmail.com"
PROJECT_DESCRIPTION = "AI-powered computer vision solutions for smart city applications"

# Base Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATASETS_DIR = PROJECT_ROOT / 'dataset'
UTILS_DIR = PROJECT_ROOT / 'utils'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Model Configurations
MODEL_CONFIGS = {
    'garbage': {
        'model_type': 'yolov8n.pt',  # Nano for speed
        'config_file': 'garbage-detection/garbage_config.yaml',
        'classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'patience': 50,
        'description': 'Automated waste management and environmental monitoring'
    },
    'helmet': {
        'model_type': 'yolov8s.pt',  # Small for balanced performance
        'config_file': 'helmet-detection/helmet_config.yaml',
        'classes': ['helmet', 'no-helmet', 'person'],
        'epochs': 150,
        'batch_size': 14,
        'img_size': 640,
        'patience': 75,
        'description': 'Construction site safety compliance monitoring'
    },
    'traffic': {
        'model_type': 'yolov8m.pt',  # Medium for high accuracy
        'config_file': 'traffic-detection/traffic_config.yaml',
        'classes': ['10', '11', '12', '13', '13 0 0 0 0', '14', '15', '16', '17', '18', '19', '20', '20 0 0 0 0', '22', '220', '2225', '224', '225', '23', '2323', '23230', '24', '25', '4 0 0 0 0'],
        'epochs': 200,
        'batch_size': 12,
        'img_size': 640,
        'patience': 100,
        'description': 'Smart traffic management and violation detection'
    }
}

# Training Parameters
TRAINING_DEFAULTS = {
    'optimizer': 'AdamW',
    'lr0': 0.01,  # Initial learning rate
    'lrf': 0.01,  # Final learning rate (lr0 * lrf)
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,  # Cosine learning rate scheduler
    'amp': True,  # Automatic Mixed Precision
    'deterministic': True,
    'single_cls': False,
    'seed': 42
}

# Data Augmentation Settings
AUGMENTATION_PARAMS = {
    'garbage': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.8,
        'shear': 2.0,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1
    },
    'helmet': {
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 3.0,
        'translate': 0.1,
        'scale': 0.9,
        'shear': 1.0,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.1,
        'copy_paste': 0.05
    },
    'traffic': {
        'hsv_h': 0.02,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'degrees': 10.0,
        'translate': 0.2,
        'scale': 0.7,
        'shear': 3.0,
        'perspective': 0.0002,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.1
    }
}

# Inference Settings
INFERENCE_DEFAULTS = {
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 300,
    'img_size': 640,
    'device': 'cpu',  # Will be auto-detected
    'half_precision': False
}

# Output Settings
OUTPUT_CONFIG = {
    'save_results': True,
    'save_plots': True,
    'save_weights': True,
    'results_format': 'png',
    'timestamp_format': '%Y%m%d_%H%M%S',
    'create_subdirs': True
}

# Visualization Settings
VISUALIZATION_CONFIG = {
    'colors': {
        'garbage': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
        'helmet': [(0, 255, 0), (0, 0, 255), (255, 165, 0)],
        'traffic': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                   (0, 255, 255), (255, 165, 0), (128, 0, 128), (255, 192, 203), (165, 42, 42)]
    },
    'thickness': 2,
    'font_scale': 0.5,
    'font_thickness': 2,
    'label_padding': 5
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'num_workers': 4,  # DataLoader workers
    'pin_memory': True,
    'prefetch_factor': 2,
    'persistent_workers': True
}

# Logging Settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_dir': 'logs',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Hardware Requirements
HARDWARE_REQUIREMENTS = {
    'minimum': {
        'ram': '8GB',
        'gpu': 'Integrated GPU or GTX 1050',
        'storage': '10GB',
        'training_time': '2-4 hours per model'
    },
    'recommended': {
        'ram': '16GB',
        'gpu': 'GTX 1660 or RTX 3060',
        'storage': '50GB',
        'training_time': '30-90 minutes per model'
    },
    'optimal': {
        'ram': '32GB',
        'gpu': 'RTX 3080 or better',
        'storage': '100GB SSD',
        'training_time': '15-30 minutes per model'
    }
}

# Model Performance Targets
PERFORMANCE_TARGETS = {
    'garbage': {
        'map50': 0.85,
        'map50_95': 0.60,
        'inference_fps': 60,
        'model_size_mb': 6
    },
    'helmet': {
        'map50': 0.90,
        'map50_95': 0.70,
        'inference_fps': 45,
        'model_size_mb': 22
    },
    'traffic': {
        'map50': 0.80,
        'map50_95': 0.55,
        'inference_fps': 30,
        'model_size_mb': 52
    }
}

# Smart City Applications
SMART_CITY_APPLICATIONS = {
    'garbage': [
        'Automated waste collection optimization',
        'Smart bin monitoring and alerts',
        'Recycling facility automation',
        'Environmental compliance monitoring',
        'Public space cleanliness assessment'
    ],
    'helmet': [
        'Construction site safety monitoring',
        'Workplace compliance enforcement',
        'Accident prevention systems',
        'Safety audit automation',
        'Regulatory compliance reporting'
    ],
    'traffic': [
        'Automated traffic violation detection',
        'Smart intersection monitoring',
        'Pedestrian safety systems',
        'Traffic flow optimization',
        'Emergency response coordination'
    ]
}

def get_model_config(model_type):
    """Get configuration for specific model type"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type]

def get_augmentation_params(model_type):
    """Get augmentation parameters for specific model type"""
    if model_type not in AUGMENTATION_PARAMS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(AUGMENTATION_PARAMS.keys())}")
    return AUGMENTATION_PARAMS[model_type]

def create_output_directory(model_type, create_timestamp_subdir=True):
    """Create output directory for model results"""
    import datetime
    
    model_output_dir = OUTPUTS_DIR / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    if create_timestamp_subdir:
        timestamp = datetime.datetime.now().strftime(OUTPUT_CONFIG['timestamp_format'])
        output_dir = model_output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    return model_output_dir

if __name__ == "__main__":
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"Configuration loaded successfully!")
    print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")