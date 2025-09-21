#!/usr/bin/env python3
"""
Smart City Computer Vision - Logging Utilities

This module provides centralized logging functionality for all components
of the smart city computer vision project with configurable levels and outputs.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (optional)
        log_dir: Log directory path
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        console: Whether to enable console logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Set up rotating file handler
        file_path = log_path / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_model_logger(model_type: str) -> logging.Logger:
    """
    Get a logger configured for specific model type.
    
    Args:
        model_type: Type of model (garbage, helmet, traffic)
        
    Returns:
        Model-specific logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{model_type}_model_{timestamp}.log"
    
    return setup_logger(
        name=f"smartcity.{model_type}",
        level="INFO",
        log_file=log_file,
        console=True
    )


def get_training_logger(model_type: str) -> logging.Logger:
    """
    Get a logger configured for training sessions.
    
    Args:
        model_type: Type of model being trained
        
    Returns:
        Training-specific logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{model_type}_training_{timestamp}.log"
    
    return setup_logger(
        name=f"smartcity.training.{model_type}",
        level="DEBUG",
        log_file=log_file,
        console=True
    )


def get_evaluation_logger(model_type: str) -> logging.Logger:
    """
    Get a logger configured for model evaluation.
    
    Args:
        model_type: Type of model being evaluated
        
    Returns:
        Evaluation-specific logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{model_type}_evaluation_{timestamp}.log"
    
    return setup_logger(
        name=f"smartcity.evaluation.{model_type}",
        level="INFO",
        log_file=log_file,
        console=True
    )


def get_inference_logger(model_type: str) -> logging.Logger:
    """
    Get a logger configured for inference/demo sessions.
    
    Args:
        model_type: Type of model for inference
        
    Returns:
        Inference-specific logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{model_type}_inference_{timestamp}.log"
    
    return setup_logger(
        name=f"smartcity.inference.{model_type}",
        level="INFO",
        log_file=log_file,
        console=True
    )


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging and tracking.
    
    Args:
        logger: Logger instance to use
    """
    import platform
    import torch
    import cv2
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("SMART CITY COMPUTER VISION - SYSTEM INFORMATION")
    logger.info("=" * 60)
    
    # System information
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Package versions
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    
    # CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("CUDA available: No")
    
    logger.info("=" * 60)


def log_training_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log training configuration parameters.
    
    Args:
        logger: Logger instance to use
        config: Training configuration dictionary
    """
    logger.info("TRAINING CONFIGURATION")
    logger.info("-" * 40)
    
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("-" * 40)


def log_model_summary(logger: logging.Logger, model_info: Dict[str, Any]) -> None:
    """
    Log model architecture and parameter summary.
    
    Args:
        logger: Logger instance to use
        model_info: Model information dictionary
    """
    logger.info("MODEL SUMMARY")
    logger.info("-" * 40)
    
    logger.info(f"Model type: {model_info.get('type', 'Unknown')}")
    logger.info(f"Architecture: {model_info.get('architecture', 'Unknown')}")
    logger.info(f"Parameters: {model_info.get('parameters', 'Unknown'):,}")
    logger.info(f"Model size: {model_info.get('size_mb', 'Unknown')} MB")
    logger.info(f"Input size: {model_info.get('input_size', 'Unknown')}")
    logger.info(f"Classes: {model_info.get('num_classes', 'Unknown')}")
    
    logger.info("-" * 40)


def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, float]) -> None:
    """
    Log model performance metrics.
    
    Args:
        logger: Logger instance to use
        metrics: Performance metrics dictionary
    """
    logger.info("PERFORMANCE METRICS")
    logger.info("-" * 40)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    logger.info("-" * 40)


def setup_ultralytics_logging(level: str = "INFO") -> None:
    """
    Configure Ultralytics YOLO logging to integrate with project logging.
    
    Args:
        level: Logging level for Ultralytics
    """
    # Configure ultralytics logging
    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove default handlers if any
    ultralytics_logger.handlers.clear()
    
    # Create a custom handler that formats ultralytics logs
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - YOLO - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    ultralytics_logger.addHandler(handler)


# Default logger for the project
project_logger = setup_logger(
    name="smartcity",
    level="INFO",
    log_file="smartcity_main.log",
    console=True
)


if __name__ == "__main__":
    # Test logging functionality
    test_logger = setup_logger("test", "DEBUG", "test.log")
    
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    test_logger.critical("Critical message")
    
    log_system_info(test_logger)
    
    print("Logging test completed. Check test.log file.")