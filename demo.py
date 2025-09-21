#!/usr/bin/env python3
"""
Smart City Computer Vision - Unified Demo Script

This script provides a unified interface for running inference with any trained model
(garbage detection, helmet detection, or traffic violation detection).

Usage Examples:
    # Run on image
    python demo.py --model garbage --source test.jpg
    python demo.py --model helmet --source safety_check.png
    python demo.py --model traffic --source traffic_scene.jpg
    
    # Run on video  
    python demo.py --model helmet --source safety_video.mp4
    python demo.py --model traffic --source traffic_cam.avi
    
    # Run on webcam
    python demo.py --model garbage --source 0
    python demo.py --model helmet --source webcam
    
    # Additional options
    python demo.py --model traffic --source video.mp4 --conf 0.5 --save --show

Author: Smart City Computer Vision Project
"""

import argparse
import cv2
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import load_model, get_class_names, draw_predictions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart City Computer Vision - Unified Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Image inference
    python demo.py --model garbage --source test.jpg
    python demo.py --model helmet --source safety.png --conf 0.6
    python demo.py --model traffic --source street.jpg --save
    
    # Video inference
    python demo.py --model helmet --source video.mp4 --show
    python demo.py --model traffic --source cam_feed.avi --save --output results/
    
    # Webcam inference
    python demo.py --model garbage --source 0
    python demo.py --model helmet --source webcam --conf 0.7
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['garbage', 'helmet', 'traffic'],
        help='Model type to use for inference'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source for inference (image path, video path, webcam index, or "webcam")'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights (if not provided, will use default path)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions (default: 0.5)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save inference results'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='Display inference results (default: True)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='runs/inference',
        help='Output directory for saved results (default: runs/inference)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for inference (default: 640)'
    )
    
    return parser.parse_args()


def get_model_path(model_type: str, custom_weights: Optional[str] = None) -> str:
    """
    Get the path to model weights.
    
    Args:
        model_type (str): Type of model ('garbage', 'helmet', 'traffic')
        custom_weights (Optional[str]): Custom weights path
        
    Returns:
        str: Path to model weights
    """
    if custom_weights and os.path.exists(custom_weights):
        return custom_weights
    
    # Default paths
    project_root = Path(__file__).parent
    default_paths = {
        'garbage': project_root / 'models' / 'garbage' / 'garbage_best.pt',
        'helmet': project_root / 'models' / 'helmet' / 'helmet_best.pt',
        'traffic': project_root / 'models' / 'traffic' / 'traffic_best.pt'
    }
    
    model_path = default_paths.get(model_type)
    
    if model_path and model_path.exists():
        return str(model_path)
    else:
        raise FileNotFoundError(
            f"Model weights not found for {model_type}. "
            f"Expected path: {model_path}\n"
            f"Please train the model first using {model_type}_train.py"
        )


def process_source(source: str) -> tuple:
    """
    Process the source input and determine the type.
    
    Args:
        source (str): Source input
        
    Returns:
        tuple: (source_path, source_type)
    """
    # Check if it's a webcam
    if source.lower() == 'webcam' or source == '0':
        return 0, 'webcam'
    
    # Try to convert to int for webcam index
    try:
        webcam_id = int(source)
        return webcam_id, 'webcam'
    except ValueError:
        pass
    
    # Check if it's a file
    if os.path.exists(source):
        # Check if it's an image
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')
        
        source_lower = source.lower()
        if source_lower.endswith(image_extensions):
            return source, 'image'
        elif source_lower.endswith(video_extensions):
            return source, 'video'
        else:
            return source, 'unknown'
    else:
        raise FileNotFoundError(f"Source file not found: {source}")


def run_image_inference(model, image_path: str, class_names: dict, args):
    """Run inference on a single image."""
    print(f"📸 Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Run inference
    results = model(image, conf=args.conf, iou=args.iou, imgsz=args.img_size)
    
    # Draw predictions
    annotated_image = draw_predictions(image, results, class_names)
    
    # Save results if requested
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{Path(image_path).stem}_predicted{Path(image_path).suffix}"
        cv2.imwrite(str(output_path), annotated_image)
        print(f"💾 Results saved to: {output_path}")
    
    # Display results if requested
    if args.show:
        # Resize image if too large for display
        height, width = annotated_image.shape[:2]
        if height > 800 or width > 1200:
            scale = min(800/height, 1200/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            annotated_image = cv2.resize(annotated_image, (new_width, new_height))
        
        cv2.imshow(f'Smart City CV - {args.model.title()} Detection', annotated_image)
        print("👁️ Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print detection summary
    if results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"✅ Found {num_detections} objects")
        
        # Count detections per class
        if num_detections > 0:
            class_counts = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                class_name = class_names.get(class_id, f"Class_{class_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("📊 Detection summary:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")
    else:
        print("❌ No objects detected")


def run_video_inference(model, source, class_names: dict, args):
    """Run inference on video or webcam."""
    source_type = 'webcam' if isinstance(source, int) else 'video'
    print(f"🎥 Processing {source_type}: {source}")
    
    # Open video capture
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open {source_type}: {source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if source_type == 'video' else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📺 Video properties: {width}x{height} @ {fps}fps")
    
    # Setup video writer if saving
    video_writer = None
    if args.save and source_type == 'video':
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{Path(source).stem}_predicted.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"💾 Saving results to: {output_path}")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.img_size)
            
            # Draw predictions
            annotated_frame = draw_predictions(frame, results, class_names)
            
            # Add frame info
            fps_text = f"FPS: {frame_count / (time.time() - start_time):.1f}"
            detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
            detection_text = f"Detections: {detection_count}"
            
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, detection_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Model: {args.model.title()}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame if requested
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Display frame if requested
            if args.show:
                # Resize frame if too large for display
                display_frame = annotated_frame.copy()
                if height > 720 or width > 1280:
                    scale = min(720/height, 1280/width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                cv2.imshow(f'Smart City CV - {args.model.title()} Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("🛑 Stopping inference...")
                    break
                elif key == ord('s'):  # 's' to save current frame
                    if args.save:
                        output_dir = Path(args.output)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_path), annotated_frame)
                        print(f"💾 Frame saved: {frame_path}")
            
            # Print periodic updates for video files
            if source_type == 'video' and frame_count % (fps * 5) == 0:  # Every 5 seconds
                print(f"⏱️ Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\n🛑 Inference interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📈 Processing complete:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")


def main():
    """Main function to run the demo."""
    args = parse_arguments()
    
    print("🚀 Smart City Computer Vision - Demo")
    print("=" * 60)
    print(f"Model: {args.model.title()} Detection")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print("=" * 60)
    
    try:
        # Load model
        print("📥 Loading model...")
        model_path = get_model_path(args.model, args.weights)
        model = load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Get class names
        class_names = get_class_names(args.model)
        print(f"🏷️ Classes: {list(class_names.values())}")
        
        # Process source
        source, source_type = process_source(args.source)
        
        # Run inference based on source type
        if source_type == 'image':
            run_image_inference(model, source, class_names, args)
        elif source_type in ['video', 'webcam']:
            run_video_inference(model, source, class_names, args)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        print("\n🎉 Inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()