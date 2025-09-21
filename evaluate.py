#!/usr/bin/env python3
"""
Smart City Computer Vision - Model Evaluation and Benchmarking

This script provides comprehensive evaluation and benchmarking capabilities
for all three detection models in the smart city computer vision project.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import load_model, get_class_names, count_dataset_files
from config import MODEL_CONFIGS, PERFORMANCE_TARGETS, create_output_directory


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Smart City Computer Vision - Model Evaluation')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['garbage', 'helmet', 'traffic', 'all'],
                       help='Model type to evaluate')
    parser.add_argument('--weights', type=str, help='Path to model weights (optional)')
    parser.add_argument('--data', type=str, help='Path to dataset config (optional)')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference speed benchmarking')
    parser.add_argument('--compare', action='store_true',
                       help='Compare against performance targets')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save evaluation plots')
    
    return parser.parse_args()


def evaluate_model(model_type, weights_path=None, data_config=None, conf=0.5, iou=0.45):
    """
    Evaluate a specific model and return metrics
    
    Args:
        model_type (str): Type of model ('garbage', 'helmet', 'traffic')
        weights_path (str): Path to model weights
        data_config (str): Path to dataset configuration
        conf (float): Confidence threshold
        iou (float): IoU threshold
        
    Returns:
        dict: Evaluation metrics and results
    """
    print(f"🔍 Evaluating {model_type} detection model...")
    
    # Load model configuration
    config = MODEL_CONFIGS[model_type]
    
    # Determine paths
    if weights_path is None:
        weights_path = f"models/{model_type}/{model_type}_best.pt"
    
    if data_config is None:
        data_config = config['config_file']
    
    # Check if model exists
    if not os.path.exists(weights_path):
        print(f"❌ Model weights not found: {weights_path}")
        print(f"Please train the model first using: python {model_type}_train.py")
        return None
    
    # Load model
    try:
        model = YOLO(weights_path)
        print(f"✅ Model loaded: {weights_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Run validation
    print("📊 Running model validation...")
    try:
        results = model.val(
            data=data_config,
            conf=conf,
            iou=iou,
            plots=True,
            save=True,
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'model_type': model_type,
            'weights_path': weights_path,
            'data_config': data_config,
            'confidence_threshold': conf,
            'iou_threshold': iou,
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr)),
            'class_count': len(config['classes']),
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Add per-class metrics if available
        if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
            class_names = config['classes']
            class_metrics = {}
            for i, class_name in enumerate(class_names):
                if i < len(results.box.ap50):
                    class_metrics[class_name] = {
                        'ap50': float(results.box.ap50[i]),
                        'ap50_95': float(results.box.ap[i]) if results.box.ap is not None and i < len(results.box.ap) else 0.0
                    }
            metrics['per_class_metrics'] = class_metrics
        
        print(f"✅ Evaluation completed for {model_type} model")
        print(f"📈 mAP@0.5: {metrics['map50']:.4f}")
        print(f"📈 mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        print(f"📈 Precision: {metrics['precision']:.4f}")
        print(f"📈 Recall: {metrics['recall']:.4f}")
        print(f"📈 F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None


def benchmark_inference_speed(model_type, weights_path=None, num_runs=100, img_size=640):
    """
    Benchmark inference speed for a model
    
    Args:
        model_type (str): Type of model
        weights_path (str): Path to model weights
        num_runs (int): Number of inference runs
        img_size (int): Input image size
        
    Returns:
        dict: Speed benchmarking results
    """
    print(f"⚡ Benchmarking {model_type} model inference speed...")
    
    if weights_path is None:
        weights_path = f"models/{model_type}/{model_type}_best.pt"
    
    if not os.path.exists(weights_path):
        print(f"❌ Model weights not found: {weights_path}")
        return None
    
    try:
        # Load model
        model = YOLO(weights_path)
        
        # Create dummy input
        import torch
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        # Warmup runs
        print("🔥 Warming up...")
        for _ in range(10):
            _ = model.predict(dummy_input, verbose=False)
        
        # Benchmark runs
        print(f"🏃 Running {num_runs} inference iterations...")
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            _ = model.predict(dummy_input, verbose=False)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        benchmark_results = {
            'model_type': model_type,
            'num_runs': num_runs,
            'img_size': img_size,
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'min_inference_time_ms': min_time * 1000,
            'max_inference_time_ms': max_time * 1000,
            'fps': fps,
            'benchmark_date': datetime.now().isoformat()
        }
        
        print(f"⚡ Benchmark Results for {model_type}:")
        print(f"   Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"   FPS: {fps:.1f}")
        print(f"   Min/Max time: {min_time*1000:.2f}/{max_time*1000:.2f} ms")
        
        return benchmark_results
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return None


def compare_with_targets(metrics, benchmark_results=None):
    """
    Compare evaluation results with performance targets
    
    Args:
        metrics (dict): Evaluation metrics
        benchmark_results (dict): Speed benchmark results
        
    Returns:
        dict: Comparison results
    """
    model_type = metrics['model_type']
    targets = PERFORMANCE_TARGETS[model_type]
    
    print(f"🎯 Comparing {model_type} model with performance targets...")
    
    comparison = {
        'model_type': model_type,
        'metrics_comparison': {},
        'overall_score': 0,
        'recommendations': []
    }
    
    # Compare accuracy metrics
    map50_score = metrics['map50'] / targets['map50'] if targets['map50'] > 0 else 0
    map50_95_score = metrics['map50_95'] / targets['map50_95'] if targets['map50_95'] > 0 else 0
    
    comparison['metrics_comparison']['map50'] = {
        'actual': metrics['map50'],
        'target': targets['map50'],
        'score': map50_score,
        'status': 'PASS' if map50_score >= 0.95 else 'FAIL'
    }
    
    comparison['metrics_comparison']['map50_95'] = {
        'actual': metrics['map50_95'],
        'target': targets['map50_95'],
        'score': map50_95_score,
        'status': 'PASS' if map50_95_score >= 0.95 else 'FAIL'
    }
    
    # Compare speed metrics if available
    if benchmark_results:
        fps_score = benchmark_results['fps'] / targets['inference_fps'] if targets['inference_fps'] > 0 else 0
        comparison['metrics_comparison']['fps'] = {
            'actual': benchmark_results['fps'],
            'target': targets['inference_fps'],
            'score': fps_score,
            'status': 'PASS' if fps_score >= 0.8 else 'FAIL'
        }
    
    # Calculate overall score
    scores = [comp['score'] for comp in comparison['metrics_comparison'].values()]
    comparison['overall_score'] = np.mean(scores) if scores else 0
    
    # Generate recommendations
    if map50_score < 0.95:
        comparison['recommendations'].append("Consider increasing training epochs or improving dataset quality")
    if map50_95_score < 0.95:
        comparison['recommendations'].append("Fine-tune model architecture or augmentation parameters")
    if benchmark_results and benchmark_results['fps'] < targets['inference_fps'] * 0.8:
        comparison['recommendations'].append("Consider using a smaller model variant for better speed")
    
    # Print comparison results
    print(f"📊 Performance Comparison Results:")
    for metric, comp in comparison['metrics_comparison'].items():
        status_emoji = "✅" if comp['status'] == 'PASS' else "❌"
        print(f"   {status_emoji} {metric}: {comp['actual']:.4f} (target: {comp['target']:.4f}) - {comp['status']}")
    
    print(f"🏆 Overall Score: {comparison['overall_score']:.2f}")
    
    if comparison['recommendations']:
        print("💡 Recommendations:")
        for rec in comparison['recommendations']:
            print(f"   • {rec}")
    
    return comparison


def create_evaluation_report(all_results, output_dir):
    """
    Create comprehensive evaluation report
    
    Args:
        all_results (dict): All evaluation results
        output_dir (str): Output directory
    """
    print("📝 Creating evaluation report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_path / 'evaluation_report.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary table
    summary_data = []
    for model_type, results in all_results.items():
        if 'evaluation' in results and results['evaluation']:
            eval_metrics = results['evaluation']
            benchmark_metrics = results.get('benchmark', {})
            
            summary_data.append({
                'Model': model_type.title(),
                'mAP@0.5': eval_metrics['map50'],
                'mAP@0.5:0.95': eval_metrics['map50_95'],
                'Precision': eval_metrics['precision'],
                'Recall': eval_metrics['recall'],
                'F1 Score': eval_metrics['f1_score'],
                'FPS': benchmark_metrics.get('fps', 'N/A'),
                'Avg Time (ms)': benchmark_metrics.get('avg_inference_time_ms', 'N/A')
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = output_path / 'evaluation_summary.csv'
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        create_evaluation_plots(df, output_path)
        
        print(f"✅ Evaluation report saved to: {output_dir}")
        print(f"📄 JSON report: {json_path}")
        print(f"📊 CSV summary: {csv_path}")


def create_evaluation_plots(df, output_dir):
    """Create evaluation visualization plots"""
    try:
        # Accuracy metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Smart City Computer Vision - Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # mAP@0.5
        axes[0, 0].bar(df['Model'], df['mAP@0.5'], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        axes[0, 0].set_title('mAP@0.5', fontweight='bold')
        axes[0, 0].set_ylabel('mAP Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(df['mAP@0.5']):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # mAP@0.5:0.95
        axes[0, 1].bar(df['Model'], df['mAP@0.5:0.95'], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        axes[0, 1].set_title('mAP@0.5:0.95', fontweight='bold')
        axes[0, 1].set_ylabel('mAP Score')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(df['mAP@0.5:0.95']):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Precision vs Recall
        axes[1, 0].scatter(df['Recall'], df['Precision'], 
                          c=['#ff6b6b', '#4ecdc4', '#45b7d1'], s=100, alpha=0.7)
        axes[1, 0].set_title('Precision vs Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        for i, model in enumerate(df['Model']):
            axes[1, 0].annotate(model, (df['Recall'].iloc[i], df['Precision'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        # FPS comparison (if available)
        fps_data = df[df['FPS'] != 'N/A']
        if not fps_data.empty:
            fps_values = [float(x) if x != 'N/A' else 0 for x in df['FPS']]
            axes[1, 1].bar(df['Model'], fps_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            axes[1, 1].set_title('Inference Speed (FPS)', fontweight='bold')
            axes[1, 1].set_ylabel('Frames per Second')
            for i, v in enumerate(fps_values):
                if v > 0:
                    axes[1, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'No benchmark data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Inference Speed (FPS)', fontweight='bold')
        
        plt.tight_layout()
        plot_path = output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plots saved: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create plots: {e}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("🏙️ Smart City Computer Vision - Model Evaluation")
    print("=" * 60)
    
    # Determine models to evaluate
    models_to_evaluate = ['garbage', 'helmet', 'traffic'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_type in models_to_evaluate:
        print(f"\n{'='*20} {model_type.upper()} MODEL {'='*20}")
        
        results = {
            'model_type': model_type,
            'evaluation': None,
            'benchmark': None,
            'comparison': None
        }
        
        # Run evaluation
        evaluation_results = evaluate_model(
            model_type=model_type,
            weights_path=args.weights,
            data_config=args.data,
            conf=args.conf,
            iou=args.iou
        )
        results['evaluation'] = evaluation_results
        
        # Run benchmark if requested
        if args.benchmark and evaluation_results:
            benchmark_results = benchmark_inference_speed(
                model_type=model_type,
                weights_path=args.weights or f"models/{model_type}/{model_type}_best.pt"
            )
            results['benchmark'] = benchmark_results
            
            # Compare with targets if requested
            if args.compare:
                comparison_results = compare_with_targets(evaluation_results, benchmark_results)
                results['comparison'] = comparison_results
        elif args.compare and evaluation_results:
            comparison_results = compare_with_targets(evaluation_results)
            results['comparison'] = comparison_results
        
        all_results[model_type] = results
    
    # Create comprehensive report
    if any(results['evaluation'] for results in all_results.values()):
        create_evaluation_report(all_results, args.output)
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"📁 Results saved to: {args.output}")
    print("🚀 Ready for model deployment and smart city applications!")


if __name__ == "__main__":
    main()