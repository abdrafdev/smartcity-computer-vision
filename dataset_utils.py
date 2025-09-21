#!/usr/bin/env python3
"""
Smart City Computer Vision - Dataset Utilities

This script provides dataset management, validation, and preprocessing utilities
for all three detection models in the smart city computer vision project.
"""

import os
import sys
import json
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import yaml

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIGS, DATASETS_DIR


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Smart City Computer Vision - Dataset Utils')
    parser.add_argument('--action', type=str, required=True,
                       choices=['validate', 'analyze', 'split', 'convert', 'stats', 'visualize'],
                       help='Action to perform')
    parser.add_argument('--model', type=str, required=True,
                       choices=['garbage', 'helmet', 'traffic'],
                       help='Model type')
    parser.add_argument('--input', type=str, help='Input directory or file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--split-ratio', type=str, default='0.7,0.2,0.1',
                       help='Train,valid,test split ratio (default: 0.7,0.2,0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--min-size', type=int, default=32,
                       help='Minimum image size for validation')
    parser.add_argument('--max-size', type=int, default=4096,
                       help='Maximum image size for validation')
    
    return parser.parse_args()


def validate_dataset(model_type, dataset_path=None):
    """
    Validate dataset structure and integrity
    
    Args:
        model_type (str): Type of model ('garbage', 'helmet', 'traffic')
        dataset_path (str): Path to dataset directory
        
    Returns:
        dict: Validation results
    """
    print(f"🔍 Validating {model_type} dataset...")
    
    if dataset_path is None:
        dataset_path = DATASETS_DIR / f"{model_type.title().replace('_', ' ')} Detection"
    
    validation_results = {
        'model_type': model_type,
        'dataset_path': str(dataset_path),
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        validation_results['valid'] = False
        validation_results['errors'].append(f"Dataset directory not found: {dataset_path}")
        return validation_results
    
    # Check splits
    splits = ['train', 'valid', 'test']
    split_stats = {}
    
    for split in splits:
        split_path = Path(dataset_path) / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        split_info = {
            'images_count': 0,
            'labels_count': 0,
            'matched_pairs': 0,
            'orphaned_images': 0,
            'orphaned_labels': 0,
            'invalid_images': 0,
            'invalid_labels': 0
        }
        
        if images_path.exists() and labels_path.exists():
            # Count images
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            split_info['images_count'] = len(image_files)
            
            # Count labels
            label_files = list(labels_path.glob('*.txt'))
            split_info['labels_count'] = len(label_files)
            
            # Check for matching pairs
            image_names = {f.stem for f in image_files}
            label_names = {f.stem for f in label_files}
            
            split_info['matched_pairs'] = len(image_names & label_names)
            split_info['orphaned_images'] = len(image_names - label_names)
            split_info['orphaned_labels'] = len(label_names - image_names)
            
            # Validate image files
            for img_file in image_files:
                try:
                    with Image.open(img_file) as img:
                        if img.size[0] < 32 or img.size[1] < 32:
                            split_info['invalid_images'] += 1
                            validation_results['warnings'].append(f"Small image: {img_file}")
                except Exception as e:
                    split_info['invalid_images'] += 1
                    validation_results['errors'].append(f"Invalid image: {img_file} - {e}")
            
            # Validate label files
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                split_info['invalid_labels'] += 1
                                validation_results['errors'].append(
                                    f"Invalid label format: {label_file}:{line_num+1}")
                                break
                            
                            # Check class ID
                            try:
                                class_id = int(parts[0])
                                config = MODEL_CONFIGS[model_type]
                                if class_id >= len(config['classes']):
                                    split_info['invalid_labels'] += 1
                                    validation_results['errors'].append(
                                        f"Invalid class ID {class_id}: {label_file}:{line_num+1}")
                            except ValueError:
                                split_info['invalid_labels'] += 1
                                validation_results['errors'].append(
                                    f"Non-numeric class ID: {label_file}:{line_num+1}")
                            
                            # Check bbox coordinates
                            try:
                                coords = [float(x) for x in parts[1:5]]
                                if any(c < 0 or c > 1 for c in coords):
                                    validation_results['warnings'].append(
                                        f"Coordinates out of range: {label_file}:{line_num+1}")
                            except ValueError:
                                split_info['invalid_labels'] += 1
                                validation_results['errors'].append(
                                    f"Invalid coordinates: {label_file}:{line_num+1}")
                
                except Exception as e:
                    split_info['invalid_labels'] += 1
                    validation_results['errors'].append(f"Cannot read label: {label_file} - {e}")
            
            # Report warnings for orphaned files
            if split_info['orphaned_images'] > 0:
                validation_results['warnings'].append(
                    f"{split} split has {split_info['orphaned_images']} images without labels")
            
            if split_info['orphaned_labels'] > 0:
                validation_results['warnings'].append(
                    f"{split} split has {split_info['orphaned_labels']} labels without images")
            
        else:
            validation_results['errors'].append(f"Missing {split} directory structure")
            validation_results['valid'] = False
        
        split_stats[split] = split_info
    
    validation_results['statistics'] = split_stats
    
    # Overall validation status
    if validation_results['errors']:
        validation_results['valid'] = False
    
    # Print results
    if validation_results['valid']:
        print("✅ Dataset validation passed!")
    else:
        print("❌ Dataset validation failed!")
        print("Errors:")
        for error in validation_results['errors']:
            print(f"  • {error}")
    
    if validation_results['warnings']:
        print("⚠️ Warnings:")
        for warning in validation_results['warnings']:
            print(f"  • {warning}")
    
    return validation_results


def analyze_dataset(model_type, dataset_path=None):
    """
    Analyze dataset statistics and class distribution
    
    Args:
        model_type (str): Type of model
        dataset_path (str): Path to dataset directory
        
    Returns:
        dict: Analysis results
    """
    print(f"📊 Analyzing {model_type} dataset...")
    
    if dataset_path is None:
        dataset_path = DATASETS_DIR / f"{model_type.title().replace('_', ' ')} Detection"
    
    config = MODEL_CONFIGS[model_type]
    class_names = config['classes']
    
    analysis_results = {
        'model_type': model_type,
        'dataset_path': str(dataset_path),
        'class_distribution': defaultdict(lambda: defaultdict(int)),
        'image_statistics': defaultdict(dict),
        'annotation_statistics': defaultdict(dict)
    }
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = Path(dataset_path) / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        if not (images_path.exists() and labels_path.exists()):
            continue
        
        # Image statistics
        image_sizes = []
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    image_sizes.append(img.size)
            except:
                continue
        
        if image_sizes:
            widths, heights = zip(*image_sizes)
            analysis_results['image_statistics'][split] = {
                'count': len(image_sizes),
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': np.min(widths),
                'max_width': np.max(widths),
                'min_height': np.min(heights),
                'max_height': np.max(heights),
                'aspect_ratios': [w/h for w, h in image_sizes]
            }
        
        # Class distribution and annotation statistics
        total_annotations = 0
        annotations_per_image = []
        
        for label_file in labels_path.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    annotations_per_image.append(len(lines))
                    total_annotations += len(lines)
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    analysis_results['class_distribution'][split][class_name] += 1
                                else:
                                    analysis_results['class_distribution'][split]['unknown'] += 1
                            except ValueError:
                                analysis_results['class_distribution'][split]['invalid'] += 1
            except:
                continue
        
        if annotations_per_image:
            analysis_results['annotation_statistics'][split] = {
                'total_annotations': total_annotations,
                'avg_annotations_per_image': np.mean(annotations_per_image),
                'max_annotations_per_image': np.max(annotations_per_image),
                'min_annotations_per_image': np.min(annotations_per_image),
                'std_annotations_per_image': np.std(annotations_per_image)
            }
    
    # Print analysis summary
    print("\n📈 Dataset Analysis Summary:")
    print("-" * 50)
    
    for split in splits:
        if split in analysis_results['image_statistics']:
            img_stats = analysis_results['image_statistics'][split]
            ann_stats = analysis_results['annotation_statistics'].get(split, {})
            
            print(f"\n{split.upper()} Split:")
            print(f"  Images: {img_stats['count']}")
            print(f"  Avg size: {img_stats['avg_width']:.0f}×{img_stats['avg_height']:.0f}")
            print(f"  Size range: {img_stats['min_width']}×{img_stats['min_height']} to "
                  f"{img_stats['max_width']}×{img_stats['max_height']}")
            
            if ann_stats:
                print(f"  Total annotations: {ann_stats['total_annotations']}")
                print(f"  Avg annotations/image: {ann_stats['avg_annotations_per_image']:.1f}")
    
    print(f"\n📊 Class Distribution:")
    for split in splits:
        if split in analysis_results['class_distribution']:
            print(f"\n{split.upper()}:")
            class_dist = analysis_results['class_distribution'][split]
            total_classes = sum(class_dist.values())
            for class_name, count in sorted(class_dist.items()):
                percentage = (count / total_classes) * 100 if total_classes > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return analysis_results


def visualize_dataset_statistics(analysis_results, output_dir=None):
    """
    Create visualization plots for dataset statistics
    
    Args:
        analysis_results (dict): Analysis results from analyze_dataset()
        output_dir (str): Output directory for plots
    """
    if output_dir is None:
        output_dir = f"dataset_analysis_{analysis_results['model_type']}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Class distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{analysis_results["model_type"].title()} Dataset Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Class distribution by split
    splits_with_data = [split for split in ['train', 'valid', 'test'] 
                       if split in analysis_results['class_distribution']]
    
    if splits_with_data:
        split_data = []
        for split in splits_with_data:
            class_dist = analysis_results['class_distribution'][split]
            for class_name, count in class_dist.items():
                if class_name not in ['unknown', 'invalid']:
                    split_data.append({'Split': split.title(), 'Class': class_name, 'Count': count})
        
        if split_data:
            df = pd.DataFrame(split_data)
            pivot_df = df.pivot(index='Class', columns='Split', values='Count').fillna(0)
            
            pivot_df.plot(kind='bar', ax=axes[0, 0], rot=45)
            axes[0, 0].set_title('Class Distribution by Split')
            axes[0, 0].set_ylabel('Number of Annotations')
            axes[0, 0].legend()
    
    # Image size distribution
    if 'train' in analysis_results['image_statistics']:
        img_stats = analysis_results['image_statistics']['train']
        if 'aspect_ratios' in img_stats:
            axes[0, 1].hist(img_stats['aspect_ratios'], bins=30, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Aspect Ratio Distribution (Train)')
            axes[0, 1].set_xlabel('Width/Height Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
            axes[0, 1].legend()
    
    # Annotations per image
    ann_data = []
    for split in splits_with_data:
        if split in analysis_results['annotation_statistics']:
            ann_stats = analysis_results['annotation_statistics'][split]
            ann_data.append({
                'Split': split.title(),
                'Avg Annotations': ann_stats['avg_annotations_per_image'],
                'Max Annotations': ann_stats['max_annotations_per_image']
            })
    
    if ann_data:
        df_ann = pd.DataFrame(ann_data)
        x = range(len(df_ann))
        
        axes[1, 0].bar([i - 0.2 for i in x], df_ann['Avg Annotations'], 
                      width=0.4, label='Average', color='lightcoral')
        axes[1, 0].bar([i + 0.2 for i in x], df_ann['Max Annotations'], 
                      width=0.4, label='Maximum', color='lightblue')
        
        axes[1, 0].set_title('Annotations per Image')
        axes[1, 0].set_ylabel('Number of Annotations')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df_ann['Split'])
        axes[1, 0].legend()
    
    # Dataset size comparison
    sizes_data = []
    for split in splits_with_data:
        if split in analysis_results['image_statistics']:
            img_count = analysis_results['image_statistics'][split]['count']
            ann_count = analysis_results['annotation_statistics'].get(split, {}).get('total_annotations', 0)
            sizes_data.append({'Split': split.title(), 'Images': img_count, 'Annotations': ann_count})
    
    if sizes_data:
        df_sizes = pd.DataFrame(sizes_data)
        x = range(len(df_sizes))
        
        ax2 = axes[1, 1].twinx()
        bars1 = axes[1, 1].bar([i - 0.2 for i in x], df_sizes['Images'], 
                              width=0.4, label='Images', color='gold')
        bars2 = ax2.bar([i + 0.2 for i in x], df_sizes['Annotations'], 
                       width=0.4, label='Annotations', color='mediumpurple')
        
        axes[1, 1].set_title('Dataset Size Comparison')
        axes[1, 1].set_ylabel('Number of Images', color='gold')
        ax2.set_ylabel('Number of Annotations', color='mediumpurple')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df_sizes['Split'])
        
        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plot_path = output_path / 'dataset_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Dataset analysis plots saved: {plot_path}")
    
    # Save analysis report
    report_path = output_path / 'analysis_report.json'
    with open(report_path, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        serializable_results = {
            'model_type': analysis_results['model_type'],
            'dataset_path': analysis_results['dataset_path'],
            'class_distribution': {k: dict(v) for k, v in analysis_results['class_distribution'].items()},
            'image_statistics': dict(analysis_results['image_statistics']),
            'annotation_statistics': dict(analysis_results['annotation_statistics'])
        }
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"📄 Analysis report saved: {report_path}")


def get_dataset_statistics(model_type):
    """
    Get quick statistics for a dataset
    
    Args:
        model_type (str): Type of model
        
    Returns:
        dict: Dataset statistics
    """
    dataset_path = DATASETS_DIR / f"{model_type.title().replace('_', ' ')} Detection"
    
    if not dataset_path.exists():
        return None
    
    stats = {'model_type': model_type}
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        images_path = dataset_path / split / 'images'
        labels_path = dataset_path / split / 'labels'
        
        if images_path.exists() and labels_path.exists():
            image_count = len(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
            label_count = len(list(labels_path.glob('*.txt')))
            
            stats[split] = {
                'images': image_count,
                'labels': label_count
            }
        else:
            stats[split] = {'images': 0, 'labels': 0}
    
    return stats


def main():
    """Main function"""
    args = parse_args()
    
    print("🏙️ Smart City Computer Vision - Dataset Utilities")
    print("=" * 60)
    
    if args.action == 'validate':
        results = validate_dataset(args.model, args.input)
        
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f'{args.model}_validation.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    elif args.action == 'analyze':
        results = analyze_dataset(args.model, args.input)
        
        if args.output:
            visualize_dataset_statistics(results, args.output)
    
    elif args.action == 'stats':
        stats = get_dataset_statistics(args.model)
        if stats:
            print(f"\n📊 {args.model.title()} Dataset Statistics:")
            print("-" * 40)
            for split in ['train', 'valid', 'test']:
                if split in stats:
                    print(f"{split.title():>6}: {stats[split]['images']} images, {stats[split]['labels']} labels")
        else:
            print(f"❌ Dataset not found for {args.model}")
    
    elif args.action == 'visualize':
        # First analyze, then visualize
        results = analyze_dataset(args.model, args.input)
        visualize_dataset_statistics(results, args.output)
    
    print("\n" + "=" * 60)
    print("✅ Dataset utilities completed!")


if __name__ == "__main__":
    main()