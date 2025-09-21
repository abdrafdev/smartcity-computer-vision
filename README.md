# 🏙️ AI for Smart Cities – Computer Vision Applications

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements **three computer vision applications** for smart city infrastructure using **YOLOv8** object detection models. The system addresses critical urban challenges through automated monitoring and detection of:

1. **🗑️ Garbage Detection** - Automated waste management and environmental monitoring
2. **⛑️ Helmet Detection** - Workplace safety compliance in construction zones  
3. **🚦 Traffic Violation Detection** - Smart traffic monitoring and law enforcement

### 🎯 Key Features

- **Multiple Detection Models**: Separate YOLOv8 models trained for different smart city applications
- **Unified Interface**: Single demo script handles all model types and input sources
- **Real-time Processing**: Support for images, videos, and live webcam feeds
- **Production Ready**: Clean, modular code with comprehensive documentation
- **Offline Capable**: Runs entirely offline without cloud dependencies
- **Portfolio Quality**: Professional-grade implementation suitable for showcasing

## 🏗️ Project Structure

```
smartcity-computer-vision/
├── 📁 garbage-detection/
│   ├── data/
│   │   ├── train/          # Training images and labels
│   │   ├── valid/          # Validation images and labels  
│   │   └── test/           # Test images and labels
│   └── garbage_config.yaml # Dataset configuration
├── 📁 helmet-detection/
│   ├── data/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── helmet_config.yaml
├── 📁 traffic-detection/
│   ├── data/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── traffic_config.yaml
├── 📁 models/              # Trained model weights
│   ├── garbage/
│   ├── helmet/
│   └── traffic/
├── 📁 utils/               # Utility functions
│   ├── __init__.py
│   └── common.py
├── 📁 notebooks/           # Jupyter notebooks for analysis
├── 🐍 garbage_train.py     # Garbage detection training
├── 🐍 helmet_train.py      # Helmet detection training
├── 🐍 traffic_train.py     # Traffic detection training
├── 🐍 demo.py              # Unified inference demo
├── 📄 requirements.txt     # Python dependencies
└── 📖 README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd smartcity-computer-vision

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Datasets

Each detection type requires a YOLO-format dataset. Place your data in the following structure:

```
{model-name}-detection/data/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Training labels (.txt)
├── valid/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── test/
    ├── images/     # Test images
    └── labels/     # Test labels
```

### 3. Train Models

Train each model separately using the provided training scripts:

```bash
# Train garbage detection model
python garbage_train.py

# Train helmet detection model  
python helmet_train.py

# Train traffic detection model
python traffic_train.py
```

### 4. Run Inference

Use the unified demo script to test your trained models:

```bash
# Image inference
python demo.py --model garbage --source test_image.jpg
python demo.py --model helmet --source safety_check.png
python demo.py --model traffic --source street_scene.jpg

# Video inference
python demo.py --model helmet --source safety_video.mp4 --save
python demo.py --model traffic --source traffic_cam.avi --conf 0.6

# Webcam inference
python demo.py --model garbage --source 0
python demo.py --model helmet --source webcam --conf 0.7
```

## 📊 Dataset Information

### 🗑️ Garbage Detection Dataset
- **Classes**: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- **Application**: Automated waste sorting and environmental monitoring
- **Use Cases**: Smart bins, recycling facilities, street cleaning optimization

### ⛑️ Helmet Detection Dataset  
- **Classes**: `helmet`, `no-helmet`, `person`
- **Application**: Construction site safety compliance monitoring
- **Use Cases**: Workplace safety, regulatory compliance, accident prevention

### 🚦 Traffic Violation Detection Dataset
- **Classes**: `car`, `motorcycle`, `bus`, `truck`, `traffic-light`, `stop-sign`, `person`, `bicycle`, `violation`
- **Application**: Smart traffic management and violation detection
- **Use Cases**: Automated traffic enforcement, intersection monitoring, pedestrian safety

## 🛠️ Training Configuration

### Model Architecture
- **Garbage Detection**: YOLOv8 Nano (fast, lightweight)
- **Helmet Detection**: YOLOv8 Small (balanced speed/accuracy)  
- **Traffic Detection**: YOLOv8 Medium (high accuracy for complex scenes)

### Training Parameters
- **Image Size**: 640×640 pixels
- **Batch Size**: 12-16 (depending on model complexity)
- **Epochs**: 100-200 (with early stopping)
- **Optimizer**: AdamW with cosine learning rate scheduling
- **Data Augmentation**: HSV, rotation, translation, scaling, flipping

### Hardware Requirements
- **Minimum**: 8GB RAM, integrated GPU
- **Recommended**: 16GB RAM, dedicated GPU (GTX 1060 or better)
- **Training Time**: 1-4 hours per model (depending on dataset size and hardware)

## 🎮 Demo Usage

### Command Line Options

```bash
python demo.py --help
```

**Required Arguments:**
- `--model {garbage,helmet,traffic}`: Model type to use
- `--source PATH`: Input source (image, video, webcam)

**Optional Arguments:**
- `--weights PATH`: Custom model weights path
- `--conf FLOAT`: Confidence threshold (default: 0.5)
- `--iou FLOAT`: IoU threshold for NMS (default: 0.45)
- `--save`: Save inference results
- `--show`: Display results (default: True)
- `--output PATH`: Output directory (default: runs/inference)
- `--img-size INT`: Inference image size (default: 640)

### Usage Examples

#### Image Processing
```bash
# Basic image inference with default settings
python demo.py --model garbage --source waste_pile.jpg

# High confidence threshold for precise detections
python demo.py --model helmet --source construction_site.png --conf 0.8

# Save results to custom directory
python demo.py --model traffic --source intersection.jpg --save --output results/traffic/
```

#### Video Processing
```bash
# Process video file with visualization
python demo.py --model helmet --source safety_footage.mp4 --show

# Process and save video results
python demo.py --model traffic --source dashcam_video.avi --save --conf 0.6

# Batch process without display (faster)
python demo.py --model garbage --source timelapse.mp4 --save --no-show
```

#### Live Webcam
```bash
# Default webcam (index 0)
python demo.py --model helmet --source 0

# Specific webcam device
python demo.py --model traffic --source 1 --conf 0.7

# Alternative webcam syntax
python demo.py --model garbage --source webcam
```

#### Interactive Controls (Video/Webcam)
- **Q** or **ESC**: Quit inference
- **S**: Save current frame (when --save enabled)

## 📈 Model Performance

### Expected Performance Metrics
- **mAP@0.5**: 0.80+ for well-prepared datasets
- **Inference Speed**: 20-60 FPS (depending on hardware and model size)
- **Model Size**: 6MB (nano) to 52MB (medium)

### Performance Tips
- Use smaller models (nano/small) for real-time applications
- Use larger models (medium/large) for highest accuracy
- Optimize batch size based on available GPU memory
- Use mixed precision training (AMP) for faster training

## 🔧 Troubleshooting

### Common Issues

**1. ImportError: No module named 'ultralytics'**
```bash
pip install ultralytics
```

**2. CUDA out of memory during training**
- Reduce batch size in training scripts
- Use a smaller model variant (nano instead of small)
- Close other GPU-intensive applications

**3. No webcam detected**
- Check webcam index (try 0, 1, 2...)
- Ensure webcam is not being used by other applications
- Install proper webcam drivers

**4. Poor model performance**
- Ensure dataset is properly labeled and balanced
- Increase training epochs
- Verify dataset structure matches YOLO format
- Check class distribution in your dataset

**5. Slow inference speed**
- Use smaller model variants for real-time applications
- Reduce input image size (--img-size parameter)
- Ensure GPU acceleration is working (check CUDA installation)

### Dataset Preparation Tips

1. **Image Quality**: Use high-resolution, clear images
2. **Class Balance**: Ensure roughly equal samples per class
3. **Annotation Quality**: Precise bounding box labeling is crucial
4. **Data Diversity**: Include various lighting, angles, and conditions
5. **Dataset Size**: Minimum 100-500 images per class for good results

## 🏗️ Extension Ideas

### Smart City Applications
- **Parking Violation Detection**: Detect illegally parked vehicles
- **Crowd Density Monitoring**: Monitor pedestrian density in public spaces
- **Infrastructure Damage Detection**: Identify potholes, broken signs, etc.
- **Emergency Response**: Detect accidents, fires, or unusual activities

### Technical Enhancements
- **Multi-Model Ensemble**: Combine predictions from multiple models
- **Tracking Integration**: Add object tracking for video sequences
- **Alert System**: Integrate with notification systems for violations
- **Database Integration**: Store detection results in databases
- **Web Dashboard**: Create web interface for monitoring and analytics

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For questions, issues, or suggestions:
- 📧 Create an issue in the repository
- 💬 Contact the development team
- 📖 Check the documentation and examples

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 implementation
- **OpenCV**: For computer vision utilities
- **PyTorch**: For deep learning framework
- **Smart City Research Community**: For inspiration and use cases

---

**Made with ❤️ for Smart Cities and Computer Vision**

*This project demonstrates practical applications of AI in urban environments, contributing to safer, cleaner, and more efficient cities.*