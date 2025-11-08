# MENINGIOMA Tumor Detection System 🧠

A high-performance deep learning system for detecting MENINGIOMA brain tumors using PyTorch and GPU acceleration.

## 🎯 Performance

- **Accuracy**: 96.82%
- **MENINGIOMA Detection**: 98.26% accuracy
- **No Tumor Detection**: 95.24% accuracy
- **Inference Speed**: 186.8 images/second
- **Average Confidence**: 94.25%

## 🚀 Features

- ✅ GPU-accelerated training with CUDA support
- ✅ Transfer learning using ResNet50
- ✅ Real-time inference (5.35ms per image)
- ✅ Comprehensive evaluation metrics
- ✅ Data augmentation for improved accuracy
- ✅ Medical-grade performance (>95% accuracy)

## 📋 Requirements

```
Python 3.8+
PyTorch 2.5.1+
CUDA 12.1+ (for GPU support)
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/meningioma-detection.git
cd meningioma-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📁 Project Structure

```
meningioma-detection/
├── meningioma_detection_gpu.py    # Main training script
├── test_meningioma_model.py       # Comprehensive testing script
├── gpu_setup.py                   # GPU configuration utility
├── pytorch_gpu_test.py            # GPU verification script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
├── models/                        # Saved model files
│   └── best_meningioma_model.pth
├── outputs/                       # Generated outputs
│   ├── confusion_matrix.png
│   ├── confidence_analysis.png
│   └── sample_predictions.png
└── data/                          # Dataset directory
    ├── Training/
    │   ├── meningioma_tumor/
    │   └── no_tumor/
    └── Testing/
        ├── meningioma_tumor/
        └── no_tumor/
```

## 🎓 Usage

### Training the Model

```bash
python meningioma_detection_gpu.py
```

**Training Configuration:**
- Image Size: 224x224
- Batch Size: 32
- Epochs: 50 (with early stopping)
- Learning Rate: 0.001
- Optimizer: Adam with weight decay

### Testing the Model

```bash
python test_meningioma_model.py
```

**Test Outputs:**
- Detailed classification report
- Confusion matrix visualization
- Confidence score analysis
- Sample prediction images
- Speed benchmarks

### GPU Setup Verification

```bash
python gpu_setup.py
```

## 📊 Results

### Classification Report

```
                  precision    recall  f1-score   support

meningioma_tumor     0.9576    0.9826    0.9700       115
        no_tumor     0.9804    0.9524    0.9662       105

        accuracy                         0.9682       220
```

### Performance Metrics

| Metric | MENINGIOMA | No Tumor | Overall |
|--------|------------|----------|---------|
| Accuracy | 98.26% | 95.24% | **96.82%** |
| Precision | 95.76% | 98.04% | 96.90% |
| Recall | 98.26% | 95.24% | 96.75% |
| F1-Score | 97.00% | 96.62% | 96.81% |

### Speed Benchmarks

- **Inference Time**: 5.35ms per image
- **Throughput**: 186.8 images/second
- **GPU Memory**: 0.10 GB
- **Training Time**: ~90 seconds (3 epochs)

## 🏗️ Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned last 20 layers
- **Custom Head**: 
  - Dropout(0.5)
  - Linear(2048 → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 2)

## 🔬 Data Augmentation

- Random rotation (±15 degrees)
- Random horizontal flip (50% probability)
- Color jitter (brightness & contrast ±20%)
- ImageNet normalization

## 💻 Hardware Requirements

### Minimum
- CPU: Multi-core processor
- RAM: 8GB
- Storage: 5GB

### Recommended (for GPU training)
- GPU: NVIDIA GPU with CUDA support (6GB+ VRAM)
- RAM: 16GB
- Storage: 10GB SSD

## 📈 Training Process

1. **Data Loading**: Loads training and testing datasets
2. **Preprocessing**: Resizes images to 224x224, applies augmentation
3. **Model Initialization**: Loads pre-trained ResNet50
4. **Training**: Fine-tunes model with early stopping
5. **Validation**: Evaluates on test set after each epoch
6. **Model Saving**: Saves best model based on validation accuracy

## 🧪 Testing & Evaluation

The testing script provides:
- Comprehensive dataset evaluation
- Per-class accuracy metrics
- Confusion matrix visualization
- Confidence score distribution
- Sample prediction visualization
- Speed benchmarking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet50 architecture from torchvision
- Brain tumor dataset for training and evaluation
- PyTorch team for the excellent deep learning framework

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

## ⚠️ Disclaimer

This system is designed for research and educational purposes. It should not be used as the sole basis for medical diagnosis. Always consult with qualified medical professionals for clinical decisions.

## 🔮 Future Improvements

- [ ] Multi-class tumor classification (glioma, pituitary, etc.)
- [ ] Web interface for easy deployment
- [ ] DICOM image support
- [ ] Explainable AI visualizations (Grad-CAM)
- [ ] Model quantization for edge deployment
- [ ] REST API for integration

---

**Made with ❤️ for advancing medical AI**
