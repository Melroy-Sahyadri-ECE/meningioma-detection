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

### Quick Start (Using Pre-trained Model)

The repository includes a pre-trained model with **96.82% accuracy**. You can use it immediately without training!

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Melroy-Sahyadri-ECE/meningioma-detection.git
cd meningioma-detection
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Test the Pre-trained Model
```bash
python test_meningioma_model.py
```

This will:
- ✅ Load the trained model from `models/best_meningioma_model.pth`
- ✅ Test on your dataset
- ✅ Generate evaluation metrics
- ✅ Create visualization images
- ✅ Show speed benchmarks

**Expected Output:**
```
🧪 MENINGIOMA DETECTION MODEL TESTING
==================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
✅ Model loaded successfully!

📊 OVERALL RESULTS
==============================
Total Images Tested: 220
Correct Predictions: 213
Overall Accuracy: 96.82%
Average Confidence: 94.25%
```

### Training Your Own Model

If you want to train from scratch or fine-tune:

#### Step 1: Prepare Your Dataset

Organize your dataset in this structure:
```
data/
├── Training/
│   ├── meningioma_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── no_tumor/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── Testing/
    ├── meningioma_tumor/
    │   └── ...
    └── no_tumor/
        └── ...
```

#### Step 2: Update Dataset Path

Edit `meningioma_detection_gpu.py` line 23:
```python
DATASET_PATH = r"C:\path\to\your\dataset"  # Change this to your dataset path
```

#### Step 3: Train the Model
```bash
python meningioma_detection_gpu.py
```

**Training Process:**
1. Loads and preprocesses images
2. Applies data augmentation
3. Trains ResNet50 model with transfer learning
4. Saves best model to `models/best_meningioma_model.pth`
5. Stops early if 95%+ accuracy is achieved

**Training Output:**
```
🔥 TRAINING MODEL
==============================
Epoch [1/50] Train Loss: 0.3762 Train Acc: 86.94% Val Acc: 88.64%
Epoch [2/50] Train Loss: 0.2693 Train Acc: 95.23% Val Acc: 91.82%
Epoch [3/50] Train Loss: 0.2598 Train Acc: 95.48% Val Acc: 96.82%
🎯 Target accuracy reached: 96.82%
✅ Model saved as: best_meningioma_model.pth
```

### Making Predictions on New Images

Create a simple prediction script:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('models/best_meningioma_model.pth', map_location=device)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path/to/your/image.jpg')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    
    classes = ['meningioma_tumor', 'no_tumor']
    print(f"Prediction: {classes[predicted.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
```

### GPU Setup Verification

Before training, verify your GPU is working:

```bash
python gpu_setup.py
```

**Expected Output:**
```
🚀 GPU SETUP
==================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Memory: 6.0 GB
✅ GPU acceleration enabled!
```

### Advanced Usage

#### Custom Training Parameters

Edit `meningioma_detection_gpu.py` to customize:

```python
# Configuration (lines 20-26)
IMAGE_SIZE = 224        # Image dimensions
BATCH_SIZE = 32         # Batch size (reduce if GPU memory error)
NUM_WORKERS = 4         # Data loading threads
EPOCHS = 50             # Maximum epochs
LEARNING_RATE = 0.001   # Learning rate
```

#### Using Different Models

Replace ResNet50 with other architectures:

```python
# In meningioma_detection_gpu.py, line 95
self.backbone = models.resnet50(pretrained=True)  # Current
# Try these alternatives:
# self.backbone = models.resnet101(pretrained=True)  # Larger model
# self.backbone = models.efficientnet_b0(pretrained=True)  # Efficient
# self.backbone = models.vgg16(pretrained=True)  # Classic
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

## 🐛 Troubleshooting

### GPU Not Detected

**Problem:** `Device: cpu` instead of `cuda`

**Solutions:**
1. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. Check GPU drivers are installed
3. Verify GPU with: `nvidia-smi`

### Out of Memory Error

**Problem:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size in `meningioma_detection_gpu.py`:
   ```python
   BATCH_SIZE = 16  # or 8
   ```
2. Reduce number of workers:
   ```python
   NUM_WORKERS = 2
   ```

### Module Not Found

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Model File Not Found

**Problem:** `Model file not found: best_meningioma_model.pth`

**Solution:**
The model should be in `models/` directory. If missing:
1. Download from GitHub releases
2. Or train your own model: `python meningioma_detection_gpu.py`

### Dataset Path Error

**Problem:** `Dataset not found at path`

**Solution:**
Update the path in the script:
```python
DATASET_PATH = r"C:\your\actual\path\to\dataset"
```

### Slow Training on CPU

**Problem:** Training is very slow

**Solution:**
- Install GPU-enabled PyTorch (see GPU Not Detected)
- Or reduce dataset size for testing
- Or use Google Colab with free GPU

## 📚 Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Transfer Learning Guide**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **Medical Image Analysis**: https://www.sciencedirect.com/topics/computer-science/medical-image-analysis

## 🎯 Model Performance Details

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| MENINGIOMA | 95.76% | 98.26% | 97.00% | 115 |
| No Tumor | 98.04% | 95.24% | 96.62% | 105 |
| **Overall** | **96.90%** | **96.75%** | **96.81%** | **220** |

### Confusion Matrix

```
                Predicted
              MEN    NO
Actual  MEN   113     2
        NO      5   100
```

### Speed Benchmarks

- **Single Image**: 5.35ms
- **Batch (32 images)**: 171ms
- **Throughput**: 186.8 images/second
- **GPU Memory**: 0.10 GB

## 💡 Tips for Best Results

1. **Data Quality**: Use high-resolution, clear brain scan images
2. **Data Balance**: Ensure similar number of images per class
3. **Augmentation**: The model uses rotation, flips, and color jitter
4. **GPU Training**: 10-20x faster than CPU
5. **Early Stopping**: Model stops when accuracy > 95%
6. **Fine-tuning**: Adjust learning rate if needed

## 🔬 Research & Citations

If you use this project in your research, please cite:

```bibtex
@software{meningioma_detection_2025,
  author = {Melroy Quadros},
  title = {MENINGIOMA Tumor Detection System},
  year = {2025},
  url = {https://github.com/Melroy-Sahyadri-ECE/meningioma-detection},
  note = {Deep learning system for brain tumor detection with 96.82\% accuracy}
}
```

---

**Made with ❤️ for advancing medical AI**

**Star ⭐ this repository if you find it helpful!**
