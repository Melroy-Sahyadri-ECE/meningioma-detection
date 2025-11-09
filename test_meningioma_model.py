#!/usr/bin/env python3
"""
Test MENINGIOMA Detection Model
Comprehensive testing and evaluation of the trained model
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os
import time

# Configuration
# UPDATE THIS PATH TO YOUR DATASET LOCATION
DATASET_PATH = r"C:\Users\Melroy Quadros\Brain-Tumor-Classification-DataSet-master"
# Or use: DATASET_PATH = "path/to/your/dataset"

MODEL_PATH = "best_meningioma_model.pth"
IMAGE_SIZE = 224

class BrainTumorClassifier(nn.Module):
    """ResNet-based Brain Tumor Classifier"""
    
    def __init__(self, num_classes):
        super(BrainTumorClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, num_classes, device):
    """Load the trained model"""
    print("🔄 Loading trained model...")
    model = BrainTumorClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and convert image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def predict_single_image(model, image_tensor, device, class_names):
    """Make prediction on a single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score, probabilities.cpu().numpy()[0]

def test_on_dataset(model, device):
    """Test model on the entire test dataset"""
    print("\n🧪 COMPREHENSIVE DATASET TESTING")
    print("=" * 50)
    
    # Load test data
    test_dir = os.path.join(DATASET_PATH, "Testing")
    all_predictions = []
    all_labels = []
    all_confidences = []
    class_names = []
    
    # Get class names
    for class_name in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, class_name)):
            class_names.append(class_name)
    
    class_names.sort()  # Ensure consistent ordering
    print(f"Classes found: {class_names}")
    
    # Test each image
    total_images = 0
    correct_predictions = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        class_correct = 0
        class_total = 0
        
        print(f"\n📁 Testing {class_name}...")
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # Preprocess and predict
                    image_tensor, _ = preprocess_image(img_path)
                    predicted_class, confidence, probabilities = predict_single_image(
                        model, image_tensor, device, class_names
                    )
                    
                    # Store results
                    all_predictions.append(predicted_class)
                    all_labels.append(class_name)
                    all_confidences.append(confidence)
                    
                    # Count accuracy
                    total_images += 1
                    class_total += 1
                    if predicted_class == class_name:
                        correct_predictions += 1
                        class_correct += 1
                
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        
        # Class-specific accuracy
        class_accuracy = (class_correct / class_total) * 100 if class_total > 0 else 0
        print(f"   {class_name}: {class_correct}/{class_total} ({class_accuracy:.2f}%)")
    
    # Overall results
    overall_accuracy = (correct_predictions / total_images) * 100
    
    print(f"\n📊 OVERALL RESULTS")
    print("=" * 30)
    print(f"Total Images Tested: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Average Confidence: {np.mean(all_confidences):.2f}%")
    
    return all_predictions, all_labels, all_confidences, class_names

def generate_detailed_report(predictions, labels, confidences, class_names):
    """Generate detailed classification report"""
    print(f"\n📋 DETAILED CLASSIFICATION REPORT")
    print("=" * 50)
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=class_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - MENINGIOMA Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Number of Predictions')
    
    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(class_names):
        class_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == class_name]
        plt.hist(class_confidences, bins=15, alpha=0.6, label=class_name)
    plt.title('Confidence by Class')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_sample_images(model, device, class_names, num_samples=6):
    """Test and visualize predictions on sample images"""
    print(f"\n🖼️ SAMPLE IMAGE PREDICTIONS")
    print("=" * 40)
    
    test_dir = os.path.join(DATASET_PATH, "Testing")
    sample_images = []
    
    # Collect sample images from each class
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Take first few images from each class
        for img_file in images[:num_samples//len(class_names)]:
            img_path = os.path.join(class_dir, img_file)
            sample_images.append((img_path, class_name))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (img_path, true_label) in enumerate(sample_images[:6]):
        # Load and preprocess image
        image_tensor, original_image = preprocess_image(img_path)
        
        # Make prediction
        predicted_class, confidence, probabilities = predict_single_image(
            model, image_tensor, device, class_names
        )
        
        # Display image
        axes[i].imshow(original_image)
        
        # Create title with prediction info
        color = 'green' if predicted_class == true_label else 'red'
        title = f"True: {true_label}\nPred: {predicted_class}\nConf: {confidence:.1f}%"
        axes[i].set_title(title, color=color, fontweight='bold')
        axes[i].axis('off')
        
        # Print detailed results
        print(f"\nImage {i+1}: {os.path.basename(img_path)}")
        print(f"  True Label: {true_label}")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Probabilities: {dict(zip(class_names, probabilities))}")
    
    plt.suptitle('Sample Predictions - MENINGIOMA Detection Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def benchmark_speed(model, device):
    """Benchmark model inference speed"""
    print(f"\n⚡ SPEED BENCHMARK")
    print("=" * 30)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Images per second: {1/avg_time:.1f}")
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

def main():
    """Main testing pipeline"""
    print("🧪 MENINGIOMA DETECTION MODEL TESTING")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please run the training script first!")
        return
    
    # Load model (assuming 2 classes based on training output)
    class_names = ['meningioma_tumor', 'no_tumor']
    model = load_model(MODEL_PATH, len(class_names), device)
    
    # Run comprehensive tests
    predictions, labels, confidences, class_names = test_on_dataset(model, device)
    
    # Generate detailed reports
    generate_detailed_report(predictions, labels, confidences, class_names)
    
    # Test sample images with visualization
    test_sample_images(model, device, class_names)
    
    # Speed benchmark
    benchmark_speed(model, device)
    
    print(f"\n🎉 TESTING COMPLETE!")
    print("=" * 30)
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - confidence_analysis.png") 
    print("  - sample_predictions.png")

if __name__ == "__main__":
    main()