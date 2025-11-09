#!/usr/bin/env python3
"""
MENINGIOMA Tumor Detection with GPU Training
Complete PyTorch implementation for brain tumor classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
# UPDATE THIS PATH TO YOUR DATASET LOCATION
DATASET_PATH = r"C:\Users\Melroy Quadros\Brain-Tumor-Classification-DataSet-master"
# Or use: DATASET_PATH = "path/to/your/dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 0.001

class BrainTumorDataset(Dataset):
    """Custom Dataset for Brain Tumor Classification"""
    
    def __init__(self, data_dir, split='Training', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images and labels
        self._load_data()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"✅ Loaded {len(self.images)} images from {split}")
        print(f"   Classes: {self.label_encoder.classes_}")
        
    def _load_data(self):
        """Load image paths and labels"""
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_file)
                        self.images.append(img_path)
                        self.labels.append(class_name)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.encoded_labels[idx]
        return image, label
    
    def get_class_names(self):
        return self.label_encoder.classes_

class BrainTumorClassifier(nn.Module):
    """ResNet-based Brain Tumor Classifier"""
    
    def __init__(self, num_classes):
        super(BrainTumorClassifier, self).__init__()
        
        # Use pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def setup_gpu():
    """Setup GPU and check availability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print("🚀 GPU SETUP")
    print("=" * 50)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        print("✅ GPU acceleration enabled!")
    else:
        print("⚠️ Using CPU (slower training)")
    
    return device

def create_data_loaders():
    """Create training and testing data loaders"""
    print("\n📂 LOADING DATASET")
    print("=" * 30)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BrainTumorDataset(DATASET_PATH, split='Training', transform=train_transform)
    test_dataset = BrainTumorDataset(DATASET_PATH, split='Testing', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    class_names = train_dataset.get_class_names()
    num_classes = len(class_names)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    return train_loader, test_loader, class_names, num_classes

def train_model(model, train_loader, test_loader, device, num_classes):
    """Train the model with GPU acceleration"""
    print(f"\n🔥 TRAINING MODEL")
    print("=" * 30)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training history
    train_losses = []
    train_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100*correct/total:.2f}%")
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation
        val_acc = evaluate_model(model, test_loader, device, verbose=False)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} "
              f"Train Acc: {epoch_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}% "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_meningioma_model.pth')
            print(f"✅ New best model saved! Accuracy: {best_accuracy:.2f}%")
        
        # Early stopping if accuracy > 95%
        if val_acc >= 95.0:
            print(f"🎯 Target accuracy reached: {val_acc:.2f}%")
            break
    
    return train_losses, train_accuracies, best_accuracy

def evaluate_model(model, test_loader, device, verbose=True):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    if verbose:
        print(f"\n📊 FINAL EVALUATION")
        print("=" * 30)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Correct: {correct}/{total}")
    
    return accuracy

def main():
    """Main training pipeline"""
    print("🧠 MENINGIOMA TUMOR DETECTION TRAINING")
    print("=" * 50)
    
    # Setup
    device = setup_gpu()
    train_loader, test_loader, class_names, num_classes = create_data_loaders()
    
    # Create model
    print(f"\n🏗️ BUILDING MODEL")
    print("=" * 30)
    model = BrainTumorClassifier(num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model loaded on: {device}")
    
    # Train model
    train_losses, train_accuracies, best_accuracy = train_model(
        model, train_loader, test_loader, device, num_classes
    )
    
    # Load best model and final evaluation
    model.load_state_dict(torch.load('best_meningioma_model.pth'))
    final_accuracy = evaluate_model(model, test_loader, device, verbose=True)
    
    # Results summary
    print(f"\n🎯 TRAINING COMPLETE!")
    print("=" * 30)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Model saved as: best_meningioma_model.pth")
    
    if final_accuracy >= 95.0:
        print("🏆 SUCCESS! Target accuracy (95%+) achieved!")
    else:
        print(f"📈 Close! Need {95.0 - final_accuracy:.2f}% more for target")
    
    # GPU memory summary
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()