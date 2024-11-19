import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_height = 512
img_width = 512
batch_size = 32
epochs = 300
base_dir = "C:/Users/VERMA/Downloads/dataset_inner"

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset(base_dir):
    images = []
    labels = []
    
    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if os.path.isdir(label_path):
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                if os.path.isfile(img_path):
                    images.append(img_path)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

# Loading dataset
print("Loading dataset...")
images, labels = load_dataset(base_dir)
print(f"Loaded {len(images)} images from {len(np.unique(labels))} classes")

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels_encoded, test_size=0.3, stratify=labels_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_dataset = CustomDataset(X_train, y_train, transform=transform)
val_dataset = CustomDataset(X_val, y_val, transform=transform)
test_dataset = CustomDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, N, C = x.shape  
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x

        # MLP
        x = self.norm2(x)
        x = self.mlp(x) + x

        return x

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=4, stride=4)  
        self.pool1 = nn.MaxPool2d(2)
        self.transformer_block = SwinTransformerBlock(96, num_heads=4, window_size=4)
        self.fc1 = nn.Linear(96, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.flatten(2).transpose(1, 2)  
        x = self.transformer_block(x)  
        x = x.mean(dim=1)  
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  
        return x

# Save and load functions for the model
def save_model_to_h5(model, filepath):
    with h5py.File(filepath, 'w') as f:
        for param_tensor, param in model.state_dict().items():
            f.create_dataset(param_tensor, data=param.cpu().numpy())
    print(f"Model saved to {filepath}")

def load_model_from_h5(model, filepath):
    with h5py.File(filepath, 'r') as f:
        state_dict = {}
        for param_tensor in f.keys():
            state_dict[param_tensor] = torch.tensor(f[param_tensor][()])
        model.load_state_dict(state_dict)
    print(f"Model loaded from {filepath}")

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Store predictions and true labels for metric calculation
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')  # changed to 'weighted'
    recall = recall_score(all_labels, all_preds, average='weighted')  # changed to 'weighted'
    f1 = f1_score(all_labels, all_preds, average='weighted')  # changed to 'weighted'

    return accuracy, precision, recall, f1, avg_loss, all_labels, all_preds


# Plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.show()

# Visualize predictions
def visualize_predictions(model, data_loader, num_samples=5):
    model.eval()
    images, true_labels = next(iter(data_loader))
    images = images.to(device)
    true_labels = true_labels.numpy()

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy()

    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title(f'True: {le.inverse_transform([true_labels[i]])[0]}\nPred: {le.inverse_transform([preds[i]])[0]}')
        plt.axis('off')

    plt.show()

# Training loop with pseudo-labeling
def train_with_pseudo_labels(model, train_loader, unlabeled_loader, val_loader, criterion, optimizer, epochs=epochs, patience=20, threshold=0.95):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # Training on labeled data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_preds / total_preds)

        # Validation phase
        val_accuracy, val_precision, val_recall, val_f1, val_loss, _, _ = compute_metrics(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]}")
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model_to_h5(model, 'best_model_bone.h5')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping...")
            break

    load_model_from_h5(model, 'best_model_bone.h5')
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies

# Function to plot history of accuracy and loss
def plot_history(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    plt.legend()

    plt.show()

# Model evaluation on all datasets
model = SwinTransformerModel(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, train_losses, train_accuracies, val_losses, val_accuracies = train_with_pseudo_labels(
    model, train_loader, test_loader, val_loader, criterion, optimizer
)

# Training Metrics and Confusion Matrix
print("\nTraining Metrics:")
train_accuracy, train_precision, train_recall, train_f1, train_loss, train_labels, train_preds = compute_metrics(model, train_loader, criterion)
print(f"Train Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}, Loss: {train_loss}")
plot_confusion_matrix(train_labels, train_preds, 'Train')

# Validation Metrics and Confusion Matrix
print("\nValidation Metrics:")
val_accuracy, val_precision, val_recall, val_f1, val_loss, val_labels, val_preds = compute_metrics(model, val_loader, criterion)
print(f"Validation Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, Loss: {val_loss}")
plot_confusion_matrix(val_labels, val_preds, 'Validation')

# Test Metrics and Confusion Matrix
print("\nTest Metrics:")
test_accuracy, test_precision, test_recall, test_f1, test_loss, test_labels, test_preds = compute_metrics(model, test_loader, criterion)
print(f"Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}, Loss: {test_loss}")
plot_confusion_matrix(test_labels, test_preds, 'Test')


visualize_predictions(model, val_loader)
visualize_predictions(model, test_loader)

# History plot
plot_history(train_losses, train_accuracies, val_losses, val_accuracies)