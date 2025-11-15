import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class config:
    image_size = (256, 256)
    components = 100


def load_rgb_images(folder, label, size=config.image_size):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading RGB from {folder}"):
        if filename.endswith('.jpg') or filename.endswith('.tiff'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB').resize(size)
            images.append(np.array(img).flatten())
            labels.append(label)
    return images, labels


def load_ndvi_images(folder, size=config.image_size):
    images = []
    filenames = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading NDVI from {folder}"):
        if filename.endswith('.jpg') or filename.endswith('.tiff'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(size)
            images.append(np.array(img).flatten())
            filenames.append(filename)
    return images, filenames


# Define paths
label_path = "labels/labels"   # NDVI images
sample_path = "samples/samples" # RGB images

# Load RGB samples (training features)
rgb_images, _ = load_rgb_images(sample_path, label=0)
samples_array = np.array(rgb_images)

# Load NDVI labels (ground truth)
ndvi_images, ndvi_filenames = load_ndvi_images(label_path)
labels_array = np.array(ndvi_images)

mean_ndvi = np.nanmean(labels_array, axis=1)
threshold = np.median(mean_ndvi)

# Binary labels from NDVI
y_labels = (mean_ndvi > threshold).astype(int)

print("Ground Truth Label Distribution (from NDVI):")
print(f"Class 0 (Unhealthy): {np.sum(y_labels == 0)} images")
print(f"Class 1 (Healthy):   {np.sum(y_labels == 1)} images")
print(f"\nNDVI Threshold: {threshold:.4f}")
print(f"Mean NDVI range: {mean_ndvi.min():.4f} to {mean_ndvi.max():.4f}")

X = samples_array
y = y_labels

print(f"Training data (RGB flattened): {X.shape}")
print(f"Ground truth labels (NDVI):   {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} images")
print(f"Test set:  {X_test.shape[0]} images")

# PCA
pca = PCA(n_components=config.components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Original shape (before PCA):", X_train.shape)
print("Shape after PCA:", X_train_pca.shape)

# # SVM model
# svm = SVC(kernel='rbf', probability=True, random_state=42)
# svm.fit(X_train_pca, y_train)
#
# y_pred = svm.predict(X_test_pca)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# # SVM decision boundary visualization (2D PCA)
# clf = SVC(kernel='linear', random_state=42)
# clf.fit(X_train_pca[:, :2], y_train)
#
# X_2d = X_train_pca[:, :2]
# x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
# y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
#
# h = (x_max - x_min) / 200
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# plt.figure(figsize=(11, 9))
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50),
#              cmap='coolwarm', alpha=0.3)
# plt.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1],
#             linewidths=[2, 3, 2], linestyles=['--', '-', '--'])
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='coolwarm',
#             s=35, edgecolors='white', linewidths=0.6, alpha=0.9)
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=220, linewidth=2, facecolors='none',
#             edgecolors='darkblue', label='Support Vectors')
#
# plt.xlabel('Principal Component 1', fontsize=13, fontweight='bold')
# plt.ylabel('Principal Component 2', fontsize=13, fontweight='bold')
# plt.title('Linear SVM Decision Regions (2D PCA Projection)',
#           fontsize=16, fontweight='bold', pad=15)
#
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
#
# legend_elements = [
#     Patch(facecolor='#f4a582', edgecolor='black', label='Unhealthy (0)'),
#     Patch(facecolor='#92c5de', edgecolor='black', label='Healthy (1)'),
#     Line2D([0], [0], color='black', linewidth=3, label='Decision Boundary'),
#     Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Margins (±1)'),
#     Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
#            markeredgecolor='darkblue', markersize=12, markeredgewidth=2, label='Support Vectors')
# ]
#
# plt.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()
#
# # Summary stats
# theta = clf.coef_[0]
# theta0 = clf.intercept_[0]
# print("=" * 75)
# print("SVM (2D PCA Projection) Summary")
# print("=" * 75)
# print(f"Support vectors: {len(clf.support_vectors_)}")
# print(f"Class counts: {clf.n_support_}")
# print(f"Margin width: {2 / np.linalg.norm(theta):.4f}")
# print(f"Hyperplane: {theta[0]:.4f}*PC1 + {theta[1]:.4f}*PC2 + {theta0:.4f} = 0")
#
# y_pred_2d = clf.predict(X_2d)
# acc_2d = np.mean(y_pred_2d == y_train)
# print(f"Training accuracy (2D PCA): {acc_2d:.2%}")
# print(f"Variance explained (2 PCs): {np.sum(pca.explained_variance_ratio_[:2]) * 100:.2f}%")
# print("=" * 75)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 output classes (healthy/unhealthy)
        )

    def forward(self, x):
      return self.linear_relu_stack(x)


print("Running NN")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_size=config.components).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


from sklearn.metrics import classification_report

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report (Neural Network):")
print(classification_report(y_true, y_pred))
