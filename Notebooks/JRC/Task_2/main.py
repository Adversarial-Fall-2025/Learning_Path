import os
import glob
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# find mean of the pixels (rgb)
# use labels


# -------------------------------
# Paths
# -------------------------------
SAMPLES_PATH = "samples/samples/"
LABELS_PATH = "labels/labels/"

# Get all .tiff files sorted alphabetically
sample_files = sorted(glob.glob(os.path.join(SAMPLES_PATH, "*.tiff")))
label_files = sorted(glob.glob(os.path.join(LABELS_PATH, "*.tiff")))

print(f"Found {len(sample_files)} sample tiles and {len(label_files)} label tiles.")

# -------------------------------
# Load and Prepare Data
# -------------------------------
X_list, y_list = [], []

for sample_path, label_path in zip(sample_files, label_files):
    with rasterio.open(sample_path) as s, rasterio.open(label_path) as l:
        sample = s.read()   # (bands, height, width)
        label = l.read(1)   # (height, width)

        n_bands, height, width = sample.shape
        X = sample.reshape(n_bands, -1).T
        y = label.flatten()

        # Mask out invalid/no-data pixels
        mask = (y > 0) & ~np.isnan(y)
        X_list.append(X[mask])
        y_list.append(y[mask])

# Combine all tiles
X = np.vstack(X_list)
y = np.hstack(y_list)
print("Combined dataset shape:", X.shape, y.shape)

# -------------------------------
# Filter rare classes
# -------------------------------
class_counts = Counter(y)
min_samples = 500   # remove classes with fewer pixels
valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]
print(f"\n✅ After filtering: {len(valid_classes)} valid classes, {len(y)} total samples")

# -------------------------------
# Optional: Take top N classes
# -------------------------------
top_n = 25
top_classes = [cls for cls, _ in Counter(y).most_common(top_n)]
mask = np.isin(y, top_classes)
X = X[mask]
y = y[mask]
print(f"\n✅ Using top {top_n} classes, total samples: {len(y)}")

# -------------------------------
# Normalize features
# -------------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Subsample per class for faster training
# -------------------------------
subset_size_per_class = 10000
X_sub_list, y_sub_list = [], []

for cls in np.unique(y):
    cls_indices = np.where(y == cls)[0]
    chosen = cls_indices if len(cls_indices) <= subset_size_per_class else np.random.choice(cls_indices, subset_size_per_class, replace=False)
    X_sub_list.append(X_scaled[chosen])
    y_sub_list.append(y[chosen])

X_sub = np.vstack(X_sub_list)
y_sub = np.hstack(y_sub_list)
print(f"\n✅ Subsampled dataset shape: {X_sub.shape}, {len(y_sub)} samples")

# -------------------------------
# Split train/test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
)

# -------------------------------
# Train Random Forest
# -------------------------------
svm = SVC(kernel='linear',C=10,  decision_function_shape='ovr')
svm.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = svm.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# -------------------------------
# Optional: Predict first tile
# -------------------------------
with rasterio.open(sample_files[0]) as src:
    sample = src.read()
    n_bands, height, width = sample.shape
    X_tile = sample.reshape(n_bands, -1).T
    X_tile_scaled = scaler.transform(X_tile)
    pred_tile = svm.predict(X_tile_scaled).reshape(height, width)

plt.figure(figsize=(8,6))
plt.imshow(pred_tile, cmap='viridis')
plt.title("Random Forest Vegetation Classification")
plt.colorbar()
plt.show()



