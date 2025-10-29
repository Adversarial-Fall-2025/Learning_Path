SVM (PyTorch) for Vegetation Classification

What this does
- Builds a simple SVM-like linear classifier using PyTorch to classify tiles into 3 vegetation classes
  based on NDVI labels and image statistics extracted from sample GeoTIFF tiles.

Assumptions
- `samples/` contains image GeoTIFF tiles named like `*_img_{i}.tiff`.
- `labels/` contains NDVI GeoTIFF tiles named like `*_ndvi_{i}.tiff` and aligned to sample indices.
- Tiles are paired by the trailing numeric index before `.tiff`.
- NDVI thresholds map continuous NDVI to 3 classes: non-veg, sparse veg, dense veg (defaults 0.2 and 0.5).

Files
- `train_svm_pytorch.py`: training script. Saves `outputs/svm_model.pth` by default.
- `requirements.txt`: Python dependencies.

Quick start (Windows PowerShell)

# Create venv and install deps
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r Task_1\requirements.txt

# Run training (from repo root)
python Task_1\train_svm_pytorch.py --data_dir Task_1 --samples_dir samples --labels_dir labels --out_dir outputs

Notes and next steps
- Current feature extractor uses mean and std per band. For better accuracy, consider using CNN features
  or texture features.
- The NDVI-to-class thresholds are heuristic; tune them or use labeled classes if available.
- If you prefer a classical SVM, you can extract features (using the same script) and train scikit-learn's
  SVC instead.
