import numpy as np
import tifffile
import os

sample_folder = "samples"
sample = [i for i in os.listdir(sample_folder) if i.endswith(".tiff") or i.endswith(".tif")]
label_folder = "labels"
labels = [h for h in os.listdir(label_folder) if h.endswith(".tiff") or h.endswith(".tif")]

def image_preprocess(sample, label):
	processedSamples = []
	processedLabels = []
	differentSizeSamples = []
	differentSizeLabels = []
	for img in sample: #Verificacion para tamaños de samples
		s = tifffile.imread(f"samples/{img}")
		if s.shape == (256,256,3):
			processedSamples.append(img)
		else:
			differentSizeSamples.append(img)
	for img in label: #Verificaion para tamaños de labels
		l = tifffile.imread(f"labels/{img}")
		if l.shape == (256,256):
			processedLabels.append(img)
		else:
			differentSizeLabels.append(img)
	print(f"Processed Samples: {len(processedSamples)}")
	print(f"Processed Labels: {len(processedLabels)}")
	print(f"Different Sized Samples: {len(differentSizeSamples)}")		
	print(f"Different Sized Labels: {len(differentSizeLabels)}")

image_preprocess(sample, labels)

def min_max_finder(labels):
	min_val = float('inf')
	max_val = float('-inf')

	for label_img in labels:
		img = tifffile.imread(f"{label_folder}/{label_img}")
		current_min = np.min(img)
		current_max = np.max(img)

		if current_min < min_val:
			min_val = current_min
		if current_max > max_val:
			max_val = current_max

	return min_val, max_val

def ndvi_normalization(labels):
#Normalization of NDVI pixels in labels. 
	NDVI_Values = []
	min_val, max_val = min_max_finder(labels)
	
	for label_img in range(5):
		img = tifffile.imread(f"{label_folder}/{labels[label_img]}").astype(float)
		normalized_NDVI = ((img - min_val)/(max_val - min_val)) * 2-1
		normalized_NDVI = np.clip(normalized_NDVI, -1,1)
		
		NDVI_Values.append(normalized_NDVI)
	return NDVI_Values

print(f"NDVI values of the first 5 images: {ndvi_normalization(labels)}")
print("---------------EOF---------------")
