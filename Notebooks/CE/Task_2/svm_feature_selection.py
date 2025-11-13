import numpy as np
import tifffile
import os
#from collections import Counter

sample_folder = "C:/Users/Carlos Escobar/Desktop/CLASSES/PandaHat Adversarial/Task_1/samples"
sample = [i for i in os.listdir(sample_folder) if i.endswith(".tiff") or i.endswith(".tif")]
label_folder = "C:/Users/Carlos Escobar/Desktop/CLASSES/PandaHat Adversarial/Task_1/labels"
labels = [h for h in os.listdir(label_folder) if h.endswith(".tiff") or h.endswith(".tif")]

def min_max_finder(labels):
	#Searching for min and max in scaled NDVI values
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
	#Normalization for NDVI values of labels img pixels
	NDVI_Values = []
	min_val, max_val = min_max_finder(labels)
	
	for label_img in labels:
		img = tifffile.imread(f"{label_folder}/{label_img}").astype(float)
			#Ensuring NDVI value [-1,1]
		normalized_NDVI = ((img - min_val)/(max_val - min_val)) * 2-1
		normalized_NDVI = np.clip(normalized_NDVI, -1,1) 
		
		NDVI_Values.append(normalized_NDVI)
	return NDVI_Values

NDVI_list = ndvi_normalization(labels)

def feature_selection(NDVI_list):
	X = []
	y = []
	for ndvi in NDVI_list:
		mean = np.mean(ndvi)
		std = np.std(ndvi)
		X.append([mean, std])
		if mean > 0.5 and std < mean:
			y.append(1)
		else:
			y.append(0)
	X = np.array(X)
	y = np.array(y)
	
	return X,y

X,y = feature_selection(NDVI_list)
	#For analysis of data/Analyzing possibility of overfitting
#print(f"Min of Y: {np.min(y)}")
#print(f"Max of Y: {np.max(y)}")
#print(f"Min of X: {np.min(X)}")
#print(f"Max of X: {np.max(X)}")
#print(f"Counter: {Counter(y)}")
