import numpy as np
import tifffile
import os

sample_folder = "C:/Users/Carlos Escobar/Desktop/CLASSES/PandaHat Adversarial/Task_1/samples"
sample = [i for i in os.listdir(sample_folder) if i.endswith(".tiff") or i.endswith(".tif")]
label_folder = "C:/Users/Carlos Escobar/Desktop/CLASSES/PandaHat Adversarial/Task_1/labels"
labels = [h for h in os.listdir(label_folder) if h.endswith(".tiff") or h.endswith(".tif")]

def image_analysis(sample,labels):
        #Calculo de valores NDVI para todas las imagenes. 
        NDVI_Values = [] 
        if len(sample) == len(labels):
                for sample_file, label_file in zip(sample,labels): 
                        l = tifffile.imread(f"{label_folder}/{label_file}")
                        s = tifffile.imread(f"{sample_folder}/{sample_file}")
                        
                        NDVI = (l.astype(float) - s[:,:,0].astype(float))/ (l.astype(float) + s[:,:,0].astype(float) + 1e-6) 
                        NDVI_Values.append(NDVI)  
        else:
                print("Cannot complete analysis, Samples and Labels are NOT of equal length")
        return NDVI_Values
NDVI_list = image_analysis(sample,labels)
#print(NDVI_list[0].shape)
#print(NDVI_list[0][:5,:5])
#print(f"Lenght of NDVI Values list: {len(NDVI_list)}")
#print(f"Lenght of NDVI Values list[0]: {len(NDVI_list[0])}")

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
	
	return X,y

X,y = feature_selection(NDVI_list)
#print(f"Length of labels (y list): {len(y)}")
#print(f"First element in feature list: {X[:4]}")
#print(f"Feature List length: {len(X)}")

