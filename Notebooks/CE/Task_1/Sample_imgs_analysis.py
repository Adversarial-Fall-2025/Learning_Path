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
def image_analysis(sample,labels):
	#Analisis para los primeros 10 samples y labels
	for t in range(10):
		l = tifffile.imread(f"labels/{labels[t]}") 	
		s = tifffile.imread(f"samples/{sample[t]}") 
		print("Pixel analysis of SAMPLE")
		print("        R     G     B")

		for i in range(10): #Analisis de los primeros 10 pixeles en imagenes de sample y labels
			pixel = s[1,i]
			print(f"Sample Pixel#{i+1} {pixel}")
			lpixel = l[1,i]
			NDVI_Value = ((int(lpixel)) - int(pixel[0]))/(int((lpixel)) + int(pixel[0])) #Calculo para valor NDVI de cada pixel analisado
			print(f"Label Pixel#{i+1}, NDVI Value: {np.round(NDVI_Value, 4)}")

image_analysis(sample, labels)
print("---------------EOF---------------")
