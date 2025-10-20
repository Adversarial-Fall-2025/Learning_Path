from PIL import Image
import os

samples = [i for i in os.listdir("samples") if i.endswith(".tiff") or i.endswith(".tif")]

def image_resizer(samples): #Resize de imagen de 256x256 ~ 300x200
	newSize = (300, 200)
	for i in range(10):
		s = Image.open(f"samples/{samples[i]}")
		newS = s.resize(newSize)
		newS.save(f"Resized Images/ResizedSample{i+1}.tiff")
	print("Finished Resizing Images")
image_resizer(samples)
		
