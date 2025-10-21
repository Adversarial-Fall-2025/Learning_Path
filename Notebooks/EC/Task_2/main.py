from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from math import floor
from collections import Counter 
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":
    sample_files = os.listdir("./samples/")
    label_files = os.listdir("./labels/")
    label_names = [label_files[i].split('.')[0] for i in range(len(label_files))]

    
    samples = np.ndarray(shape=(len(label_files), 256, 256, 3))
    labels  = np.ndarray(shape=(len(label_files), 256, 256))

    for k, f in enumerate(sample_files):
        sample = Image.open("./samples/" + f)
        label = Image.open("./labels/" + label_files[label_names.index(f.split('.')[0])])

        sample_arr = np.array(sample)
        label_arr = np.array(label)

        samples[k] = sample_arr
        labels[k] = label_arr


        # Turn data into average channel amounts.


    samples = samples.reshape(614, 256)
    print(samples.shape)

    #samples = samples.reshape(len(samples),, )
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    print(clf)


############# Functions ##############

def label_convert():
    label_files = os.listdir("./labels/")

    classes = {
        0: [],
        1: []
    }

    for f in label_files:
        label = Image.open("./labels/" + f)
        label = np.array(label)

        avg_color = np.mean(label)
        if avg_color >= 170:
            classes[0].append(f.split('.')[0])
        else:
            classes[1].append(f.split('.')[0])


    print(f"Class 0: {len(classes[0])}\nClass 1: {len(classes[1])}")

def resize():
    sample = Image.open("./samples/" + os.listdir("./samples/")[0])

    size = sample.size[0]
    sample = sample.resize((size*3, size*3))
    sample.show()

def average_sample_label():
    sample_files = os.listdir("./samples/")
    label_files = os.listdir("./labels/")
    label_names = [label_files[i].split('.')[0] for i in range(len(label_files))]

    avg_label = np.ndarray(shape=(256, 256, len(sample_files))) 
    avg_sample = np.ndarray(shape=(256, 256, len(sample_files), 3)) 

    for k, f in enumerate(sample_files):
        sample = Image.open("./samples/" + f)
        label = Image.open("./labels/" + label_files[label_names.index(f.split('.')[0])])

        sample_arr = np.array(sample)
        label_arr = np.array(label)

        for i in range(256):
            for j in range(256):
                avg_label[i][j][k] = label_arr[i][j]

                for q in range(3):
                    avg_sample[i][j][k][q] = sample_arr[i][j][q]

        #(np.append(avg_label[i][j], label_arr[i][j]) for i in range(256) for j in range(256))

    avg_label_final = np.ndarray(shape=(256, 256))
    avg_sample_final = np.ndarray(shape=(256, 256, 3))

    for i in range(256):
        for j in range(256):
            avg_label_final[i][j] = np.mean(avg_label[i][j])
            
            for q in range(3):
                color = []
                for k in range(len(sample_files)):

                    color.append(avg_sample[i][j][k][q])
                    #print(avg_sample[i][j][k][q])
                avg_sample_final[i][j][q] = floor(np.mean(color))
            

    image_data_uint8 = avg_sample_final.astype(np.uint8)
    print(avg_sample_final)


    plt.title("Average Label")
    plt.imshow(avg_label_final, cmap='Grays')
    plt.show()

    plt.title("Average Sample")
    plt.imshow(image_data_uint8)
    plt.show()

def print_metadata_statistics():
    sample_dir = "./samples/"
    label_dir = "./labels/"

    label_files = os.listdir(label_dir)
    label_files = [label_files[i].split('.')[0] for i in range(len(label_files))]

    data = {
        'sizes': [],
        'found': []
    }

    for file in os.listdir(sample_dir):
        im = Image.open(sample_dir + file)

        data['sizes'].append(im.size)
        data['found'].append(str(file.split('.')[0] in label_files))

    counts = {
        'sizes': Counter(data['sizes']),
        'found': Counter(data['found'])
    }
    __import__('pprint').pprint(counts)

def plot_dataset_channels():
    dir = "./samples/"

    red, green, blue = np.ndarray(shape=(255,)), np.ndarray(shape=(255,)), np.ndarray(shape=(255,))

    for file in os.listdir(dir):
        im = Image.open(dir + file)
        hist = im.histogram()

        r_lst, g_lst, b_lst = np.array(hist[0:255]), np.array(hist[255:510]), np.array(hist[510:765])

        red += r_lst
        green += g_lst
        blue += b_lst

    _, axes = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.8)

    for i in range(3): 
        axes[i].set_xlabel("RGB Value")
        axes[i].set_ylabel("Frequency")
        axes[i].set_xticks(np.arange(0, 255+51, 51))

    axes[0].set_title("Red, Green, and Blue Channels Histogram for Entire Dataset")

    bins = [i*85 for i in range(4)]

    axes[0].hist(red, color='red', bins=bins, label="Red Channel")
    axes[0].legend()

    axes[1].hist(green, color='green', bins=bins, label="Green Channel")
    axes[1].legend()

    axes[2].hist(blue, color='blue', bins=bins, label="Blue Channel")
    axes[2].legend()

    plt.show()

def plot_first_sample_channels():
    dir = "./samples/"
    files = os.listdir(dir)

    im = Image.open(dir + files[0])
    #im.show()
    print(im.size)

    hist = im.histogram()

    _, axes = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.8)

    red, green, blue = hist[0:255], hist[255:510], hist[510:765]

    for i in range(3): axes[i].set_xlabel("Value"); axes[i].set_ylabel("Frequency")

    axes[0].hist(red, color='red')
    axes[0].set_title("Red, Green, and Blue Channels Histogram for Sample Image #1")
    axes[1].hist(green, color='green')
    axes[2].hist(blue, color='blue')

    #plt.title()
    plt.show()
