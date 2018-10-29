import os 
import numpy as np 
import pandas as pd 
import pickle as pickle from natsort 
import natsorted 
import cv2 
from skimage import exposure 
from matplotlib import pyplot 
from skimage.io import imread 
from PIL import Image 
from skimage.io import imshow 
from skimage.filters import sobel 
from skimage import feature 
from sklearn.preprocessing 
import StandardScaler 
PATH = '/dev/Bmp/' 
LABELS = '/dev/Book1.csv' 
maxPixel = 64 imageSize = maxPixel * maxPixel 
num_features = imageSize def plot_sample(x):
    img = x.reshape(maxPixel, maxPixel)
    imshow(img)
    pyplot.show() def load_images(path):
    print ('reading file names ... ')
    #print (path)
    names=[]
    #names = [d for d in os.walk(path) if d.endswith('.png')]
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".png")]:
            names.append(os.path.join(dirpath, filename))
    #print names
    names = natsorted(names)
    num_rows = len(names)
    print (num_rows)
    print ('making dataset ... ')
    train_image = np.zeros((num_rows, num_features), dtype = float)
    levels = np.zeros((num_rows, 1), dtype = str)
    file_names = []
    i = 0
    for n in names:
        print ('testing')
        print (n.split('.')[0])
  
        data = np.array(np.array(cv2.imread(os.path.join(path, n))))
         
        #data = data.transpose(2,1,0).reshape(-1,data.shape[1]*data.shape[1]) print(data.shape)
        
        #print(data.shape[1])
        image = imread(os.path.join(path, n), as_grey = True)
        #print(image.shape)
        train_image[i, 0:num_features] = np.reshape(image, (1, num_features))
        #print (labels)
        if (labels.Path.any() == n.split('.')[0]):
            levels[i] = labels[Class].values
        
        i += 1
    return train_image, levels labels = pd.read_csv(LABELS, dtype = str) 
np.save('train_64.npy', np.hstack((train, levels))) 
