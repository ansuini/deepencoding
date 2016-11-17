'''
Load pretrained VGG16 and extract features of images
in directory 'datapath' and all its subdirectories

The images are in png format  
'''

# imports

import os
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# global pars

datapath = os.path.join('/home','ans','repositories',
                   'repos_thesis_HPC','deepencoding','data','Renders_bw')

filelist = []
for root, dirs, files in os.walk(datapath):
    for file in files:
        if file.endswith(".png") and not file.startswith("."):
             filelist.append(os.path.join(root, file))

print filelist
print("\nList length : %d\n" % (len(filelist) ) ) 

# model

#model = VGG16(weights='imagenet', include_top=True)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#features = model.predict(x)



