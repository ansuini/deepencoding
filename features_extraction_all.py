'''
Load pretrained VGG16 and extract from all layers
in parallel features of images specified in filelist

The images are in png format  
'''


import pickle
import os
from time import time
import numpy as np
from matplotlib import pyplot as plt
from utils import *


from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# construct models

base_model = VGG16(weights='imagenet')

models = []
for layer in base_model.layers:
    models.append( Model(input=base_model.input,
                         output=base_model.get_layer(layer.name).output) )

# load file list

f = open( "filelist_bycat.p", "rb" )
filelist = pickle.load(f)
f.close()
