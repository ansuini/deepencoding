'''
Load pretrained VGG16 and extract from all layers
in parallel features of images specified in filelist

The images are in png format  
'''

import pickle
import os
from time import time
import timeit
import numpy as np
from matplotlib import pyplot as plt
from operator import mul

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# model

base_model = VGG16(weights='imagenet')

models = []
for i in range(5):
    models.append( Model(input=base_model.input,
                         output=base_model.get_layer('block' + str(i+1) + '_conv1' ).output) )
models.append(Model(input=base_model.input, output=base_model.get_layer('predictions' ).output))


def productory(list):
    return reduce(mul, list)

    
# preprocessing and features computation
img_path = '/home/ans/repositories/repos_thesis_HPC/deepencoding/samples/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


for model in models:

    tin = time()
    feats = model.predict(x)
    print('Elapsed : %g --- num pars : %d' % ( time() - tin, productory(list(feats.shape) ) ) )
    


f = open( "filelist_bycat.p", "rb" )
filelist = pickle.load(f)
f.close()
