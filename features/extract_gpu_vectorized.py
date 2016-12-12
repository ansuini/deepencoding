'''
Parallel extraction of features with mpi4py from VGG16

In this implementation we use 1 gpu and vectorized prediction
'''
import pickle
import os
from time import time
import numpy as np
from utils import productory

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# models

base_model = VGG16(weights='imagenet') 
models = []
for layer in base_model.layers:
    models.append( Model(input=base_model.input,

                         output=base_model.get_layer(layer.name).output) )

# load data

X = np.load('../aux_data/preprocessed.npy')
X = np.squeeze(X)

# define models for whom to compute features

for model in models:
    
    ti = time()            
    tfeats = model.predict(X)
    name = model.layers[-1].name
    print("Model : %s elapsed time --- %g --- with gpu" % ( model.name,  time() - ti )  )
    np.save(name, tfeats)
    del tfeats
