'''
Parallel extraction of features with mpi4py from VGG16

In this implementation we use MPI dividing the data
among processors and recollecting it at the end, for
each model. In this way we divide the work equally
among all the processors
'''

from mpi4py import MPI

# parallel section 

comm = MPI.COMM_WORLD

print("============================================================================")
print(" Running %d parallel MPI processes" % comm.size)

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

print("============================================================================")
print("Processor %d loading model..." % comm.size)

# preprocessing and features computation

f = open( "filelist_bycat.p", "rb" )
filelist = pickle.load(f)
f.close()

# local filelist

nprocs = comm.size
N = len(filelist)
chunk = int(N/nprocs)
sizes = []
for i in range(nprocs):
    if i < N % nprocs:
        sizes.append(chunk + 1)
    else:
        sizes.append(chunk)

# loop over images of the local chunk

idx = np.hstack( ( 0, np.cumsum(sizes) ) )
myfilelist = filelist[idx[comm.rank] : idx[comm.rank + 1] ]


for file in myfilelist:
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)


    for model in models:
        tfeats = model.predict(x)
        
# barrier

comm.Barrier()
