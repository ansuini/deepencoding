'''
Parallel extraction of features with mpi4py from VGG16

In this implementation we use MPI in the simplest possible
way, allocating everything on every processor, and with 
the iteration over the images one by one
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

# define local features vector

maxfilters  = 3
mymodel     = models[comm.rank]
mylayer     = mymodel.layers[-1]
myshape     = mylayer.output_shape[1:]

newshape    = []
newshape.append( len(filelist) )
newshape.append( min(maxfilters, myshape[0])*productory(myshape[1:]) )
newshape    = tuple(newshape)
myfeatures  = np.empty((0, newshape[1]))

# loop over images

tin   = time()
count = 0
for file in filelist[0:3]:
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    tfeats = mymodel.predict(x)
    tfeats = tfeats[:,0:min(maxfilters, myshape[0] ),:]
    myfeatures  = np.reshape(tfeats, (1,-1)  )
    count += 1
    print('Processed %d / %d from processor %d' % (count, len(filelist), comm.rank  ) )
print('Preprocessing and features computation completed in %ds ' % (time() - tin ))

# save features

print myfeatures.shape

f = open("features_bycat" + str(comm.rank) + ".p", "wb" )
pickle.dump(myfeatures, f)
f.close()

# barrier

comm.Barrier()
