'''
Parallel extraction of features with mpi4py from VGG16

In this implementation we use MPI dividing the data
among processors and recollecting it at the end, for
each model. In this way we divide the work equally
among all the processors
'''

from mpi4py import MPI

# parallel section 

tic = MPI.Wtime()
comm = MPI.COMM_WORLD

if comm.rank == 0:
    print(" Running %d parallel MPI processes" % comm.size)

import pickle
import os
from time import time
import numpy as np
from utils import productory

from keras.models import Model
from keras.applications.vgg16 import VGG16

# modelsa

base_model = VGG16(weights='imagenet') 
models = []
for layer in base_model.layers:
    models.append( Model(input=base_model.input,
                         output=base_model.get_layer(layer.name).output) )

if comm.rank == 0:
    print("Processor %d loading model..." % comm.size)

# load preprocessed data

X = np.load('../aux_data/preprocessed.npy')
X = X[0:100,:]

# divide work

nprocs = comm.size
N = X.shape[0]
chunk = int(N/nprocs)
sizes = []
for i in range(nprocs):
    if i < N % nprocs:
        sizes.append(chunk + 1)
    else:
        sizes.append(chunk)

idx = np.hstack( ( 0, np.cumsum(sizes) ) )
myX = X[idx[comm.rank] : idx[comm.rank + 1], : ]
del X

# iterate over models

for model in models:
   
    # manage shape

    if comm.rank == 0:
        ti = time()

    layer = model.layers[-1]
    key = layer.name
    shape = layer.output_shape[1:]
    newshape =  []
    newshape.append( myX.shape[0] )
    if len(shape) == 3:
        newshape.append( shape[0]*productory(shape[1:]) )
    else:
        newshape.append( shape[0]*productory(shape) )
    newshape = tuple(newshape)

    # compute features

    features = model.predict(np.squeeze(myX))
    outfile = 'results_mpi/' + key + '_' + str(comm.rank)

    if len(shape) == 3:
        np.save(outfile, np.reshape(features, (features.shape[0],-1)  ) )
    else:
        np.save(outfile, features)

    del features

    if comm.rank == 0:
       print("Time for model : %g --- nprocs = %d" % ( time() - ti, comm.size )  )

        
# barrier

comm.Barrier()
print("Whole time --- %g --- nprocs = %d" % ( MPI.Wtime() - tic, comm.size )  )
