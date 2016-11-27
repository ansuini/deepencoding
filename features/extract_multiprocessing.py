'''
Load pretrained VGG16 and extract from all layers
in parallel features of images specified in filelist

The images are in png format  
'''


import pickle
import os
import argparse
from time import time
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# worker

def worker(x, models, pid):
    print('I am %d inside worker' % (pid) )
    model = models[-1]
    pred = model.predict(x[0,:])  
    return pred


# main

if __name__ == '__main__':

    # Handle command line options
    parser = argparse.ArgumentParser(description='Compute features in parallel')
    parser.add_argument('--numprocs', required=True, type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use. ' + \
                        "Default for this machine is %d" % (multiprocessing.cpu_count(),) )
    
    args = parser.parse_args()

    if args.numprocs < 1:
        sys.exit('Number of processors to use must be greater than 0')

    # Start my pool
    pool = multiprocessing.Pool( args.numprocs )

    print("Using %d processors..." % ( args.numprocs ) )

    # construct models

    base_model = VGG16(weights='imagenet')

    models = []
    for layer in base_model.layers:
        models.append( Model(input=base_model.input,
                             output=base_model.get_layer(layer.name).output) )
    # load data

    f = open(os.path.join(os.environ['WORKDIR'], 'deepencoding', 'preprocessed_small.p'), 'rb')
    X = pickle.load(f)
    #X = np.squeeze(X)
    f.close()

    # Build task list

    N = X.shape[0]
    chunk = int(N/args.numprocs)
    
    tasks = []
    proc = 0
    while proc < args.numprocs:
        proc += 1
        if proc < ( N % args.numprocs ):
            tasks.append( ( X[proc*chunk : proc*chunk + chunk + 1,:], models, proc,) )
        else:
            tasks.append( ( X[proc*chunk : proc*chunk + chunk    ,:], models, proc,) )

    # send data and models to procs

    print("Sending data and processing ...")

    ti = time()
    results = [pool.apply_async( worker, t ) for t in tasks] 
    
    # process results

    for result in results:
        a = result.get()
        print("Result: x[0] : %g" % ( a[0][0]  ) )

    # close pool

    pool.close()
    pool.join()

    print("N. procs : %d --- elapsed time : %g" % ( args.numprocs, time() - ti) )    
