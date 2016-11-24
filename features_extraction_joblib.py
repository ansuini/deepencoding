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
from joblib import Parallel, delayed

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# parallel features computation 

def compute_features(model, X, n_jobs):

    N = X.shape[0]
    chunk = int(N/n_jobs)

    parallelizer = Parallel(n_jobs=n_jobs, backend="threading")

    # prepare tasks

    X = np.squeeze(X)
    
    sizes = []
    for i in range(n_jobs):
        if i < N % n_jobs:
            sizes.append(chunk + 1)
        else:
            sizes.append(chunk)
    
    idx = np.hstack( ( 0, np.cumsum(sizes) ) )
    blocks = [  X[idx[i] : idx[i + 1], : ] for i in range(n_jobs)  ]

    tasks_iterator = ( delayed(model.predict, check_pickle=False)(block) for block in blocks )
  
    partial = parallelizer( tasks_iterator )
    result  = np.vstack(partial) 
   
    return result
     

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

    # construct models
    
    ti = time()
    base_model = VGG16(weights='imagenet')
    models = []
    for layer in base_model.layers:
        models.append( Model(input=base_model.input,
                             output=base_model.get_layer(layer.name).output) )
    
    print("Models construction elapsed time : %g" % ( time() - ti) )
  
    # load data

    f = open(os.path.join(os.environ['WORKDIR'], 'deepencoding', 'preprocessed.p'), 'rb')
    X = pickle.load(f)
    f.close()


 
    # compute features last layer
    ti = time()
    result = compute_features(models[-1], X, args.numprocs)
    print("N. procs : %d --- elapsed time : %g" % ( args.numprocs, time() - ti) )    

    # save

    f = open("features_parallel_" + str(args.numprocs) + ".p" , "wb")
    pickle.dump(result, f)
    f.close()    
