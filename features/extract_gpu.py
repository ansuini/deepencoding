'''
Parallel extraction of features with mpi4py from VGG16

In this implementation we use gpu
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


# preprocessing and features computation

f = open( "../aux_data/filelist_bycat.p", "rb" )
filelist = pickle.load(f)
f.close()

# define models for whom to compute features

keys    = [model.layers[-1].name             for model in  models]
shapes  = [model.layers[-1].output_shape[1:] for model in  models]
#features = dict.fromkeys(keys)

maxfilters = 1e6

ti = time()

for file in filelist:
   
    

    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    for model in models:
    
        #ti = time()
        

        layer       = model.layers[-1]
        key         = layer.name
        shape       = layer.output_shape[1:]
    
        newshape    =  []
        newshape.append( len(filelist) )
       
        if len(shape) == 3:
            newshape.append( min(maxfilters, shape[0])*productory(shape[1:]) )
        else:
            newshape.append( min(maxfilters, shape[0])*productory(shape) )
        
        newshape = tuple(newshape)
        tfeats = model.predict(x)

        if len(shape) == 3:
            tfeats = tfeats[:,0:min(maxfilters, shape[0] ),:]
            tfeats  = np.reshape(tfeats, (1,-1)  )
        

        #print("Elapsed time --- %g " % ( time() - ti )  ) 
        #if features[key] == None:
        #    features[key] = tfeats
        #else:
        #    features[key] = np.vstack((features[key], tfeats ))
        
print("Elapsed time --- %g --- with gpu" % ( time() - ti )  ) 
