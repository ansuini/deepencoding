'''
Load pretrained VGG16 and extract features of images
in directory 'datapath' and all its subdirectories

The images are in png format  
'''

# imports

import pickle
import os
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# global pars

COMPUTE_FEATS = True

datapath = os.path.join('/home','ans','repositories',
                   'repos_thesis_HPC','deepencoding','data','Renders_bw')

filelist = []
for root, dirs, files in os.walk(datapath):
    for file in files:
        if file.endswith(".png") and not file.startswith("."):
             filelist.append(os.path.join(root, file))

f = open( "filelist.p", "wb" )
pickle.dump(filelist, f)
f.close()

# model

model = VGG16(weights='imagenet', include_top=True)

# preprocessing and features computation (this can be done in parallel

if COMPUTE_FEATS == True:
    
    listfeatures = []
    features = np.zeros((len(filelist), 1000))
    count = 0
    tin = time.time()
    for file in filelist:
        img = image.load_img(file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        tfeats = model.predict(x)
        features[count,:] = tfeats
        count += 1
        print('Processed %d / %d' % (count, len(filelist)) )

    f = open("features.p", "wb" )
    pickle.dump(features, f)
    f.close()
    print('Preprocessing and features computation completed in %ds ' % (time.time() - tin ))

else:

    f = open("features.p", "rb" )
    features = pickle.load(f)
    f.close()
    print('Features loaded')

