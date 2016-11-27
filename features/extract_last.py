'''
Load pretrained VGG16 and extract last layer features of images
specified in filelist

The images are in png format  
'''

import pickle
import os
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# model

model = VGG16(weights='imagenet', include_top=True)

# preprocessing and features computation

f = open( "filelist_bycat.p", "rb" )
filelist = pickle.load(f)
f.close()

    
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

f = open("features_bycat.p", "wb" )
pickle.dump(features, f)
f.close()
print('Preprocessing and features computation completed in %ds ' % (time.time() - tin ))
