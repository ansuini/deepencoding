'''
Preprocess according to a list of files  
'''

import pickle
import os
from time import time
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

X = np.zeros((len(filelist), 1,3,224,224 ) )
count = 0
tin = time()
for file in filelist:
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X[count,:] = np.squeeze(x)
    count += 1

    print('Processed %d / %d' % (count, len(filelist)) )

f = open("preprocessed.p", "wb" )
pickle.dump(X, f)
f.close()
print('Preprocessing completed in %ds ' % (time() - tin ))
