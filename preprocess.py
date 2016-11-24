'''
Preprocess according to a list of files  
'''

import pickle
import os
from time import time
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# preprocessing and features computation

f = open( "filelist_bycat.p", "rb" )
filelist = pickle.load(f)
f.close()


limit = 20
X = np.zeros((limit, 1,3,224,224 ) )
count = 0

tin = time()
for file in filelist[0:limit]:
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X[count,:] = np.squeeze(x)
    count += 1

    print('Processed %d / %d' % (count, len(filelist)) )

np.save('preprocessed', X)

print('Preprocessing completed in %ds ' % (time() - tin ))
