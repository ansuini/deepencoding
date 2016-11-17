'''
Load pretrained VGG16 and check time required for loading,
preprocessing and computing features 
'''

# imports

import time
import timeit
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# global pars

SHOW_SUMMARY = False
SHOW_IMG = False


# model

model = VGG16(weights='imagenet', include_top=True)

if SHOW_SUMMARY:
    model.summary()

# import image

tin = time.time()
img_path = 'samples/bear.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print('Import completed in %ds ' % (time.time() - tin ))


if SHOW_IMG:
    plt.imshow(img)
    plt.show()

# preprocessing

tin = time.time()
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Preprocessing completed in %ds ' % (time.time() - tin ))

# compute features
      
tin = time.time()
features = model.predict(x)
print('Features extracted in %ds' % (time.time() - tin ))
