'''
First encoding model with pretrained ResNet50
'''

# imports

import time 
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

# global pars

SHOW_SUMMARY = False
SHOW_IMG = False


# model

model = ResNet50(weights='imagenet', include_top=True)

if SHOW_SUMMARY:
    model.summary()

# process image

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

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



    
