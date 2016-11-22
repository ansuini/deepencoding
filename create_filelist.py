'''
Create list of files following folders order
'''

import pickle
import os
import numpy as np


datapath = os.path.join('/home','ans','repositories',
                   'repos_thesis_HPC','deepencoding','data','Renders_bw')

filelist = []
for root, dirs, files in os.walk(datapath):
    for file in files:
        if file.endswith(".png") and not file.startswith("."):
             filelist.append(os.path.join(root, file))

f = open( "filelist_bycat.p", "wb" )
pickle.dump(filelist, f)
f.close()

print("List created.")

