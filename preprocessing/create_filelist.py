'''
Create list of files following index order
'''

import pickle
import os
import numpy as np

root     = os.path.join(os.environ['WORKDIR'])
datapath = os.path.join(root,'deepencoding','data','Renders_bw')

print("Files in index order from directory : " + datapath) 

f = open(os.path.join(datapath, 'trials.pkl'), 'rb')
L = pickle.load(f)
f.close()

filelist = map(lambda l: l['stimPath'], L)
filelist = [ os.path.join(root, 'deepencoding', 'data', file[2:] ) for file in filelist ]


f = open( "filelist.p", "wb" )
pickle.dump(filelist, f)
f.close()

print("List created.")

