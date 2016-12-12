from time import time
import numpy as np
import pickle

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn import model_selection


# load features and permute to indexes

layer = 'block4'
features = np.load('../binaries/' + layer + '_pool.npy')
f = open('../data/Renders_bw/perm_cat2ind.p', 'rb')
perm_cat2ind = pickle.load(f)
f.close()

features = features[perm_cat2ind,:]
features = np.reshape(features, (features.shape[0], -1) )

# load firing rates

M = np.load('../data/Renders_bw/M.npy')

# permutation 

seed = 1121
np.random.seed(seed)
idx = np.random.permutation(range(1440))

# define model and grid space

nprocs = 20
lasso = Lasso(random_state=0)
alphas = np.logspace(-2, 1, 50)
scores = list()
scores_std = list()
n_folds = 5

for neuron in range(17):      
    
    scores = []
    scores_std = []

    # log transform features

    X_train = features
    y_train = M[neuron,:]
    
    tin = time()
    for alpha in alphas:
        lasso.alpha = alpha
        this_scores = model_selection.cross_val_score(lasso, X_train, y_train, cv=n_folds, n_jobs=nprocs)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))
    print('Elapsed time : %g' % (time() - tin))
    np.save(layer + '_scores_' + str(neuron + 1),scores)
    np.save(layer + '_scores_std_' + str(neuron + 1),scores_std)
