
# Load packages
import numpy as np
from main import *

# Dummy data
n = 100
p = 10
X = np.random.normal(size=(n,p))
A = np.random.normal(size=(n,1))
Y = np.random.normal(size=(n,1))
data = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X],axis=1)

hyperpars = {'alpha_scw': 0.05,
             'order_scw': 1,
             'm_scw': 3,
             'layer_size_bae': 10,
             'rep_size_bae': 4,
             'drop_bae': 0,
             'lr_bae': 0.0005,
             'n_epochs_bae': 300,
             'b_size_bae': 22,
             'alpha_bae': 0.1}

#%% Run the code on dummy data

# Step 1: Balancing autoencoder
data_train_bae, mod_BAE = BAE(data, hyperpars)    

# Step 2: Counterfactual generator
a_max = np.max(data[:,1])
a_min = np.min(data[:,1])
a_random = np.random.uniform(a_min, a_max, (data_train_bae.shape[0], hyperpars['m_scw']))
data_train_scw = SCw(data_train_bae, a_random, hyperpars)

# Step 3: Inference model
b_coef, sig_hat, a_coef = GPS(data_train_scw)
