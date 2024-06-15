
#%% Load packages 

import numpy as np
import pandas as pd
import ast
import torch
from main import *
from eval_cf_MISE import *

#%% Data 

# Data train
df = pd.read_csv('data/Sim_data_train.csv')
data_train = df.to_numpy()

# Data test counterfactual
data_cf = np.load('data/data_cf.npz')
A_cf = data_cf['A_cf']
Y_cf = data_cf['Y_cf']
X_cf = data_cf['X_cf']

#%% Sensitivity alpha_bae: Point estimate and variance (averaged over 10 runs)

# Hyperpar list
hyper_opt_list = open("hyperpars/hyper_opt_list_HIV2017_sim.txt", "r")
hyper_opt_list = hyper_opt_list.read()
hyper_opt = ast.literal_eval(hyper_opt_list)

# Convert hyperpar_opt_list so that its values are iterable
for i in range(len(hyper_opt)):
    for key in hyper_opt[i].keys():
        hyper_opt[i][key] = [hyper_opt[i][key]]
            
# Select GPS hyperparameters
hyper_opt = hyper_opt[-1]

alpha_bae = [0.1, 1, 5]
model = 'cgct_gps'

# Set all seeds
np.random.seed(123)
torch.manual_seed(123)

# Get results
res_table = np.empty(shape=(3,10))
for l in range(10):
    test_loss = []
    for i, alp_bae in enumerate(alpha_bae):
        print('i=',i)   
        hyper_opt['alpha_bae'] = [alp_bae]
        cv_results = eval_MISE(data_train, X_cf, A_cf, Y_cf, model, hyper_opt)
        test_loss.append(cv_results[0]['loss'])
    res_table[:,l] = np.array(test_loss)

#%% Sensitivity m_scw: Point estimate and variance (averaged over 10 runs)

# Hyperpar list
hyper_opt_list = open("hyperpars/hyper_opt_list_HIV2017_sim.txt", "r")
hyper_opt_list = hyper_opt_list.read()
hyper_opt = ast.literal_eval(hyper_opt_list)

# Convert hyperpar_opt_list so that its values are iterable
for i in range(len(hyper_opt)):
    for key in hyper_opt[i].keys():
        hyper_opt[i][key] = [hyper_opt[i][key]]
            
# Select GPS hyperparameters
hyper_opt = hyper_opt[-1]

m_scw = [1, 5, 7]
model = 'cgct_gps'

# Set all seeds
np.random.seed(123)
torch.manual_seed(123)

# Get results
res_table = np.empty(shape=(3,10))
for l in range(10):
    test_loss = []
    for i, m in enumerate(m_scw):
        print('i=',i)   
        hyper_opt['m_scw'] = [m]
        cv_results = eval_MISE(data_train, X_cf, A_cf, Y_cf, model, hyper_opt)
        test_loss.append(cv_results[0]['loss'])
    res_table[:,l] = np.array(test_loss)
