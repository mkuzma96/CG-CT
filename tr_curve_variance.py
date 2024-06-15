
# Load packages

import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
import torch
from main import *

#%% Estimated treatment-response curves 

# Real-world - train
df = pd.read_csv('data/HIV_data.csv') 
year = 2016
df = df[df['year'] == year]
hiv_rate_previous = df['hiv_rate'].values

# Prepare real world data
X = df.iloc[:, 5:-2].to_numpy()
A = df.iloc[:, 4].to_numpy()
Y = df.iloc[:, -1].to_numpy()

n = X.shape[0]
p = X.shape[1]

data = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X],axis=1)

# Data standardization: min-max scaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:,2:])
data_train = np.concatenate([data[:,0:2], data_scaled], axis=1)

# Real world - test
df = pd.read_csv('data/HIV_data.csv')
year = 2017
df = df[df['year'] == year]
population = df['population'].values

# Prepare real world data
X = df.iloc[:, 5:-2].to_numpy()
A = df.iloc[:, 4].to_numpy()
Y = df.iloc[:, -1].to_numpy()

n = X.shape[0]
p = X.shape[1]

data2 = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X],axis=1)

# Data standardization: min-max scaler
scaler = MinMaxScaler()
data_scaled2 = scaler.fit_transform(data2[:,2:])
data_test = np.concatenate([data2[:,0:2], data_scaled2], axis=1)

# Hyperpar list
hyper_opt_list = open("hyperpars/hyper_opt_list_HIV2017.txt", "r")
hyper_opt_list = hyper_opt_list.read()
hyper_opt = ast.literal_eval(hyper_opt_list)
opt_hyperpars = hyper_opt[-1]

# Set all seeds
np.random.seed(123)
torch.manual_seed(123)

runs = 10
b_coef = np.empty((5,runs))
sig_hat = np.empty((1,runs))
a_coef = np.empty((6,runs))
df_bae = np.empty((105,6,runs))

# Estimate model
for i in range(10):
    data_train_bae, mod_BAE = BAE(data_train, opt_hyperpars)    
    device = torch.device('cpu')  
    X = torch.from_numpy(data_test[:,2:].astype(np.float32))
    X = X.to(device)
    X_reduced, _, _ = mod_BAE(X)
    X_reduced = X_reduced.cpu().detach().numpy()
    data_test_bae = np.concatenate([data_test[:,0:2],X_reduced], axis=1)       
    df_bae[:,:,i] = data_test_bae         
    aid_max = np.max(data_train[:,1])
    aid_min = np.min(data_train[:,1])
    aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
    data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
    b, s, a = GPS(data_train_scw)
    b_coef[:,i:(i+1)], sig_hat[:,i:(i+1)], a_coef[:,i:(i+1)] = b, s, a
