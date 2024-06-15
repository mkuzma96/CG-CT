
# Load packages

import numpy as np
import pandas as pd
import scipy
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

# Estimate model
data_train_bae, mod_BAE = BAE(data_train, opt_hyperpars)    
device = torch.device('cpu')  
X = torch.from_numpy(data_test[:,2:].astype(np.float32))
X = X.to(device)
X_reduced, _, _ = mod_BAE(X)
X_reduced = X_reduced.cpu().detach().numpy()
data_test_bae = np.concatenate([data_test[:,0:2],X_reduced], axis=1)                                     
aid_max = np.max(data_train[:,1])
aid_min = np.min(data_train[:,1])
aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
b_coef, sig_hat, a_coef = GPS(data_train_scw)

# Optimal allocation

def optimal_allocation(data_test, b_coef, sig_hat, a_coef, hiv_rate_previous, population):
    
    n = data_test.shape[0]
    A_obs = data_test[:,1:2]
    budget = np.sum(A_obs)
    covariate_data = data_test[:,2:]
    Y_obs_pred = np.sum(hiv_rate_previous*(1-GPS_pred(data_test[:,1:], b_coef, sig_hat, a_coef))*population*1000)

    def optim_fn(A_prop, covariate_data, b_coef, sig_hat, a_coef, hiv_rate_previous, population):
        n = covariate_data.shape[0]
        Y_pred = np.sum(hiv_rate_previous*(1-GPS_pred(np.concatenate([A_prop.reshape((n,1)), covariate_data],axis=1), 
                                                      b_coef, sig_hat, a_coef))*population*1000)
        return Y_pred
    
    def constraint(A_prop, budget):
        return  budget - np.sum(A_prop) 
    
    cons = [{'type':'ineq', 'fun': constraint, 'args': [budget]}]
    
    A_std = np.std(A_obs)
    A_max = np.max(A_obs)
    bounds = scipy.optimize.Bounds(np.zeros(n), (A_max+A_std)*np.ones(n))
    
    total_cases = hiv_rate_previous*population*1000
    A_init = budget*(total_cases/np.sum(total_cases))
    
    A_opt = scipy.optimize.minimize(fun=optim_fn, x0=A_init, method='SLSQP',
                                    args=(covariate_data, b_coef, sig_hat, a_coef, hiv_rate_previous, population), 
                                    bounds=bounds, constraints=cons).x
    Y_opt_pred = np.sum(hiv_rate_previous*(1-GPS_pred(np.concatenate([A_opt.reshape((n,1)), covariate_data],axis=1), 
                                                      b_coef, sig_hat, a_coef))*population*1000)
    
    return A_opt, Y_opt_pred, Y_obs_pred

A_opt, Y_opt_pred, Y_obs_pred = optimal_allocation(data_test_bae, b_coef, sig_hat, a_coef, hiv_rate_previous, population)

#%% # Bootstrap CI

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

# Bootstrap intervals
boot_samples = 20
n = data_train.shape[0]
Y_obs_pred, Y_opt_pred = np.zeros(boot_samples), np.zeros(boot_samples)
for b in range(boot_samples):
    print(b)
    b_ind = np.random.choice(n, n, replace=True)
    data_train_bae, mod_BAE = BAE(data_train[b_ind,:], opt_hyperpars)    
    device = torch.device('cpu')  
    X = torch.from_numpy(data_test[:,2:].astype(np.float32))
    X = X.to(device)
    X_reduced, _, _ = mod_BAE(X)
    X_reduced = X_reduced.cpu().detach().numpy()
    data_test_bae = np.concatenate([data_test[:,0:2],X_reduced], axis=1)                                     
    aid_max = np.max(data_train[:,1])
    aid_min = np.min(data_train[:,1])
    aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
    data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
    b_coef, sig_hat, a_coef = GPS(data_train_scw)
    
    A_opt , Yopt, Yobs = optimal_allocation(data_test_bae, b_coef, sig_hat, a_coef, hiv_rate_previous, population)
    Y_opt_pred[b], Y_obs_pred[b] = Yopt, Yobs

