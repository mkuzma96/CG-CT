
# Load packages

import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
import torch
from main import *

#%% Sensitivity analysis w.r.t. past aid of neighbor countries

# Real-world - train
df_all = pd.read_csv('data/HIV_data.csv') 
year = 2016
df = df_all[df_all['year'] == year]

# Prepare real world data
X = df.iloc[:, 5:-2].to_numpy()
A = df.iloc[:, 4].to_numpy()
Y = df.iloc[:, -1].to_numpy()

n = X.shape[0]
p = X.shape[1] + 1

# Add covariate
borders = pd.read_csv('data/borders.csv')
borders = borders[borders['country_name'].isin(df['country'])]
borders = borders[borders['country_border_name'].isin(df['country'])]
past_aid_neigh = np.empty((n,1))
for i in range(n):
    ctry = df['country'].values[i]
    neighs = borders[borders['country_name'] == ctry]['country_border_name'].values
    if len(neighs) > 0:
        past_aid_neigh[i] = np.mean(df_all[(df_all['year'] == 2015) & (df_all['country'].isin(neighs))]['hiv_aid'])
    else:
        past_aid_neigh[i] = df_all[(df_all['year'] == 2015) & (df_all['country'] == ctry)]['hiv_aid']
    
data = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X, past_aid_neigh],axis=1)

# Data standardization: min-max scaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:,2:])
data_train = np.concatenate([data[:,0:2], data_scaled], axis=1)

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
aid_max = np.max(data_train[:,1])
aid_min = np.min(data_train[:,1])
aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
_, _, a1 = GPS(data_train_scw)

#%% Sensitivity analysis w.r.t. past aid 

# Real-world - train
df_all = pd.read_csv('data/HIV_data.csv') 
year = 2016
df = df_all[df_all['year'] == year]

# Prepare real world data
X = df.iloc[:, 5:-2].to_numpy()
A = df.iloc[:, 4].to_numpy()
Y = df.iloc[:, -1].to_numpy()

n = X.shape[0]
p = X.shape[1] + 1

# Add covariate
past_aid = df_all[df_all['year'] == 2015]['hiv_aid'].values.reshape(n,1)

data = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X, past_aid],axis=1)

# Data standardization: min-max scaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:,2:])
data_train = np.concatenate([data[:,0:2], data_scaled], axis=1)

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
aid_max = np.max(data_train[:,1])
aid_min = np.min(data_train[:,1])
aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
_, _, a2 = GPS(data_train_scw)

#%% Sensitivity analysis w.r.t. past HIV infection rate 

# Real-world - train
df_all = pd.read_csv('data/HIV_data.csv') 
year = 2016
df = df_all[df_all['year'] == year]

# Prepare real world data
X = df.iloc[:, 5:-2].to_numpy()
A = df.iloc[:, 4].to_numpy()
Y = df.iloc[:, -1].to_numpy()

n = X.shape[0]
p = X.shape[1] + 1

# Add covariate
past_hiv = df_all[df_all['year'] == 2015]['hiv_rate'].values.reshape(n,1)

data = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X, past_hiv],axis=1)

# Data standardization: min-max scaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:,2:])
data_train = np.concatenate([data[:,0:2], data_scaled], axis=1)

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
aid_max = np.max(data_train[:,1])
aid_min = np.min(data_train[:,1])
aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
_, _, a3 = GPS(data_train_scw)

#%% Sensitivity analysis w.r.t. past HIV rate of neighbor countries 

# Real-world - train
df_all = pd.read_csv('data/HIV_data.csv') 
year = 2016
df = df_all[df_all['year'] == year]

# Prepare real world data
X = df.iloc[:, 5:-2].to_numpy()
A = df.iloc[:, 4].to_numpy()
Y = df.iloc[:, -1].to_numpy()

n = X.shape[0]
p = X.shape[1] + 1

# Add covariate
borders = pd.read_csv('data/borders.csv')
borders = borders[borders['country_name'].isin(df['country'])]
borders = borders[borders['country_border_name'].isin(df['country'])]
past_hiv_neigh = np.empty((n,1))
for i in range(n):
    ctry = df['country'].values[i]
    neighs = borders[borders['country_name'] == ctry]['country_border_name'].values
    if len(neighs) > 0:
        past_hiv_neigh[i] = np.mean(df_all[(df_all['year'] == 2015) & (df_all['country'].isin(neighs))]['hiv_rate'])
    else:
        past_hiv_neigh[i] = df_all[(df_all['year'] == 2015) & (df_all['country'] == ctry)]['hiv_rate']
    
data = np.concatenate([Y.reshape(n,1), A.reshape(n,1), X, past_hiv_neigh],axis=1)

# Data standardization: min-max scaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:,2:])
data_train = np.concatenate([data[:,0:2], data_scaled], axis=1)

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
aid_max = np.max(data_train[:,1])
aid_min = np.min(data_train[:,1])
aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], opt_hyperpars['m_scw']))
data_train_scw = SCw(data_train_bae, aid_random, opt_hyperpars)
_, _, a4 = GPS(data_train_scw)
