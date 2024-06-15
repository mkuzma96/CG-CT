
#%% Load packages 

import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
import torch
from main import *
from cv_hyper_search import *

#%% Data 

# Real-world - train
df = pd.read_csv('data/HIV_data.csv') 
year = 2016
df = df[df['year'] == year]

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

# Convert hyperpar_opt_list so that its values are iterable
for i in range(len(hyper_opt)):
    for key in hyper_opt[i].keys():
        hyper_opt[i][key] = [hyper_opt[i][key]]
            
#%% Point estimate and variance (averaged over 10 runs)

models = ['lm', 'nn', 'gps', 'dr', 'sci', 'cgct_gps']

# Set all seeds
np.random.seed(123)
torch.manual_seed(123)

# Get results
res_table = np.empty(shape=(6,10))
for l in range(10):
    test_loss = []
    for i, model in enumerate(models):
        print('i=',i)   
        cv_results = CV_hyperpar_search(data_train, data_test, model, hyper_opt[i])
        test_loss.append(cv_results[0]['loss'])
    res_table[:,l] = np.array(test_loss)
