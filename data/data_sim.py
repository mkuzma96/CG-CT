
# Load packages

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

#%% Simulated data train - based on HIV 2016

# Set seed
np.random.seed(123)

# Real-world - train
df = pd.read_csv('HIV_data.csv') 
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

# Prepare semi-synthetic data
data_interaction = data_train[:,2:].copy()
for i in range(p):
    data_interaction[:,i] = A*data_interaction[:,i]
data_lm = np.concatenate([data_train, data_interaction], axis=1)
mod_LM = LinearRegression()
coefs = mod_LM.fit(data_lm[:,1:], data_lm[:,0]).coef_
intercept = mod_LM.fit(data_lm[:,1:], data_lm[:,0]).intercept_
means = intercept + np.matmul(data_lm[:,1:], coefs)
means = (means - np.min(means))/(np.max(means) - np.min(means))
means = np.sqrt(means)
Y_sim = np.random.normal(loc=means, scale=0.01)
data_sim_train = np.concatenate([Y_sim.reshape(n,1), data_train[:,1:]],axis=1)
data_sim_train = pd.DataFrame(data_sim_train)
data_sim_train.to_csv('Sim_data_train.csv', index=False)

#%% Simulated data test - counterfactual outcome based on HIV 2017

# Set seed
np.random.seed(123)

# Real world - test
df = pd.read_csv('HIV_data.csv')
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

# Prepare semi-synthetic data
A_min = np.min(data_test[:,1])
A_max = np.max(data_test[:,1])
interval = np.arange(A_min, A_max, (A_max-A_min)/64)
A_cf = np.empty((n,64))
Y_cf = np.empty((n,64))
for i in range(64):
    A_cf[:,i] = interval[i]*np.ones(n)
    data_interaction = data_test[:,2:].copy()
    for j in range(0,p):
        data_interaction[:,j] = A_cf[:,i]*data_interaction[:,j]
    means_cf = intercept + np.matmul(np.concatenate([A_cf[:,i].reshape((n,1)), data_test[:,2:], data_interaction], axis=1), coefs)
    means_cf = (means_cf - np.min(means_cf))/(np.max(means_cf) - np.min(means_cf))
    means_cf = np.sqrt(means_cf)
    Y_cf[:,i] = means_cf
np.savez('data_cf', A_cf=A_cf, Y_cf=Y_cf, X_cf=data_test[:,2:])
    

