
# Load packages

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

#%% HIV data

# Load raw data:
df = pd.read_csv('raw/zHIV_data.csv') 

# Data imputation: 
imputer = KNNImputer(n_neighbors=5, copy=False)

for i in np.unique(df['year']):
    X = df.loc[df['year'] == i, ['hiv_rate', 'hiv_aid','gdp_per_cap_PPP','gdp_growth','inflation','unemployment',
                                 'population', 'fertility','maternal_mort','infant_mort', 'fdi', 'electricity',
                                 'life_exp','school_enr', 'tuberculosis', 'undernourishment']]
    X = imputer.fit_transform(X)
    df.loc[df['year'] == i, ['hiv_rate', 'hiv_aid','gdp_per_cap_PPP','gdp_growth','inflation','unemployment',
                             'population', 'fertility','maternal_mort','infant_mort', 'fdi', 'electricity',
                             'life_exp','school_enr', 'tuberculosis', 'undernourishment']] = X

# # Add lag for HIV
df['hiv_rate_lag'] = np.zeros(shape=df.shape[0])
for i in range(len(np.unique(df['country']))):
    ctry = np.unique(df['country'])[i]
    l = df[df['country'] == ctry].shape[0]
    for j in range(l):
        df.loc[df['country'] == ctry, 'hiv_rate_lag'] = df.loc[df['country'] == ctry, 'hiv_rate'].shift(1)
df['hiv_reduction'] = (df['hiv_rate_lag'] - df['hiv_rate'])/df['hiv_rate_lag']
df = df.dropna(axis=0)

# Data description:
    
# 1) Outcome:
# Relative reduction in HIV incidence rate - number of cases per 1000 uninfected cases

# 2) Treatment:
# Development aid in millions USD

# 3) Features description:
# a) GDP per capita PPP in thousands USD
# b) GDP growth in %
# c) Inflation in %
# d) Unemployment in %
# e) Population in millions
# f) Fertility rate - number of births per woman
# g) Maternal mortality rate - number of deaths per 100000 live births
# h) Infant mortality rate - number of deaths per 1000 live births
# i) Life expectancy in years
# j) School enrolment rate, primary - ratio of individuals receiving primary education
# k) Prevalence of undernourishment in % of population
# l) Access to electricity in % of population
# m) Foreign direct investment in millions USD
# n) Incidence of tuberculosis per 100000 people

# Store final dataset
df.to_csv('HIV_data.csv', index=False)




