# Causal Machine Learning for Cost-Effective Allocation of Development Aid

Abstract: The Sustainable Development Goals (SDGs) of the United Nations provide a blueprint of a better future by "leaving no one behind", and, to achieve the SDGs by 2030, poor countries require immense volumes of development aid. In this paper, we develop a causal machine learning framework for predicting heterogeneous treatment effects of aid disbursements to inform effective aid allocation. Specifically, our framework comprises three components: (i) a balancing autoencoder that uses representation learning to embed high-dimensional country characteristics while addressing treatment selection bias; (ii) a counterfactual generator to compute counterfactual outcomes for varying aid volumes to address small sample-size settings; and (iii) an inference model that is used to predict heterogeneous treatment-response curves. We demonstrate the effectiveness of our framework using data with official development aid earmarked to end HIV/AIDS in 105 countries, amounting to more than USD 5.2 billion. For this, we first show that our framework successfully computes heterogeneous treatment-response curves using semi-synthetic data. Then, we demonstrate our framework using real-world HIV data. Our framework points to large opportunities for a more effective aid allocation, suggesting that the total number of new HIV infections could be reduced by up to 3.3% (~50,000 cases) compared to the current allocation practice.

Paper available at: 

# Requirements 

python 3.7

pytorch 1.7

# Data

The data was acquired from the following sources:

HIV infection rate from UN SDG indicators database (https://unstats.un.org/sdgs/dataportal/database).

Development aid volume from OECD CRS database (https://stats.oecd.org/Index.aspx?DataSetCode=crs1).

Country characteristics from the World Bank database (https://data.worldbank.org/).

Country border matrix from the Github repository (https://github.com/geodatasource/country-borders).

# Demo

The script demo.py contains the implementation of our method CG-CT on a dummy dataset. 

First, we generate a dataset of size n=100 consisting of outcome Y, treatment A, and p-dimensional covariates X (with p=10). Then, for given hyperparameters, we implement our method in three steps: (i) the balancing autoencoder is used to embed the covariates while addressing treatment selection bias; (ii) the counterfactual generator is used to compute counterfactual outcomes for varying aid volumes; and (iii) the inference model (i.e., generalized propensity score (GPS)) is estimated on the resulting data. 

The demonstration code returns the estimated coefficients of the GPS model for a given dummy dataset. The runtime is around 1-2 minutes a desktop PC with Intel i7 but no specialized GPU. The functions for implementing the method are imported from the script main.py. 

# Results

Python script main.py contains the functions needed for the implementation of our method and the baselines. Python scripts eval_cf_MISE.py and cv_hyper_search.py contain implementation functions for our method and the baselines that are called when reproducing the results.

Jupyter notebook plots.ipynb contains all of the data-related plots that are presented in the manuscript. 

Running Python scripts res_run_sim.py and res_run.py reproduces the results of experiments with semi-synthetic data and with real-world data, respectively (reported in Figure 3 in the manuscript).

Running Python script optim_allocation.py reproduces the results for the reduction in the expected number of new HIV cases when comparing our suggested aid allocation vs. current practice in allocating aid (reported in Figure 4 in the manuscript).

Running Python scripts ablation.py, ablation_bae.py, ablation_cfgen.py, and sensitivity.py, causal_sensitivity.py, and robustness.py reproduces the results reported for our sensitivity analyses in the supplements.pdf in this repository. 



