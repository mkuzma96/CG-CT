

#%% Packages

import numpy as np
import torch
from torch.utils.data import DataLoader
from main import *

#%% Hyperparameter search

def eval_MISE(data_train, data_val_x, data_val_a, data_val_y, model, hyperpar_list):
    
    device = torch.device('cpu')
    cv_results = []
    def RMISE(Y_pred, Y_true):
        return np.sqrt(np.mean((Y_pred-Y_true)**2))
    
    # Linear model
    if model == 'lm':  
        for allm in hyperpar_list['alpha_lm']:
            for orlm in hyperpar_list['order_lm']:       
                        
                hyperpars = {
                    'alpha_lm': allm,
                    'order_lm': orlm                            
                    }
                
                model = LM(data_train, hyperpars)
                Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                for i in range(data_val_a.shape[1]):
                    Y_pred[:,i] = LM_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model)
                Y_true = data_val_y
                loss = RMISE(Y_pred, Y_true)
                
                # Copmute loss and add to CV data
                cv_results.append({
                    'loss': loss,
                    'alpha_lm': allm,
                    'order_lm': orlm 
                    })

    # Neural network
    if model == 'nn':
        for lsnn in hyperpar_list['layer_size_nn']:
            for lrnn in hyperpar_list['lr_nn']:
                for drnn in hyperpar_list['drop_nn']:
                    for nenn in hyperpar_list['n_epochs_nn']:
                        for bsnn in hyperpar_list['b_size_nn']:    
                            hyperpars = {
                                'layer_size_nn': lsnn,
                                'lr_nn': lrnn,
                                'drop_nn': drnn,
                                'n_epochs_nn': nenn,
                                'b_size_nn': bsnn
                            }
                            
                            model = NN(data_train, hyperpars)
                            Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                            for i in range(data_val_a.shape[1]):
                                Y_pred[:,i] = NN_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model)
                            Y_true = data_val_y
                            loss = RMISE(Y_pred, Y_true)
                            
                            # Copmute loss and add to CV data
                            cv_results.append({
                                'loss': loss,
                                'layer_size_nn': lsnn,
                                'lr_nn': lrnn,
                                'drop_nn': drnn,
                                'n_epochs_nn': nenn,
                                'b_size_nn': bsnn
                            })
         
    # Generalized propensity score
    if model == 'gps':
        
        b_coef, sig_hat, a_coef = GPS(data_train)
        Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
        for i in range(data_val_a.shape[1]):
            Y_pred[:,i] = GPS_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), b_coef, sig_hat, a_coef)
        Y_true = data_val_y
        loss = RMISE(Y_pred, Y_true)
        
        cv_results.append({
            'loss': loss
        })
    
    # Dose-response networks                               
    if model == 'dr':
        for lsdr in hyperpar_list['layer_size_dr']:
            for rsdr in hyperpar_list['rep_size_dr']:
                for lrdr in hyperpar_list['lr_dr']:
                    for drdr in hyperpar_list['drop_dr']:
                        for nedr in hyperpar_list['n_epochs_dr']:   
                            for bsdr in hyperpar_list['b_size_dr']:
                                
                                hyperpars = {
                                    'layer_size_dr': lsdr,
                                    'rep_size_dr': rsdr,
                                    'lr_dr': lrdr,
                                    'drop_dr': drdr,
                                    'n_epochs_dr': nedr,
                                    'b_size_dr': bsdr, 
                                    'E': 5
                                }
                                                                                                            
                                model, sep = DRNet(data_train, hyperpars)
                                Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                for i in range(data_val_a.shape[1]):
                                    Y_pred[:,i] = DRNet_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model, sep)
                                Y_true = data_val_y
                                loss = RMISE(Y_pred, Y_true)
                                
                                # Copmute loss and add to CV data
                                cv_results.append({
                                    'loss': loss,
                                    'layer_size_dr': lsdr,
                                    'rep_size_dr': rsdr,
                                    'lr_dr': lrdr,
                                    'drop_dr': drdr,
                                    'n_epochs_dr': nedr,
                                    'b_size_dr': bsdr, 
                                    'E': 5
                                })
                                
    # SCIGAN
    if model == 'sci':
        for alsci in hyperpar_list['alpha_sci']:
            for lssci in hyperpar_list['layer_size_sci']:
                for lrsci in hyperpar_list['lr_sci']:
                    for nesci in hyperpar_list['n_epochs_sci']:
                        for bssci in hyperpar_list['b_size_sci']:  
                            for lsscinn in hyperpar_list['layer_size_scinn']:
                                for lrscinn in hyperpar_list['lr_scinn']:
                                    for nescinn in hyperpar_list['n_epochs_scinn']:
                                        for msci in hyperpar_list['m_sci']:
                                        
                                            aid_max = np.max(data_train[:,1])
                                            aid_min = np.min(data_train[:,1])
                                    
                                            hyperpars = {
                                                'layer_size_scinn': lsscinn,
                                                'lr_scinn': lrscinn,
                                                'n_epochs_scinn': nescinn,
                                                'alpha_sci': alsci,
                                                'dosage_samples_sci': msci,
                                                'noise_dim_sci': msci,
                                                'layer_size_sci': lssci,
                                                'lr_sci': lrsci,
                                                'n_epochs_sci': nesci,
                                                'b_size_sci': bssci,
                                                'aid_max': aid_max,
                                                'aid_min': aid_min
                                            }
                                            
                                            model = SCIGAN(data_train, hyperpars)
                                            Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                            for i in range(data_val_a.shape[1]):
                                                Y_pred[:,i] = SCIGAN_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model)
                                            Y_true = data_val_y
                                            loss = RMISE(Y_pred, Y_true)
                                            
                                            # Copmute loss and add to CV data
                                            cv_results.append({
                                                'loss': loss,
                                                'layer_size_scinn': lsscinn,
                                                'lr_scinn': lrscinn,
                                                'n_epochs_scinn': nescinn,
                                                'alpha_sci': alsci,
                                                'm_sci': msci,
                                                'layer_size_sci': lssci,
                                                'lr_sci': lrsci,
                                                'n_epochs_sci': nesci,
                                                'b_size_sci': bssci
                                            })

    # CGCT with LM inference
    if model == 'cgct_lm':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
                                    
                                    for alscw in hyperpar_list['alpha_scw']:
                                        for orscw in hyperpar_list['order_scw']:
                                            for mscw in hyperpar_list['m_scw']:
                                            
                                                aid_max = np.max(data_train[:,1])
                                                aid_min = np.min(data_train[:,1])
                                                aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], mscw))
                                                
                                                params_SCw = {
                                                    'alpha_scw': alscw,
                                                    'order_scw': orscw,
                                                    'm_scw': mscw
                                                }
                                                
                                                data_train_scw = SCw(data_train_bae, aid_random, params_SCw)
                                                for allm in hyperpar_list['alpha_lm']:
                                                    for orlm in hyperpar_list['order_lm']:       
                                                                
                                                        hyperpars = {
                                                            'alpha_lm': allm,
                                                            'order_lm': orlm                            
                                                            }
                                                        
                                                        model = LM(data_train_scw, hyperpars)  
                                                        Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                                        for i in range(data_val_a.shape[1]):
                                                            Y_pred[:,i] = LM_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), model)
                                                        Y_true = data_val_y
                                                        loss = RMISE(Y_pred, Y_true)
                                                        
                                                        # Copmute loss and add to CV data
                                                        cv_results.append({
                                                            'loss': loss,
                                                            'alpha_scw': alscw,
                                                            'order_scw': orscw,
                                                            'm_scw': mscw,
                                                            'layer_size_bae': lsbae,
                                                            'rep_size_bae': rsbae,
                                                            'drop_bae': drbae,
                                                            'lr_bae': lrbae,
                                                            'n_epochs_bae': nebae,
                                                            'b_size_bae': bsbae,
                                                            'alpha_bae': albae,
                                                            'alpha_lm': allm,
                                                            'order_lm': orlm  
                                                            })
                                                               
    # CGCT with NN inference
    if model == 'cgct_nn':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
                                    
                                    for alscw in hyperpar_list['alpha_scw']:
                                        for orscw in hyperpar_list['order_scw']:
                                            for mscw in hyperpar_list['m_scw']:
                                            
                                                aid_max = np.max(data_train[:,1])
                                                aid_min = np.min(data_train[:,1])
                                                aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], mscw))
                                                
                                                params_SCw = {
                                                    'alpha_scw': alscw,
                                                    'order_scw': orscw,
                                                    'm_scw': mscw
                                                }
                                                
                                                data_train_scw = SCw(data_train_bae, aid_random, params_SCw)
                                                for lsnn in hyperpar_list['layer_size_nn']:
                                                    for lrnn in hyperpar_list['lr_nn']:
                                                        for drnn in hyperpar_list['drop_nn']:
                                                            for nenn in hyperpar_list['n_epochs_nn']:
                                                                for bsnn in hyperpar_list['b_size_nn']:    
                                                                    hyperpars = {
                                                                        'layer_size_nn': lsnn,
                                                                        'lr_nn': lrnn,
                                                                        'drop_nn': drnn,
                                                                        'n_epochs_nn': nenn,
                                                                        'b_size_nn': bsnn
                                                                    }
                                                                    
                                                                    model = NN(data_train_scw, hyperpars)
                                                                    Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                                                    for i in range(data_val_a.shape[1]):
                                                                        Y_pred[:,i] = NN_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), model)
                                                                    Y_true = data_val_y
                                                                    loss = RMISE(Y_pred, Y_true)
                                                                    
                                                                    # Copmute loss and add to CV data
                                                                    cv_results.append({
                                                                        'loss': loss,
                                                                        'layer_size_nn': lsnn,
                                                                        'lr_nn': lrnn,
                                                                        'drop_nn': drnn,
                                                                        'n_epochs_nn': nenn,
                                                                        'b_size_nn': bsnn,
                                                                        'alpha_scw': alscw,
                                                                        'order_scw': orscw,
                                                                        'm_scw': mscw,
                                                                        'layer_size_bae': lsbae,
                                                                        'rep_size_bae': rsbae,
                                                                        'drop_bae': drbae,
                                                                        'lr_bae': lrbae,
                                                                        'n_epochs_bae': nebae,
                                                                        'b_size_bae': bsbae,
                                                                        'alpha_bae': albae
                                                                        })
                                                                    
    # CGCT with GPS inference
    if model == 'cgct_gps':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
                                    
                                    for alscw in hyperpar_list['alpha_scw']:
                                        for orscw in hyperpar_list['order_scw']:
                                            for mscw in hyperpar_list['m_scw']:
                                            
                                                aid_max = np.max(data_train[:,1])
                                                aid_min = np.min(data_train[:,1])
                                                aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], mscw))
                                                
                                                params_SCw = {
                                                    'alpha_scw': alscw,
                                                    'order_scw': orscw,
                                                    'm_scw': mscw
                                                }
                                                
                                                data_train_scw = SCw(data_train_bae, aid_random, params_SCw)
                                                b_coef, sig_hat, a_coef = GPS(data_train_scw)
                                                Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                                for i in range(data_val_a.shape[1]):
                                                    Y_pred[:,i] = GPS_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), b_coef, sig_hat, a_coef)
                                                Y_true = data_val_y
                                                loss = RMISE(Y_pred, Y_true)
                                                
                                                cv_results.append({
                                                    'loss': loss,
                                                    'alpha_scw': alscw,
                                                    'order_scw': orscw,
                                                    'm_scw': mscw,
                                                    'layer_size_bae': lsbae,
                                                    'rep_size_bae': rsbae,
                                                    'drop_bae': drbae,
                                                    'lr_bae': lrbae,
                                                    'n_epochs_bae': nebae,
                                                    'b_size_bae': bsbae,
                                                    'alpha_bae': albae
                                                    })
                                                                                                        
    # CGCT with DR inference
    if model == 'cgct_dr':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
                                    
                                    for alscw in hyperpar_list['alpha_scw']:
                                        for orscw in hyperpar_list['order_scw']:
                                            for mscw in hyperpar_list['m_scw']:
                                            
                                                aid_max = np.max(data_train[:,1])
                                                aid_min = np.min(data_train[:,1])
                                                aid_random = np.random.uniform(aid_min, aid_max, (data_train_bae.shape[0], mscw))
                                                
                                                params_SCw = {
                                                    'alpha_scw': alscw,
                                                    'order_scw': orscw,
                                                    'm_scw': mscw
                                                }
                                                
                                                data_train_scw = SCw(data_train_bae, aid_random, params_SCw)
                                                for lsdr in hyperpar_list['layer_size_dr']:
                                                    for rsdr in hyperpar_list['rep_size_dr']:
                                                        for lrdr in hyperpar_list['lr_dr']:
                                                            for drdr in hyperpar_list['drop_dr']:
                                                                for nedr in hyperpar_list['n_epochs_dr']:   
                                                                    for bsdr in hyperpar_list['b_size_dr']:
                                                                        
                                                                        hyperpars = {
                                                                            'layer_size_dr': lsdr,
                                                                            'rep_size_dr': rsdr,
                                                                            'lr_dr': lrdr,
                                                                            'drop_dr': drdr,
                                                                            'n_epochs_dr': nedr,
                                                                            'b_size_dr': bsdr, 
                                                                            'E': 5
                                                                        }
                                                                                                                                                    
                                                                        model, sep = DRNet(data_train_scw, hyperpars)
                                                                        Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                                                        for i in range(data_val_a.shape[1]):
                                                                            Y_pred[:,i] = DRNet_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), model, sep)
                                                                        Y_true = data_val_y
                                                                        loss = RMISE(Y_pred, Y_true)
                                                                        
                                                                        # Copmute loss and add to CV data
                                                                        cv_results.append({
                                                                            'loss': loss,
                                                                            'layer_size_dr': lsdr,
                                                                            'rep_size_dr': rsdr,
                                                                            'lr_dr': lrdr,
                                                                            'drop_dr': drdr,
                                                                            'n_epochs_dr': nedr,
                                                                            'b_size_dr': bsdr, 
                                                                            'E': 5,
                                                                            'alpha_scw': alscw,
                                                                            'order_scw': orscw,
                                                                            'm_scw': mscw,
                                                                            'layer_size_bae': lsbae,
                                                                            'rep_size_bae': rsbae,
                                                                            'drop_bae': drbae,
                                                                            'lr_bae': lrbae,
                                                                            'n_epochs_bae': nebae,
                                                                            'b_size_bae': bsbae,
                                                                            'alpha_bae': albae,
                                                                            })

    # CGCT with LM inference
    if model == 'cgct_lm_nocfgen':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
 
                                    for allm in hyperpar_list['alpha_lm']:
                                        for orlm in hyperpar_list['order_lm']:       
                                                    
                                            hyperpars = {
                                                'alpha_lm': allm,
                                                'order_lm': orlm                            
                                                }
                                            
                                            model = LM(data_train_bae, hyperpars)  
                                            Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                            for i in range(data_val_a.shape[1]):
                                                Y_pred[:,i] = LM_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), model)
                                            Y_true = data_val_y
                                            loss = RMISE(Y_pred, Y_true)
                                            
                                            # Copmute loss and add to CV data
                                            cv_results.append({
                                                'loss': loss,
                                                'layer_size_bae': lsbae,
                                                'rep_size_bae': rsbae,
                                                'drop_bae': drbae,
                                                'lr_bae': lrbae,
                                                'n_epochs_bae': nebae,
                                                'b_size_bae': bsbae,
                                                'alpha_bae': albae,
                                                'alpha_lm': allm,
                                                'order_lm': orlm  
                                                })
                                                               
    # CGCT with NN inference
    if model == 'cgct_nn_nocfgen':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced

                                    for lsnn in hyperpar_list['layer_size_nn']:
                                        for lrnn in hyperpar_list['lr_nn']:
                                            for drnn in hyperpar_list['drop_nn']:
                                                for nenn in hyperpar_list['n_epochs_nn']:
                                                    for bsnn in hyperpar_list['b_size_nn']:    
                                                        hyperpars = {
                                                            'layer_size_nn': lsnn,
                                                            'lr_nn': lrnn,
                                                            'drop_nn': drnn,
                                                            'n_epochs_nn': nenn,
                                                            'b_size_nn': bsnn
                                                        }
                                                        
                                                        model = NN(data_train_bae, hyperpars)
                                                        Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                                        for i in range(data_val_a.shape[1]):
                                                            Y_pred[:,i] = NN_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), model)
                                                        Y_true = data_val_y
                                                        loss = RMISE(Y_pred, Y_true)
                                                        
                                                        # Copmute loss and add to CV data
                                                        cv_results.append({
                                                            'loss': loss,
                                                            'layer_size_nn': lsnn,
                                                            'lr_nn': lrnn,
                                                            'drop_nn': drnn,
                                                            'n_epochs_nn': nenn,
                                                            'b_size_nn': bsnn,
                                                            'layer_size_bae': lsbae,
                                                            'rep_size_bae': rsbae,
                                                            'drop_bae': drbae,
                                                            'lr_bae': lrbae,
                                                            'n_epochs_bae': nebae,
                                                            'b_size_bae': bsbae,
                                                            'alpha_bae': albae
                                                            })
                                                        
    # CGCT with GPS inference
    if model == 'cgct_gps_nocfgen':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
                       
                                    b_coef, sig_hat, a_coef = GPS(data_train_bae)
                                    Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                    for i in range(data_val_a.shape[1]):
                                        Y_pred[:,i] = GPS_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), b_coef, sig_hat, a_coef)
                                    Y_true = data_val_y
                                    loss = RMISE(Y_pred, Y_true)
                                    
                                    cv_results.append({
                                        'loss': loss,
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                        })
                                                                                                        
    # CGCT with DR inference
    if model == 'cgct_dr_nocfgen':
        for lsbae in hyperpar_list['layer_size_bae']:
            for rsbae in hyperpar_list['rep_size_bae']:
                for drbae in hyperpar_list['drop_bae']:
                    for lrbae in hyperpar_list['lr_bae']:
                        for nebae in hyperpar_list['n_epochs_bae']:
                            for bsbae in hyperpar_list['b_size_bae']:
                                for albae in hyperpar_list['alpha_bae']:
                                    
                                    params_BAE = {
                                        'layer_size_bae': lsbae,
                                        'rep_size_bae': rsbae,
                                        'drop_bae': drbae,
                                        'lr_bae': lrbae,
                                        'n_epochs_bae': nebae,
                                        'b_size_bae': bsbae,
                                        'alpha_bae': albae
                                    }
                                    
                                    data_train_bae, mod_BAE = BAE(data_train, params_BAE)
                                    X = torch.from_numpy(data_val_x.astype(np.float32))
                                    X = X.to(device)
                                    X_reduced, _, _ = mod_BAE(X)
                                    X_reduced = X_reduced.cpu().detach().numpy()
                                    data_val_bae_x = X_reduced
                          
                                    for lsdr in hyperpar_list['layer_size_dr']:
                                        for rsdr in hyperpar_list['rep_size_dr']:
                                            for lrdr in hyperpar_list['lr_dr']:
                                                for drdr in hyperpar_list['drop_dr']:
                                                    for nedr in hyperpar_list['n_epochs_dr']:   
                                                        for bsdr in hyperpar_list['b_size_dr']:
                                                            
                                                            hyperpars = {
                                                                'layer_size_dr': lsdr,
                                                                'rep_size_dr': rsdr,
                                                                'lr_dr': lrdr,
                                                                'drop_dr': drdr,
                                                                'n_epochs_dr': nedr,
                                                                'b_size_dr': bsdr, 
                                                                'E': 5
                                                            }
                                                                                                                                        
                                                            model, sep = DRNet(data_train_bae, hyperpars)
                                                            Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                                            for i in range(data_val_a.shape[1]):
                                                                Y_pred[:,i] = DRNet_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_bae_x], axis=1), model, sep)
                                                            Y_true = data_val_y
                                                            loss = RMISE(Y_pred, Y_true)
                                                            
                                                            # Copmute loss and add to CV data
                                                            cv_results.append({
                                                                'loss': loss,
                                                                'layer_size_dr': lsdr,
                                                                'rep_size_dr': rsdr,
                                                                'lr_dr': lrdr,
                                                                'drop_dr': drdr,
                                                                'n_epochs_dr': nedr,
                                                                'b_size_dr': bsdr, 
                                                                'E': 5,
                                                                'layer_size_bae': lsbae,
                                                                'rep_size_bae': rsbae,
                                                                'drop_bae': drbae,
                                                                'lr_bae': lrbae,
                                                                'n_epochs_bae': nebae,
                                                                'b_size_bae': bsbae,
                                                                'alpha_bae': albae,
                                                                })

    # CGCT with LM inference
    if model == 'cgct_lm_nobae':
        for alscw in hyperpar_list['alpha_scw']:
            for orscw in hyperpar_list['order_scw']:
                for mscw in hyperpar_list['m_scw']:
                
                    aid_max = np.max(data_train[:,1])
                    aid_min = np.min(data_train[:,1])
                    aid_random = np.random.uniform(aid_min, aid_max, (data_train.shape[0], mscw))
                    
                    params_SCw = {
                        'alpha_scw': alscw,
                        'order_scw': orscw,
                        'm_scw': mscw
                    }
                    
                    data_train_scw = SCw(data_train, aid_random, params_SCw)
                    for allm in hyperpar_list['alpha_lm']:
                        for orlm in hyperpar_list['order_lm']:       
                                    
                            hyperpars = {
                                'alpha_lm': allm,
                                'order_lm': orlm                            
                                }
                            
                            model = LM(data_train_scw, hyperpars)  
                            Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                            for i in range(data_val_a.shape[1]):
                                Y_pred[:,i] = LM_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model)
                            Y_true = data_val_y
                            loss = RMISE(Y_pred, Y_true)
                            
                            # Copmute loss and add to CV data
                            cv_results.append({
                                'loss': loss,
                                'alpha_scw': alscw,
                                'order_scw': orscw,
                                'm_scw': mscw,
                                'alpha_lm': allm,
                                'order_lm': orlm  
                                })
                         
    # CGCT with NN inference
    if model == 'cgct_nn_nobae':
        for alscw in hyperpar_list['alpha_scw']:
            for orscw in hyperpar_list['order_scw']:
                for mscw in hyperpar_list['m_scw']:
                
                    aid_max = np.max(data_train[:,1])
                    aid_min = np.min(data_train[:,1])
                    aid_random = np.random.uniform(aid_min, aid_max, (data_train.shape[0], mscw))
                    
                    params_SCw = {
                        'alpha_scw': alscw,
                        'order_scw': orscw,
                        'm_scw': mscw
                    }
                    
                    data_train_scw = SCw(data_train, aid_random, params_SCw)
                    for lsnn in hyperpar_list['layer_size_nn']:
                        for lrnn in hyperpar_list['lr_nn']:
                            for drnn in hyperpar_list['drop_nn']:
                                for nenn in hyperpar_list['n_epochs_nn']:
                                    for bsnn in hyperpar_list['b_size_nn']:    
                                        hyperpars = {
                                            'layer_size_nn': lsnn,
                                            'lr_nn': lrnn,
                                            'drop_nn': drnn,
                                            'n_epochs_nn': nenn,
                                            'b_size_nn': bsnn
                                        }
                                        
                                        model = NN(data_train_scw, hyperpars)
                                        Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                        for i in range(data_val_a.shape[1]):
                                            Y_pred[:,i] = NN_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model)
                                        Y_true = data_val_y
                                        loss = RMISE(Y_pred, Y_true)
                                        
                                        # Copmute loss and add to CV data
                                        cv_results.append({
                                            'loss': loss,
                                            'layer_size_nn': lsnn,
                                            'lr_nn': lrnn,
                                            'drop_nn': drnn,
                                            'n_epochs_nn': nenn,
                                            'b_size_nn': bsnn,
                                            'alpha_scw': alscw,
                                            'order_scw': orscw,
                                            'm_scw': mscw
                                            })
                                        
    # CGCT with GPS inference
    if model == 'cgct_gps_nobae':
        for alscw in hyperpar_list['alpha_scw']:
            for orscw in hyperpar_list['order_scw']:
                for mscw in hyperpar_list['m_scw']:
                
                    aid_max = np.max(data_train[:,1])
                    aid_min = np.min(data_train[:,1])
                    aid_random = np.random.uniform(aid_min, aid_max, (data_train.shape[0], mscw))
                    
                    params_SCw = {
                        'alpha_scw': alscw,
                        'order_scw': orscw,
                        'm_scw': mscw
                    }
                    
                    data_train_scw = SCw(data_train, aid_random, params_SCw)
                    b_coef, sig_hat, a_coef = GPS(data_train_scw)
                    Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                    for i in range(data_val_a.shape[1]):
                        Y_pred[:,i] = GPS_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), b_coef, sig_hat, a_coef)
                    Y_true = data_val_y
                    loss = RMISE(Y_pred, Y_true)
                    
                    cv_results.append({
                        'loss': loss,
                        'alpha_scw': alscw,
                        'order_scw': orscw,
                        'm_scw': mscw
                        })
                                                                            
    # CGCT with DR inference
    if model == 'cgct_dr_nobae':
        for alscw in hyperpar_list['alpha_scw']:
            for orscw in hyperpar_list['order_scw']:
                for mscw in hyperpar_list['m_scw']:
                
                    aid_max = np.max(data_train[:,1])
                    aid_min = np.min(data_train[:,1])
                    aid_random = np.random.uniform(aid_min, aid_max, (data_train.shape[0], mscw))
                    
                    params_SCw = {
                        'alpha_scw': alscw,
                        'order_scw': orscw,
                        'm_scw': mscw
                    }
                    
                    data_train_scw = SCw(data_train, aid_random, params_SCw)
                    for lsdr in hyperpar_list['layer_size_dr']:
                        for rsdr in hyperpar_list['rep_size_dr']:
                            for lrdr in hyperpar_list['lr_dr']:
                                for drdr in hyperpar_list['drop_dr']:
                                    for nedr in hyperpar_list['n_epochs_dr']:   
                                        for bsdr in hyperpar_list['b_size_dr']:
                                            
                                            hyperpars = {
                                                'layer_size_dr': lsdr,
                                                'rep_size_dr': rsdr,
                                                'lr_dr': lrdr,
                                                'drop_dr': drdr,
                                                'n_epochs_dr': nedr,
                                                'b_size_dr': bsdr, 
                                                'E': 5
                                            }
                                                                                                                        
                                            model, sep = DRNet(data_train_scw, hyperpars)
                                            Y_pred = np.empty((data_val_a.shape[0], data_val_a.shape[1]))
                                            for i in range(data_val_a.shape[1]):
                                                Y_pred[:,i] = DRNet_pred(np.concatenate([data_val_a[:,i:(i+1)], data_val_x], axis=1), model, sep)
                                            Y_true = data_val_y
                                            loss = RMISE(Y_pred, Y_true)
                                            
                                            # Copmute loss and add to CV data
                                            cv_results.append({
                                                'loss': loss,
                                                'layer_size_dr': lsdr,
                                                'rep_size_dr': rsdr,
                                                'lr_dr': lrdr,
                                                'drop_dr': drdr,
                                                'n_epochs_dr': nedr,
                                                'b_size_dr': bsdr, 
                                                'E': 5,
                                                'alpha_scw': alscw,
                                                'order_scw': orscw,
                                                'm_scw': mscw
                                                })

                                                                         
    return cv_results

