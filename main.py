
# Load packages

import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#%% Dimension reduction

def BAE(data, params):
    
    p = data.shape[1] - 2
    
    # Device configuration
    device = torch.device('cpu')
    
    # Hyperparameters
    layer_size = params['layer_size_bae']
    rep_size = params['rep_size_bae']
    drop = params['drop_bae']
    lr = params['lr_bae']
    n_epochs = params['n_epochs_bae']
    b_size = params['b_size_bae']
    alpha = params['alpha_bae']

    # Data pre-processing 
    train = torch.from_numpy(data.astype(np.float32))
    train_loader = DataLoader(dataset=train, batch_size=math.ceil(b_size), shuffle=True)

    # Model 
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()

    def grad_reverse(x):
        return GradReverse.apply(x)

    class Encoder(nn.Module):
        def __init__(self, x_size, layer_size, rep_size, drop):
            super(Encoder, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(x_size, layer_size),  
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(layer_size, rep_size)
                )
        def forward(self, x):
            rep = self.model(x)
            return rep  

    class Decoder(nn.Module):
        def __init__(self, x_size, layer_size, rep_size, drop):
            super(Decoder, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(rep_size, layer_size),  
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(layer_size, x_size)
                )
        def forward(self, rep):
            x_out = self.model(rep)
            return x_out  

    class T_pred(nn.Module):
        def __init__(self, t_size, layer_size, rep_size, drop):
            super(T_pred, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(rep_size, layer_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(layer_size, 1)
                )    
        def forward(self, rep):
            t_out = self.model(rep)
            return t_out

    class BalancingAE(nn.Module):
        def __init__(self, x_size, t_size, layer_size, rep_size, drop):
            super(BalancingAE, self).__init__()
            self.encoder = Encoder(x_size, layer_size, rep_size, drop)
            self.decoder = Decoder(x_size, layer_size, rep_size, drop)
            self.t_pred = T_pred(t_size, layer_size, rep_size, drop)

        def forward(self, x):
            rep = self.encoder(x)
            x_out = self.decoder(rep)
            rep2 = grad_reverse(rep)
            t_out = self.t_pred(rep2)
            return rep, x_out, t_out

    mod_BAE = BalancingAE(x_size=p, t_size=1, layer_size=layer_size, rep_size=rep_size, drop=drop).to(device) 
    optimizer = torch.optim.Adam(mod_BAE.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.97)  
    MSE = nn.MSELoss()

    # Train the model
    for epoch in range(n_epochs):
        for batch in train_loader:
            t_data = batch[:,1:2]
            x_data = batch[:,2:]
            t_data = t_data.to(device)
            x_data = x_data.to(device)

            # Forward pass
            rep, x_out, t_out = mod_BAE(x_data)

            # Losses
            lossT = MSE(t_out, t_data)
            lossX = MSE(x_out, x_data)
            loss = lossX + alpha*lossT

            # Backward and optimize 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1) % 10 == 0:
                alpha = alpha*1.03
        scheduler.step()
    
    mod_BAE.eval()
    X = torch.from_numpy(data[:,2:].astype(np.float32))
    X = X.to(device)
    X_reduced, _, _ = mod_BAE(X)
    X_reduced = X_reduced.cpu().detach().numpy()
    data_new = np.concatenate([data[:,0:2],X_reduced], axis=1)
    
    return data_new, mod_BAE

#%% Generating counterfactuals

def SCw(data, aid_random, params):
    
    n = data.shape[0]
    k = data.shape[1]
    aid_dim = aid_random.shape[1]
    
    # Hyperparameters
    order = params['order_scw']
    alpha = params['alpha_scw']

    # Optimization function and constraints
    def optim_fn(w, x_other, x_target, alpha, order):
        dist = np.sum(np.square(np.matmul(np.transpose(x_other), w) - x_target))
        penalty = alpha*np.linalg.norm(w, order)
        return dist + penalty
    def constraint(w, t_other, t_target):
        return np.matmul(t_other,w) - t_target
            
    w_store = np.empty((n,aid_dim,n))
    for i in range(n):
        for j in range(aid_dim):
            x_target = data[i:(i+1),2:].reshape(-1)
            x_other = data[:,2:]
            t_target = aid_random[i,j]
            t_other = data[:,1:2].reshape((1,n))
            cons = [{'type':'eq', 'fun': constraint, 'args': [t_other, t_target]}]
            w_opt = scipy.optimize.minimize(fun=optim_fn, x0=(1/n)*np.ones((n,1)).reshape((n,1)), 
                                            args=(x_other, x_target, alpha, order), constraints=cons).x
            w_store[i,j,:] = w_opt
        
    Y_cf = np.empty((n,aid_dim))
    for i in range(n):
        for j in range(aid_dim):
            y_other = data[:,0:1].reshape((1,n))
            Y_cf[i,j] = np.matmul(y_other, w_store[i,j,:].reshape((n,1)))

    data_cf = np.empty((n*aid_dim, k))
    for i in range(n):
        for j in range(aid_dim):
            data_cf[i+n*j, 0] = Y_cf[i,j]
            data_cf[i+n*j, 1] = aid_random[i,j]
            data_cf[i+n*j, 2:] = data[i,2:]
    
    data_new = np.concatenate([data,data_cf],axis=0)
    
    return data_new

#%% Inference models - standard machine learning

def LM(data, params):
    
    alpha = params['alpha_lm']
    order = params['order_lm']
    
    if order == 0:
        model = LinearRegression()
    elif order == 1:
        model = Lasso(alpha)
    elif order == 2:
        model = Ridge(alpha)
        
    model.fit(data[:,1:], data[:,0])
    
    return model

def NN(data, params):
    
    p = data.shape[1] - 2
    
    # Device configuration
    device = 'cpu'

    # Hyperparameters
    lr = params['lr_nn']
    h_dim = params['layer_size_nn']
    drop = params['drop_nn']
    num_epochs = params['n_epochs_nn']
    batch_size = params['b_size_nn']

    class NN(nn.Module):
        def __init__(self, x_size, h_size, drop):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(x_size+1, h_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h_size, 1)
            )

        def forward(self, x, t):
            c = torch.cat([x, t],1)
            out = self.model(c)
            return out
        
    train = torch.from_numpy(data.astype(np.float32))
    dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    model = NN(p, h_dim, drop).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    MSE = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            y_data = batch[:,0:1].to(device)
            t_data = batch[:,1:2].to(device)
            x_data = batch[:,2:].to(device)

            # Train discriminator
            y_pred = model(x_data, t_data)
            loss = MSE(y_pred, y_data)

            # Train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

def LM_pred(newdata, model):
    
    Y_pred = model.predict(newdata)
    
    return Y_pred

def NN_pred(newdata, model):
    
    model.eval()
    test = torch.from_numpy(newdata.astype(np.float32))
    device = 'cpu'
    x_data = test[:,1:].to(device)
    t_data = test[:,0:1].to(device)
    Y_pred = model(x_data, t_data).view(-1)
    Y_pred = Y_pred.cpu().detach().numpy()
    
    return Y_pred

#%% Inference models - causal machine learning

def GPS(data):
    
    n = data.shape[0]
    p = data[:,2:].shape[1]
    model = LinearRegression()
    t_model = model.fit(data[:,2:], data[:,1])
    b_coef = np.empty((p+1,1))
    b_coef[1:,0] = t_model.coef_
    b_coef[0,0] = t_model.intercept_
    T_pred = t_model.predict(data[:,2:])
    T = data[:,1]
    sig_hat = np.sqrt((1/(n-p-1))*np.sum(np.square(T-T_pred)))
    X = np.concatenate([np.ones((n,1)), data[:,2:]],axis=1)
    gps = (1/np.sqrt(2*np.pi*sig_hat**2))*np.exp(-(T - np.matmul(X, b_coef).reshape(-1))**2/(2*sig_hat**2))
    X_new = np.concatenate([T.reshape((n,1)), (T**2).reshape((n,1)), gps.reshape((n,1)), 
                            (gps**2).reshape((n,1)), (gps*T).reshape((n,1))], axis=1)
    y_model = model.fit(X_new, data[:,0])
    a_coef = np.empty((6,1))
    a_coef[1:,0] = y_model.coef_
    a_coef[0,0] = y_model.intercept_
    
    return b_coef, sig_hat, a_coef
    
def DRNet(data, params):
    
    n = data.shape[0]
    p = data.shape[1] - 2
    
    # Device configuration
    device = 'cpu'

    # Hyperparameters
    lr = params['lr_dr']
    h_dim = params['layer_size_dr']
    rep_dim = params['rep_size_dr']
    num_epochs = params['n_epochs_dr']
    batch_size = params['b_size_dr']
    drop = params['drop_dr']
    E = params['E']

    class Representation(nn.Module):
        def __init__(self, x_size, h_size, rep_size, drop):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h_size, rep_size)
            )

        def forward(self, x):
            rep = self.model(x)
            return rep

    class Head(nn.Module):
        def __init__(self, rep_size, h_size, drop):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(rep_size+1, h_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h_size, 1)
            )        

        def forward(self, x, t):
            c = torch.cat([x,t],1)
            out = self.model(c)
            return out

    class DRNet(nn.Module):
        def __init__(self, x_size, h_size1, rep_size, h_size2, E, drop):
            super().__init__()
            self.rep = Representation(x_size, h_size1, rep_size, drop)
            self.heads = nn.ModuleList([Head(rep_size, h_size2, drop) for i in range(E)])
            self.E = E

        def forward(self, x, t):
            rep = self.rep(x)
            outs = torch.empty((x.shape[0],self.E))
            for i, head in enumerate(self.heads):
                outs[:,i:(i+1)] = head(rep,t)
            return outs
    
    train = torch.from_numpy(data.astype(np.float32))
    x = train[:,2:]
    t = train[:,1:2]
    y = train[:,0:1]
    t_min = torch.min(t)
    t_max = torch.max(t)
    add = (t_max - t_min)/E
    sep = torch.empty(E+1)
    sep[0] = t_min
    sep[E] = t_max
    for i in range(E-1):
        sep[i+1] = t_min + (i+1)*add
    l_enc = torch.empty((n,E))
    for i in range(n):
        for j in range(E):
            if t[i] >= sep[j] and t[i] <= sep[j+1]:
                l_enc[i,j] = 1
            else:
                l_enc[i,j] = 0
    train = torch.cat((y,t,x,l_enc), 1)

    dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    model = DRNet(p, h_dim, rep_dim, h_dim, E, drop).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    MSE = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            y_data = batch[:,0:1].to(device)
            t_data = batch[:,1:2].to(device)
            x_data = batch[:,2:(p+2)].to(device)
            l_enc = batch[:,(p+2):].to(device)

            # Train discriminator
            y_pred = model(x_data, t_data)
            y_pred = y_pred.view(-1).to(device)
            y_pred2 = y_pred[torch.nonzero(l_enc.reshape(-1))].to(device)
            loss = MSE(y_pred2, y_data)

            # Train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model, sep

def SCIGAN(data, params):
    
    n = data.shape[0]
    p = data.shape[1] - 2
    
    # Device configuration
    device = 'cpu'

    # Hyperparameters CF generator
    lr = params['lr_sci']
    h_dim = params['layer_size_sci']
    z_dim = params['noise_dim_sci']
    y_dim = params['dosage_samples_sci']
    alpha = params['alpha_sci']
    num_epochs = params['n_epochs_sci']
    batch_size = params['b_size_sci']
    aid_max = params['aid_max']
    aid_min = params['aid_min']
    
    # Hyperparameters Inference net
    h_dim2 = params['layer_size_scinn']
    lr2 = params['lr_scinn']
    num_epochs2 = params['n_epochs_scinn']

    # Data pre-processing 
    train = torch.from_numpy(data.astype(np.float32))
    n = data.shape[0] 
    label = torch.randint(0, y_dim, (n,1)).long()
    enc = OneHotEncoder(handle_unknown='ignore')
    l_enc = enc.fit_transform(label)
    l_enc = l_enc.toarray()
    l_enc = torch.tensor(l_enc, dtype=torch.float)
    train_new = torch.cat([label,l_enc,train], 1)
    train_loader = DataLoader(dataset=train_new, batch_size=math.ceil(batch_size), shuffle=True)

    class Discriminator(nn.Module):
        def __init__(self, x_size):
            super().__init__()
            weights = torch.Tensor(1, x_size)
            self.w = nn.Parameter(weights)
            self.relu = nn.ReLU()

        def forward(self, x, y):
            I = torch.eye(y.shape[1]).to(device)
            m = torch.ones(size=(y.shape[1],1)).to(device)
            equi1 = self.relu(torch.mm(I,torch.transpose(y,0,1)) + 
                              torch.mm(torch.mm(m,torch.transpose(m,0,1)),torch.transpose(y,0,1)) + 
                              torch.mm(torch.mm(m,self.w),torch.transpose(x,0,1)))
            equi2 = self.relu(torch.mm(I,equi1) + 
                              torch.mm(torch.mm(m,torch.transpose(m,0,1)),equi1))
            return torch.transpose(equi2,0,1)

    class Generator(nn.Module):
        def __init__(self, x_size, z_size, h_size):
            super().__init__() 
            self.linear1 = nn.Linear(x_size+z_size+2, h_size)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(h_size+1, h_size)
            self.linear3 = nn.Linear(h_size, 1)

        def forward(self, x, t, y, z, d):
            c1 = torch.cat([x, t, y, z],1)
            out1 = self.relu(self.linear1(c1))
            c2 = torch.cat([out1, d],1)
            out2 = self.linear3(self.relu(self.linear2(c2)))
            return out2

    disc = Discriminator(p).to(device)
    gen = Generator(p, z_dim, h_dim).to(device)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            labels = batch[:,0].long().to(device)
            l_enc = batch[:,1:(y_dim+1)].to(device)
            y_data = batch[:,(y_dim+1):(y_dim+2)].to(device)
            t_data = batch[:,(y_dim+2):(y_dim+3)].to(device)
            x_data = batch[:,(y_dim+3):].to(device)
            b_size = batch.shape[0]

            # Train discriminator
            z_data = torch.rand((b_size, z_dim)).to(device)
            d_data = ((aid_min-aid_max)*torch.rand((b_size, y_dim))+aid_max).to(device)
            d_data = d_data.view(-1)
            d_data[torch.nonzero(l_enc.reshape(-1))] = t_data
            d_data = d_data.view(l_enc.shape) 
            y_vec_pred = torch.empty(size=d_data.shape).to(device)
            for i in range(y_dim):
                y_vec_pred[:,i] = gen(x_data, t_data, y_data, z_data, d_data[:,i:(i+1)]).reshape(-1)
            y_vec_pred = y_vec_pred.view(-1)
            yf_pred = y_vec_pred[torch.nonzero(l_enc.reshape(-1))]
            y_vec_pred[torch.nonzero(l_enc.reshape(-1))] = y_data
            y_vec_pred = y_vec_pred.view(l_enc.shape) 
            probs = disc(x_data, y_vec_pred.detach())
            lossD = CE(probs, labels)

            opt_disc.zero_grad() 
            lossD.backward()
            opt_disc.step()

            # Train generator
            probs2 = disc(x_data, y_vec_pred)
            lossG1 = -CE(probs2, labels)
            lossG2 = MSE(yf_pred, y_data)
            lossG = lossG1 + alpha*lossG2

            opt_gen.zero_grad()
            lossG.backward()
            opt_gen.step()
    
    class InferenceNet(nn.Module):
        def __init__(self, x_size, h_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(x_size+1, h_size),
                nn.ReLU(),
                nn.Linear(h_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 1)
            )
    
        def forward(self, x, t):
            c = torch.cat([x, t],1)
            out = self.model(c)
            return out
    
    mod_InferenceNet = InferenceNet(p, h_dim2).to(device)
    optimizer = torch.optim.Adam(mod_InferenceNet.parameters(), lr=lr2)
    MSE = nn.MSELoss()
    
    for epoch in range(num_epochs2):
        for batch in train_loader:
            l_enc = batch[:,1:(y_dim+1)].to(device)
            y_data = batch[:,(y_dim+1):(y_dim+2)].to(device)
            t_data = batch[:,(y_dim+2):(y_dim+3)].to(device)
            x_data = batch[:,(y_dim+3):].to(device)
            b_size = batch.shape[0]
            
            # Train inference net
            z_data = torch.rand((b_size, z_dim)).to(device)
            d_data = ((aid_min-aid_max)*torch.rand((b_size, y_dim))+aid_max).to(device)
            d_data = d_data.view(-1)
            d_data[torch.nonzero(l_enc.reshape(-1))] = t_data
            d_data = d_data.view(l_enc.shape) 
            y_vec_pred = torch.empty(size=d_data.shape).to(device)
            for i in range(y_dim):
                y_vec_pred[:,i] = gen(x_data, t_data, y_data, z_data, d_data[:,i:(i+1)]).reshape(-1)
            y_vec_pred = y_vec_pred.view(-1)
            y_vec_pred[torch.nonzero(l_enc.reshape(-1))] = y_data
            y_vec_pred = y_vec_pred.view(l_enc.shape).detach() 
    
            y_vec_inf = torch.empty(size=d_data.shape).to(device)
            for i in range(y_dim):
                y_vec_inf[:,i] = mod_InferenceNet(x_data, d_data[:,i:(i+1)]).reshape(-1)
            loss = MSE(y_vec_inf, y_vec_pred)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return mod_InferenceNet

def GPS_pred(newdata, b_coef, sig_hat, a_coef):
    
    n = newdata.shape[0]
    T = newdata[:,0]
    X = np.concatenate([np.ones((n,1)), newdata[:,1:]],axis=1)
    gps = (1/np.sqrt(2*np.pi*sig_hat**2))*np.exp(-(T - np.matmul(X, b_coef).reshape(-1))**2/(2*sig_hat**2))
    X_out = np.concatenate([np.ones((n,1)), T.reshape((n,1)), (T**2).reshape((n,1)), gps.reshape((n,1)), 
                            (gps**2).reshape((n,1)), (gps*T).reshape((n,1))], axis=1)
    Y_pred = np.matmul(X_out, a_coef).reshape(-1)
    
    return Y_pred

def DRNet_pred(newdata, model, sep):
    
    model.eval()
    device = 'cpu'
    test = torch.from_numpy(newdata.astype(np.float32))
    n = test.shape[0]
    x_data = test[:,1:]
    t_data = test[:,0:1] 
    E = len(sep)-1
    t_min = torch.min(t_data)
    t_max = torch.max(t_data)
    sep[0] = t_min
    sep[-1] = t_max
    l_enc = torch.empty((n,E))
    for i in range(n):
        for j in range(E):
            if t_data[i] >= sep[j] and t_data[i] <= sep[j+1]:
                l_enc[i,j] = 1
            else:
                l_enc[i,j] = 0
    x_data = x_data.to(device)
    t_data = t_data.to(device)
    Y_pred = model(x_data, t_data)
    Y_pred = Y_pred.view(-1)
    Y_pred = Y_pred[torch.nonzero(l_enc.reshape(-1))].view(-1)
    Y_pred = Y_pred.cpu().detach().numpy()

    return Y_pred
       
def SCIGAN_pred(newdata, model):
    
    model.eval()
    test = torch.from_numpy(newdata.astype(np.float32))
    device = 'cpu'
    x_data = test[:,1:].to(device)
    t_data = test[:,0:1].to(device)
    Y_pred = model(x_data, t_data).view(-1)
    Y_pred = Y_pred.cpu().detach().numpy()
    
    return Y_pred
