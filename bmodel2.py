
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler
import datetime
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import rankdata
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import logging
import datetime
from collections import Counter
import math

##########################################################################################
#Setup
##########################################################################################

arg1 = int(sys.argv[1])
prd1 = int(sys.argv[2])
note = datetime.datetime.now().strftime("%y%m%d%H%M")

# arg1 = '20230410'
# prd1 = 30
# note = 'test'

if(note=='test'):
  def printlog(x):
    print(datetime.datetime.now(), x)
else:
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s %(message)s', 
        filename=f'log/bmodel2_{arg1}_{note}.log',  
        filemode='a'  
    )
    printlog = logging.debug

if torch.cuda.is_available():
    printlog("GPU is available")
    device = torch.device("cuda")
else:
    printlog("GPU is not available, using CPU instead")
    device = torch.device("cpu")

printlog(f'load data: data/raw{arg1}.csv')

##########################################################################################
#Process data
##########################################################################################

raw = pd.read_csv(f'data/raw{arg1}.csv')
raw =  raw.drop_duplicates()
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values(['code','date'])
raw['did'] = raw['date'].rank(method='dense').astype(int) - 1

def fill_missing_values(row):
    return row.fillna(method='ffill')

raws = []
for i in ['open','close','vol','high','low']:
    rawi = pd.pivot_table(raw, values=i, index=['did'], columns=['code'])
    rawi = rawi.apply(fill_missing_values,axis=0)
    rawi = rawi[rawi.index>=np.min(raw['did'][raw['date']>'2021-01-01'])]
    raws.append(rawi.iloc()[-500:,:])

raws = dict(zip(['open','close','vol','high','low'],raws))

codesel = ~np.isnan(np.ravel(raws['close'].iloc()[0,:]))
codes = np.ravel(raws['close'].columns)[codesel]
closepvt = np.asarray(raws['close'])[:,codesel]
openpvt = np.asarray(raws['open'])[:,codesel]
volpvt = np.asarray(raws['vol'])[:,codesel]
lowpvt = np.asarray(raws['low'])[:,codesel]
highpvt = np.asarray(raws['high'])[:,codesel]

# Close Processing

Xclose = []
for i in range(closepvt.shape[0]-(prd1-1)):
    xi = closepvt[range(i, i + (prd1-1)), :] / closepvt[i + (prd1-1), None, :]
    xi = np.nan_to_num(xi,nan=-1)
    Xclose.append(np.ravel(xi.T))

Xclose = np.asarray(Xclose)

# High Processing

Xhigh = []
for i in range(highpvt.shape[0]-(prd1-1)):
    xi = highpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
    xi = np.nan_to_num(xi,nan=-1)
    xi = xi[-5:,:]
    Xhigh.append(np.ravel(xi.T))

Xhigh = np.asarray(Xhigh)

# Low Processing

Xlow = []
for i in range(lowpvt.shape[0]-(prd1-1)):
    xi = lowpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
    xi = np.nan_to_num(xi,nan=-1)
    xi = xi[-5:,:]
    Xlow.append(np.ravel(xi.T))

Xlow = np.asarray(Xlow)

# Vol Processing

Xvol = []
for i in range(volpvt.shape[0]-(prd1-1)):
    xi = volpvt[range(i, i + (prd1-1)), :] / volpvt[i + (prd1-1), None, :]
    xi = np.nan_to_num(xi,nan=-1)
    xi = xi[-10:,:]
    Xvol.append(np.ravel(xi.T))

Xvol = np.asarray(Xvol)

# X Processing

X = np.concatenate((Xclose,Xhigh,Xlow,Xvol),axis=1)

# Y Processing

Y1 = []
Y2 = []
Y3 = []
for i in range(closepvt.shape[0]-prd1-5):
    Y1.append(np.ravel((openpvt[range(i+prd1+1,i+prd1+6),:]/closepvt[i + (prd1-1), None, :]).T))
    Y2.append(np.ravel((closepvt[range(i+prd1+1,i+prd1+6),:]/closepvt[i + (prd1-1), None, :]).T))
    Y3.append(np.ravel((lowpvt[range(i+prd1+1,i+prd1+6),:]/closepvt[i + (prd1-1), None, :]).T))

Y1 = np.asarray(Y1)
Y2 = np.asarray(Y2)
Y3 = np.asarray(Y3)
Y = np.concatenate((Y1,Y2,Y3),axis=1)

# Y2 Processing

P = []
B = []
for i in range(closepvt.shape[0]-prd1-5):
    profi = openpvt[range(i+prd1+1,i+prd1+6),:].mean(axis=0)/openpvt[i+prd1,None,:]
    backi = lowpvt[range(i+prd1+1,i+prd1+6),:].min(axis=0)/openpvt[i+prd1,None,:]
    P.append(profi)
    B.append(backi)

P = np.concatenate(P,axis=0)
B = np.concatenate(B,axis=0)
Z = np.concatenate((P,B),axis=1)

Y, Z = Z, Y #Y for target, z for trend

# Scale

Xscaler = StandardScaler()
Xscaler.fit(X)
X = Xscaler.transform(X)

Yscaler = StandardScaler()
Yscaler.fit(Y)
Y = Yscaler.transform(Y)

Zscaler = StandardScaler()
Zscaler.fit(Z)
Z = Zscaler.transform(Z)

# Tensoring

X2 = X[Y.shape[0]:,:]
X = X[range(Y.shape[0]),:]

X = torch.tensor(X).float().to(device)
Y = torch.tensor(Y).float().to(device)
Z = torch.tensor(Z).float().to(device)
X2 = torch.tensor(X2).float().to(device)

##########################################################################################
#X -> (K->Y) -> X
##########################################################################################

class Autoencoder(nn.Module):
    def __init__(self, X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + Z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, Y_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, Z_dim)
        )
        self.regularization = nn.ModuleList()
        self.regularization.append(nn.Linear(hidden_dim, hidden_dim))
        self.regularization.append(nn.Linear(latent_dim, latent_dim))
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
    def forward(self, x):
        k = self.encoder(x)
        zhat = self.predictor(k)
        concat = torch.cat((zhat, k), dim=1)
        yhat = self.decoder(concat)
        reg_loss = 0.0
        for layer in self.regularization:
            reg_loss += torch.norm(layer.weight)
        reg_loss *= self.l2_reg
        return yhat, zhat, reg_loss

models = []
datasets = []
# np.random.seed(777)
# samples = np.random.permutation(np.ravel(range(X.shape[0])))
# samples = np.ravel((samples%5).tolist())

##########################################################################################
# Parametering
##########################################################################################

hidden_dim = 2048
latent_dim = 256
dropout_rates = [0.7,0.6,0.5,0.4]
l2_regs = [0.01,0.01,0.01,0.01]
num_epochses = [10000,10000,10000,10000]
lrs = [0.0001,0.0002,0.0003,0.0004]
early_tols = [1.1,1.1,1.1,1.05]
patiences = [10,10,10,10]
patience2s = [10,10,10,10]
w = [1,2,3,4,5]
w = w/np.sum(w)
num_robots = 100
num_votes = int(len(codes)*0.05)
printlog([hidden_dim,latent_dim,dropout_rates,l2_regs,num_epochses,lrs,early_tols,patiences,patience2s,w,num_robots,num_votes])

##########################################################################################
# Modeling
##########################################################################################

X_dim = X.shape[1]
Y_dim = Y.shape[1]
Z_dim = Z.shape[1]
models = []

for s in range(10):
    np.random.seed(s)
    samples = np.random.permutation(np.ravel(range(X.shape[0])))
    samples = np.ravel((samples%4).tolist())
    X_train,Y_train,Z_train,X_test,Y_test,Z_test=X[samples!=3,:],Y[samples!=3,:],Z[samples!=3,:],X[samples==3,:],Y[samples==3,:],Z[samples==3,:]
    # X_train,Y_train,Z_train,X_test,Y_test,Z_test=X[samples!=s,:],Y[samples!=s,:],Z[samples!=s,:],X[samples==s,:],Y[samples==s,:],Z[samples==s,:]
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    #Model 0
    m = 0
    dropout_rate = dropout_rates[m]
    l2_reg = l2_regs[m]
    num_epochs = num_epochses[m]
    lr = lrs[m]
    early_tol = early_tols[m]
    patience = patiences[m]
    patience2 = patience2s[m]
    counter2 = 0
    best_loss = np.inf
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            loss0 = criterion(ztr, zhat)
            loss1 = criterion(ytr, yhat)
            loss = .75*loss0 + .25*loss1 + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            yhate, zhate, l2_losse = model(X_test)
            vloss0 = criterion(Z_test, zhate)
            vloss1 = criterion(Y_test, yhate)
            vloss = .75*vloss0 + .25*vloss1 + l2_losse
        if epoch>1000:
            if vloss < best_loss*early_tol:
                if vloss < best_loss:
                    best_loss = vloss
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += 0.1
            else:
                counter += 1
            if counter >= patience:
                printlog(f'Model 0 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                model.load_state_dict(best_model_state_dict)
                counter2 += 1
            if counter2 > patience2:
                printlog(f"Model 0 @ {s} Early stopping at epoch {epoch+1}")
                break      
        if (epoch+1)%1000 == 0:
            printlog(f'Model 0 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
          
    coef = model.state_dict()
    #Model 1
    m = 1
    dropout_rate = dropout_rates[m]
    l2_reg = l2_regs[m]
    num_epochs = num_epochses[m]
    lr = lrs[m]
    early_tol = early_tols[m]
    patience = patiences[m]
    patience2 = patience2s[m]
    counter2 = 0
    best_loss = np.inf
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    model.load_state_dict(coef)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.7)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            loss0 = criterion(ztr, zhat)
            loss1 = criterion(ytr, yhat)
            loss = .65*loss0 + .35*loss1 + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            yhate, zhate, l2_losse = model(X_test)
            vloss0 = criterion(Z_test, zhate)
            vloss1 = criterion(Y_test, yhate)
            vloss = .65*vloss0 + .35*vloss1 + l2_losse
        if epoch>1000:
            if vloss < best_loss*early_tol:
                if vloss < best_loss:
                    best_loss = vloss
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += 0.1
            else:
                counter += 1
            if counter >= patience:
                printlog(f'Model 1 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                model.load_state_dict(best_model_state_dict)
                counter2 += 1
            if counter2 > patience2:
                printlog(f"Model 1 @ {s} Early stopping at epoch {epoch+1}")
                break      
        if (epoch+1)%1000 == 0:
            printlog(f'Model 1 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
    #Model 2
    m = 2
    dropout_rate = dropout_rates[m]
    l2_reg = l2_regs[m]
    num_epochs = num_epochses[m]
    lr = lrs[m]
    early_tol = early_tols[m]
    patience = patiences[m]
    patience2 = patience2s[m]
    counter2 = 0
    best_loss = np.inf
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    model.load_state_dict(coef)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            loss0 = criterion(ztr, zhat)
            loss1 = criterion(ytr, yhat)
            loss = .5*loss0 + .5*loss1 + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            yhate, zhate, l2_losse = model(X_test)
            vloss0 = criterion(Z_test, zhate)
            vloss1 = criterion(Y_test, yhate)
            vloss = .5*vloss0 + .5*vloss1 + l2_losse
        if epoch>1000:
            if vloss < best_loss*early_tol:
                if vloss < best_loss:
                    best_loss = vloss
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += 0.1
            else:
                counter += 1
            if counter >= patience:
                printlog(f'Model 2 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                model.load_state_dict(best_model_state_dict)
                counter2 += 1
            if counter2 > patience2:
                printlog(f"Model 2 @ {s} Early stopping at epoch {epoch+1}")
                break      
        if (epoch+1)%1000 == 0:
            printlog(f'Model 2 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
    printlog(f"{arg1}_{prd1}_{note}_{s} training finished")
    models.append(best_model_state_dict)
    torch.save(best_model_state_dict, f'model/model2_{arg1}_{prd1}_{note}_{s}.pt')

##########################################################################################
# Voting
##########################################################################################

profit = (closepvt/openpvt)[-5:,:]
back = (lowpvt/openpvt)[-5:,:]

#Model Merging

models = []
for s in range(10):
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    model.load_state_dict(torch.load(f'model/model2_{arg1}_{prd1}_{note}_{s}.pt'))
    models.append(model)

#Voting

scores = []
votes = []

for modeli in models:
    for s in range(num_robots):    
        torch.manual_seed(s)
        Y2, Z2, _ = modeli(X2)
        Y2 = -Yscaler.inverse_transform(Y2.cpu().detach().numpy())
        Y1 = Y2[:,range(len(codes))].argsort(axis=1)
        # Y2 = Y2[:,len(codes):].argsort(axis=1)
        profiti = []
        backi = []
        for i in range(5):
            profiti.append(profit[i,Y1[i,range(num_votes)]]*w[i])
            backi.append(back[i,Y1[i,range(num_votes)]]*w[i])
        scores.append([np.sum(np.ravel(profiti))/num_votes,np.sum(np.ravel(backi))/num_votes])
        votes.append(Y1[5,:])

scores = pd.DataFrame(np.asarray(scores))
scores.columns = ['profit','back']
scores = scores.sort_values('profit',ascending=False)
scores['rank'] = np.ravel(range(scores.shape[0]))
scores['idx'] = (scores['profit']-1)/(1-scores['back'])
scores = scores.sort_values('idx',ascending=False)
robotid = scores[(scores['idx']>np.quantile(scores['idx'],0.9))&(scores['profit']>np.quantile(scores['profit'],0.9))].index.tolist()

votes = np.asarray(votes)
rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes[robotid,:][:,range(num_votes)])), orient='index', columns=['count']).sort_values('count',ascending=False)
rlt['codes'] = codes[rlt.index]
rlt['idx'] = rlt['count']/(len(models)*num_robots)/(num_votes/len(codes))
rlt['date'] = arg1
rlt['method'] = 'model1'
rlt0 = rlt

rlt = rlt[rlt['idx']>=np.quantile(rlt['idx'],0.8)]
rlt[:, 'share'] = rlt['count'] / rlt['count'].sum()
printlog(rlt)

##########################################################################################
# Models Update
##########################################################################################

#New Training

models2 = []
s = -1
for model in models:
    s += 1
    m = 3
    dropout_rate = dropout_rates[m]
    l2_reg = l2_regs[m]
    num_epochs = num_epochses[m]
    lr = lrs[m]
    early_tol = early_tols[m]
    patience = patiences[m]
    patience2 = patience2s[m]
    counter2 = 0
    best_loss = np.inf
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.3)
    train_dataset = TensorDataset(X[-200:,:], Y[-200:,:], Z[-200:,:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            loss0 = criterion(ztr, zhat)
            loss1 = criterion(ytr, yhat)
            loss = .5*loss0 + .5*loss1 + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            yhate, zhate, l2_losse = model(X[-40:,:])
            vloss0 = criterion(Z[-40:,:], zhate)
            vloss1 = criterion(Y[-40:,:], yhate)
            vloss = .5*vloss0 + .5*vloss1 + l2_losse
        if epoch>200:
            if vloss < best_loss*early_tol:
                if vloss < best_loss:
                    best_loss = vloss
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += 1
            else:
                counter += 1
            if counter >= patience:
                printlog(f'Model 3 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                model.load_state_dict(best_model_state_dict)
                counter2 += 1
            if counter2 > patience2:
                printlog(f"Model 3 @ {s} Early stopping at epoch {epoch+1}")
                break      
        # if (epoch+1)%1000 == 0:
            # printlog(f'Model 3 @ {s}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{loss0:.4f}|{loss1:.4f}], Validate:[{vloss:.4f}|{vloss0:.4f}|{vloss1:.4f}]')
    printlog(f"{arg1}_{prd1}_{note}_{s} finetuned")
    models2.append(model)

models = models2

#Voting

scores = []
votes = []

for modeli in models:
    for s in range(num_robots):    
        torch.manual_seed(s)
        Y2, Z2, _ = modeli(X2)
        Y2 = -Yscaler.inverse_transform(Y2.cpu().detach().numpy())
        Y1 = Y2[:,range(len(codes))].argsort(axis=1)
        # Y2 = Y2[:,len(codes):].argsort(axis=1)
        profiti = []
        backi = []
        for i in range(5):
            profiti.append(profit[i,Y1[i,range(num_votes)]]*w[i])
            backi.append(back[i,Y1[i,range(num_votes)]]*w[i])
        scores.append([np.sum(np.ravel(profiti))/num_votes,np.sum(np.ravel(backi))/num_votes])
        votes.append(Y1[5,:])

scores = pd.DataFrame(np.asarray(scores))
scores.columns = ['profit','back']
scores = scores.sort_values('profit',ascending=False)
scores['rank'] = np.ravel(range(scores.shape[0]))
scores['idx'] = (scores['profit']-1)/(1-scores['back'])
scores = scores.sort_values('idx',ascending=False)
robotid = scores[(scores['idx']>np.quantile(scores['idx'],0.9))&(scores['profit']>np.quantile(scores['profit'],0.9))].index.tolist()

votes = np.asarray(votes)
rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes[robotid,:][:,range(num_votes)])), orient='index', columns=['count']).sort_values('count',ascending=False)
rlt['codes'] = codes[rlt.index]
rlt['idx'] = rlt['count']/(len(models)*num_robots)/(num_votes/len(codes))
rlt['date'] = arg1
rlt['method'] = 'model2'
rlt1 = rlt

rlt = rlt[rlt['idx']>=np.quantile(rlt['idx'],0.8)]
rlt[:, 'share'] = rlt['count'] / rlt['count'].sum()
printlog(rlt)

pd.concat([rlt0,rlt1],axis=0,ignore_index=True).to_csv(f'rlt/model2_{arg1}_{prd1}_{note}.csv')

