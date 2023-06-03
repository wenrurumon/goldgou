
import torch
import pandas as pd
import numpy as np
import sys
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
import akshare as ak
from scipy.stats import binom_test

def printlog(x):
    print(datetime.datetime.now(), x)

def loaddata(date0,codes):
    codes2 = []
    for i in range(len(codes)):
        codes2.append(codes[i].replace('\n',''))
    codes = np.unique(codes2)
    raw = []
    for codei in codes:
        rawi = ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=int(date0)-20000, end_date=int(date0), adjust="qfq").iloc()[:,range(7)]
        rawi.columns = ['date','open','close','high','low','vol','val']
        rawi['code'] = codei
        raw.append(rawi)
    raw = pd.concat(raw,axis=0)
    raw = raw.drop('vol', axis=1)
    return(raw)

def fill_missing_values(row):
    return row.fillna(method='ffill')

def process(raw,prd1,seeds):
    prd2 = 5
    raw =  raw.drop_duplicates()
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values(['code','date'])
    raw['did'] = raw['date'].rank(method='dense').astype(int) - 1
    def fill_missing_values(row):
        return row.fillna(method='ffill')
    raws = []
    for i in ['open','close','val','high','low']:
        rawi = pd.pivot_table(raw, values=i, index=['did'], columns=['code'])
        rawi = rawi.apply(fill_missing_values,axis=0)
        raws.append(rawi)
    raws = dict(zip(['open','close','val','high','low'],raws))
    codesel = ~np.isnan(np.ravel(raws['close'].iloc()[0,:]))
    codes = np.ravel(raws['close'].columns)[codesel]
    closepvt = np.asarray(raws['close'])[:,codesel]
    openpvt = np.asarray(raws['open'])[:,codesel]
    valpvt = np.asarray(raws['val'])[:,codesel]
    lowpvt = np.asarray(raws['low'])[:,codesel]
    highpvt = np.asarray(raws['high'])[:,codesel]
    for i in ['open','close','val','high','low']:
        raws[i] = raws[i].iloc()[:,codesel]
    Xclose = []
    for i in range(closepvt.shape[0]-(prd1-1)):
        xi = closepvt[range(i, i + (prd1-1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xclose.append(np.ravel(xi.T))
    Xclose = np.asarray(Xclose)
    Xhigh = []
    for i in range(highpvt.shape[0]-(prd1-1)):
        xi = highpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        xi = xi[-5:,:]
        Xhigh.append(np.ravel(xi.T))
    Xhigh = np.asarray(Xhigh)
    Xlow = []
    for i in range(lowpvt.shape[0]-(prd1-1)):
        xi = lowpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        xi = xi[-5:,:]
        Xlow.append(np.ravel(xi.T))
    Xlow = np.asarray(Xlow)
    Xval = []
    Xval2 = []
    for i in range(valpvt.shape[0]-(prd1-1)):
        xi = valpvt[range(i, i + (prd1-1)), :] / valpvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xval2.append(np.ravel(xi.T))
        xi = xi[-10:,:]
        Xval.append(np.ravel(xi.T))
    Xval = np.asarray(Xval)
    Xval2 = np.asarray(Xval2)
    X = np.concatenate((Xclose,Xval2),axis=1)
    P = []
    B = []
    for i in range(closepvt.shape[0]-prd1-prd2):
        profi = openpvt[range(i+prd1+1,i+prd1+prd2+1),:].mean(axis=0)/openpvt[i+prd1,None,:]
        backi = lowpvt[range(i+prd1+1,i+prd1+prd2+1),:].min(axis=0)/openpvt[i+prd1,None,:]
        P.append(profi)
        B.append(backi)
    P = np.concatenate(P,axis=0)
    B = np.concatenate(B,axis=0)
    Z = P
    Xscaler = StandardScaler()
    Xscaler.fit(X)
    X = Xscaler.transform(X)
    Zscaler = StandardScaler()
    Zscaler.fit(Z)
    Z = Zscaler.transform(Z)
    Y = X[-Z.shape[0]:,range(Xclose.shape[1])]
    X2 = X[Z.shape[0]:,:]
    X = X[range(Z.shape[0]),:]
    X = torch.tensor(X).float().to(device)
    Y = torch.tensor(Y).float().to(device)
    Z = torch.tensor(Z).float().to(device)
    X2 = torch.tensor(X2).float().to(device)
    datasets = []
    for seed in seeds:
        np.random.seed(seed)
        samples = np.random.permutation(np.ravel(range(X.shape[0])))
        samples = np.ravel((samples%5).tolist())
        for s in range(5):
            X_train,Y_train,Z_train,X_test,Y_test,Z_test=X[samples!=s,:],Y[samples!=s,:],Z[samples!=s,:],X[samples==s,:],Y[samples==s,:],Z[samples==s,:]
            datasets.append([X_train,Y_train,Z_train,X_test,Y_test,Z_test])
    life = []
    for i in range(10, closepvt.shape[0]):
        data = closepvt[(i-9):(i+1), :]
        col_mean = np.mean(data, axis=0)
        life.append(col_mean)
    life = np.array(life)
    raws['life'] = pd.DataFrame(life)
    raws['life'].columns = raws['close'].columns
    return(datasets,X,Y,Z,X2,Zscaler,raws)

def process2(raw,prd1,seeds):
    prd2 = 5
    raw =  raw.drop_duplicates()
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values(['code','date'])
    raw['did'] = raw['date'].rank(method='dense').astype(int) - 1
    def fill_missing_values(row):
        return row.fillna(method='ffill')
    raws = []
    for i in ['open','close','val','high','low']:
        rawi = pd.pivot_table(raw, values=i, index=['did'], columns=['code'])
        rawi = rawi.apply(fill_missing_values,axis=0)
        raws.append(rawi)
    raws = dict(zip(['open','close','val','high','low'],raws))
    codesel = ~np.isnan(np.ravel(raws['close'].iloc()[0,:]))
    codes = np.ravel(raws['close'].columns)[codesel]
    closepvt = np.asarray(raws['close'])[:,codesel]
    openpvt = np.asarray(raws['open'])[:,codesel]
    valpvt = np.asarray(raws['val'])[:,codesel]
    lowpvt = np.asarray(raws['low'])[:,codesel]
    highpvt = np.asarray(raws['high'])[:,codesel]
    for i in ['open','close','val','high','low']:
        raws[i] = raws[i].iloc()[:,codesel]
    Xclose = []
    for i in range(closepvt.shape[0]-(prd1-1)):
        xi = closepvt[range(i, i + (prd1-1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xclose.append(np.ravel(xi.T))
    Xclose = np.asarray(Xclose)
    Xhigh = []
    for i in range(highpvt.shape[0]-(prd1-1)):
        xi = highpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xhigh.append(np.ravel(xi.T))
    Xhigh = np.asarray(Xhigh)
    Xlow = []
    for i in range(lowpvt.shape[0]-(prd1-1)):
        xi = lowpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xlow.append(np.ravel(xi.T))
    Xlow = np.asarray(Xlow)
    Xval = []
    for i in range(valpvt.shape[0]-(prd1-1)):
        xi = valpvt[range(i, i + (prd1-1)), :] / valpvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xval.append(np.ravel(xi.T))
    Xval = np.asarray(Xval)
    X = np.concatenate((Xclose,Xhigh,Xlow,Xval),axis=1)
    P = []
    B = []
    for i in range(closepvt.shape[0]-prd1-prd2):
        profi = openpvt[range(i+prd1+1,i+prd1+prd2+1),:].mean(axis=0)/openpvt[i+prd1,None,:]
        backi = lowpvt[range(i+prd1+1,i+prd1+prd2+1),:].min(axis=0)/openpvt[i+prd1,None,:]
        P.append(profi)
        B.append(backi)
    P = np.concatenate(P,axis=0)
    B = np.concatenate(B,axis=0)
    Z = P
    Xscaler = StandardScaler()
    Xscaler.fit(X)
    X = Xscaler.transform(X)
    Zscaler = StandardScaler()
    Zscaler.fit(Z)
    Z = Zscaler.transform(Z)
    Y = X[-Z.shape[0]:,range(Xclose.shape[1])]
    X2 = X[Z.shape[0]:,:]
    X = X[range(Z.shape[0]),:]
    X = torch.tensor(X).float().to(device)
    Y = torch.tensor(Y).float().to(device)
    Z = torch.tensor(Z).float().to(device)
    X2 = torch.tensor(X2).float().to(device)
    datasets = []
    for seed in seeds:
        np.random.seed(seed)
        samples = np.random.permutation(np.ravel(range(X.shape[0])))
        samples = np.ravel((samples%5).tolist())
        for s in range(5):
            X_train,Y_train,Z_train,X_test,Y_test,Z_test=X[samples!=s,:],Y[samples!=s,:],Z[samples!=s,:],X[samples==s,:],Y[samples==s,:],Z[samples==s,:]
            datasets.append([X_train,Y_train,Z_train,X_test,Y_test,Z_test])
    life = []
    for i in range(10, closepvt.shape[0]):
        data = closepvt[(i-9):(i+1), :]
        col_mean = np.mean(data, axis=0)
        life.append(col_mean)
    life = np.array(life)
    raws['life'] = pd.DataFrame(life)
    raws['life'].columns = raws['close'].columns
    return(datasets,X,Y,Z,X2,Zscaler,raws)

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

def train(modeli, hidden_dim, latent_dim, dropout_rate, l2_reg, lr, early_tol, patience, patience2):
    X_train,Y_train,Z_train,X_test,Y_test,Z_test = datasets[modeli]
    X_dim = X_train.shape[1]
    Y_dim = Y_train.shape[1]
    Z_dim = Z_train.shape[1]
    # Model 0
    m = 0
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    counter2 = 0
    best_loss = np.inf
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr*0.1)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            lossz = criterion(ztr, zhat)
            lossy = criterion(ytr, yhat)
            # wwz = epoch/1000
            wwz = 0.5
            wz = lossz / (wwz*lossz+lossy)
            wy = 1-wz
            loss = wy*lossy + wz*lossz + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            yhate, zhate, l2_losse = model(X_test)
            vlossz = criterion(Z_test, zhate)
            vlossy = criterion(Y_test, yhate)
            vloss = wy*vlossy + wz*vlossz + l2_losse
        if epoch==0:
            printlog(f'Model {modeli}.{m} training, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
        if epoch>0:
            if vloss < best_loss*early_tol:
                if vloss < best_loss:
                    best_loss = vloss
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += ((counter2+1)/10)
            else:
                counter += 1
            if counter >= patience:
                counter2 += 1
                # if counter2 == 1:
                    # printlog(f'Model {modeli}.{m} training, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                model.load_state_dict(best_model_state_dict)
            if counter2 > patience2:
                model.load_state_dict(best_model_state_dict)
                with torch.no_grad():
                    yhate, zhate, l2_losse = model(X_test)
                    vlossz = criterion(Z_test, zhate)
                    vlossy = criterion(Y_test, yhate)
                    vloss = wy*vlossy + wz*vlossz + l2_losse
                break    
    printlog(f"Model {modeli}.{m} stop, Epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
    return(model)

def roboting(num_robots,models):
    votes = []
    for modeli in models:
        votei = []
        for s in range(int(num_robots/len(models))):
            torch.manual_seed(s)
            Y2, Z2, _ = modeli(X2)
            Z2 = Zscaler.inverse_transform(Z2.cpu().detach().numpy())
            votei.append(((-Z2).argsort(axis=1)))
        votes.append(np.asarray(votei))
    return(np.asarray(votes))

def voting(votes,prop_votes,prop_robots):
    prd2 = 5
    rlts = []
    for i in range(votes.shape[0]):
        votesi = votes[i][:,:,range(int(prop_votes*raws['close'].shape[1]))]
        today = np.asarray(raws['close'].iloc()[-prd2:,:])/np.asarray(raws['open'].iloc()[-prd2:,:])
        tonite = (np.asarray(raws['open'].iloc()[1:,:])/np.asarray(raws['close'].iloc()[range(raws['close'].shape[0]-1),:]))[-(prd2-1):,:]
        tonite = np.concatenate((tonite,np.asarray([1]*tonite.shape[1]).reshape(1,tonite.shape[1])),axis=0)
        scoresi = []
        for i in range(votesi.shape[0]):
            votei = votesi[i]
            scorei = []
            for i in range(prd2):
                todayi = np.mean(today[i,votei[i]])
                tonitei = np.mean(tonite[i,votei[i]])
                profiti = todayi*tonitei
                scorei.append([todayi,tonitei,profiti])
            scorei = np.asarray(scorei)
            scoresi.append(np.concatenate([scorei.mean(axis=0),scorei.min(axis=0),scorei.prod(axis=0)],axis=0))
        scoresi = np.asarray(scoresi)
        robotsi = (-(scoresi[:,8])).argsort()
        robotsi = robotsi[range(int(len(robotsi)*prop_robots))]
        votesi2 = np.ravel(votesi[robotsi,prd2,:])
        rlti = pd.DataFrame.from_dict(Counter(np.ravel(votesi2)), orient='index', columns=['count']).sort_values('count',ascending=False)
        rlti['code'] = raws['close'].columns[rlti.index]
        rlts.append(rlti)
    rlt = pd.concat(rlts,axis=0).groupby('code').agg({'count':'sum'}).sort_values('count',ascending=False)
    rlt['index'] = rlt['count']/(np.sum(rlt['count'])/today.shape[1])
    rlt['code'] = rlt.index
    return(rlt)

##########################################################################################
# Modeling with one codelist
##########################################################################################

codefile = 'qs2'
codefilter = False

#Read codelist

codelist = []
datelist = []
with open(f'data/code_{codefile}.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        codelist.append(line.split(','))
        datelist.append(line.split(',')[0])

#Parameter

device = torch.device('cuda')
prd1 = 20
seeds = [303,777,101,602]
hidden_dim = 1024
latent_dim = 128
dropout_rate = 0.5
l2_reg = 0.01
num_epochs = 10000
lr = 0.001
early_tol = 1.1
patience = 20
patience2 = 10
num_robots = 10000
prop_votes = 0.05
prop_robots = 0.1

#Data

datai = (np.asarray(datelist)!='20230428').argsort()[0]
codes1 = codelist[datai]
codes0 = codelist[datai-1]
date1 = codes1[0] #buy date
date0 = codes0[0] #data ending date
date2 = codelist[datai+1][0]
printlog(date1)
codes2try = np.unique(np.ravel(np.append(codes1[1:],codes0[1:])))
raw = loaddata(date0,codes2try)

#Model

datasets,X,Y,Z,X2,Zscaler,raws = process2(raw,prd1,seeds)
models = []
for i in range(len(datasets)):
    model = train(i, hidden_dim, latent_dim, dropout_rate, l2_reg, lr, early_tol, patience, patience2)
    models.append(model)

#Voting

votes = roboting(num_robots,models)
rlt = voting(votes,prop_votes,prop_robots)
rlt = rlt[rlt['index']>5]
rlt['share'] = rlt['count'] / np.sum(rlt['count']) 
rlt['date'] = date1

#Calculation

transi = rlt
refi = []
for codei in transi['code']:
    refii = np.ravel(ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=date1, end_date=date2, adjust="qfq").iloc()[:,[1,2]])
    if len(refii)==2:
        refii = np.append(refii,[refii[1],refii[1]])
    refi.append(refii)

refi = pd.DataFrame(np.asarray(refi))
if refi.shape[1]==4:
    refi.columns = ['open0','close0','open1','close1']
else: 
    refi = pd.concat((refi,refi),axis=1)
    refi.iloc()[:,2] = refi.iloc()[:,1]
    refi.iloc()[:,3] = refi.iloc()[:,1]
    refi.columns = ['open0','close0','open1','close1']

refi.index = transi.index
transi = pd.concat([transi,refi],axis=1)
print(f'{np.unique(transi["date"])}, n:{transi.shape[0]}, ggProfit:{np.sum(transi.open1/transi.open0*transi.share):.4f},ApeProfit:{np.sum(transi.close1/transi.open0*transi.share):.4f}')
