
import torch
import pandas as pd
import numpy as np
import sys
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler
import datetime   
from datetime import timedelta
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
import statsmodels.api as sm

def printlog(x):
    print(datetime.datetime.now(), x)

def loaddata(date0,codes):
    raw = []
    for codei in codes:
        rawi = ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=int(date0)-20000, end_date=int(date0), adjust="qfq")
        rawi.columns = ['date','open','close','high','low','pricechp','pricech','vol','val2','var','val']
        rawi = rawi.iloc()[:,[0,1,2,3,4,7,10]]
        # rawi = ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=int(date0)-20000, end_date=int(date0), adjust="qfq").iloc()[:,range(7)]
        # rawi.columns = ['date','open','close','high','low','vol','val']
        rawi['code'] = codei
        raw.append(rawi)
    raw = pd.concat(raw,axis=0)
    raw = raw.drop('vol', axis=1)
    return(raw)

def fill_missing_values(row):
    return row.fillna(method='ffill')

def process1(raw,prd1,prd2,seeds):
    # prd2 = 5
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
    # Xhigh = []
    # for i in range(highpvt.shape[0]-(prd1-1)):
    #    xi = highpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
    #    xi = np.nan_to_num(xi,nan=-1)
    #    xi = xi[-5:,:]
    #    Xhigh.append(np.ravel(xi.T))
    # Xhigh = np.asarray(Xhigh)
    # Xlow = []
    # for i in range(lowpvt.shape[0]-(prd1-1)):
    #    xi = lowpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
    #    xi = np.nan_to_num(xi,nan=-1)
    #    xi = xi[-5:,:]
    #    Xlow.append(np.ravel(xi.T))
    # Xlow = np.asarray(Xlow)
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
    # X = np.concatenate((Xclose,Xval2),axis=1)
    X = np.concatenate((Xclose,Xval2,Xclose*Xval2),axis=1)
    P = []
    for i in range(closepvt.shape[0]-prd1-prd2):
        profi = (np.exp(closepvt[range(i+prd1,i+prd1+prd2),:])-1).mean(axis=0)/(np.exp(openpvt[i+prd1,None,:])-1)
        # profi = openpvt[range(i+prd1+1,i+prd1+prd2+1),:].mean(axis=0)/openpvt[i+prd1,None,:]
        P.append(profi)
    P = np.concatenate(P,axis=0)
    # for i in range(closepvt.shape[0]-prd1-prd2):
    #     close0 = np.exp(closepvt[range(i+prd1-1,i+prd1+prd2-1),:])-1
    #     open1 = np.exp(openpvt[range(i+prd1,i+prd1+prd2),:])-1
    #     close1 = np.exp(closepvt[range(i+prd1,i+prd1+prd2),:])-1
    #     high1 = np.exp(highpvt[range(i+prd1,i+prd1+prd2),:])-1
    #     low1 = np.exp(lowpvt[range(i+prd1,i+prd1+prd2),:])-1
    #     roi1 = close1+0
    #     roi1[0,:] = roi1[0,:]/open1[0,:]
    #     roi1[1:,:] = roi1[1:,:]/close1[range(4),:]
    #     roi1[high1==low1] = np.where(roi1[high1==low1]>1,1,roi1[high1==low1])
    #     P.append(roi1.prod(axis=0))
    # P = np.asarray(P)
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
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=0.3, random_state=seed)
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

def process2(raw,prd1,prd2,seeds):
    # prd2 = 5
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
    # Xhigh = []
    # for i in range(highpvt.shape[0]-(prd1-1)):
    #    xi = highpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
    #    xi = np.nan_to_num(xi,nan=-1)
    #    xi = xi[-5:,:]
    #    Xhigh.append(np.ravel(xi.T))
    # Xhigh = np.asarray(Xhigh)
    # Xlow = []
    # for i in range(lowpvt.shape[0]-(prd1-1)):
    #    xi = lowpvt[range(i, i + (prd1)), :] / closepvt[i + (prd1-1), None, :]
    #    xi = np.nan_to_num(xi,nan=-1)
    #    xi = xi[-5:,:]
    #    Xlow.append(np.ravel(xi.T))
    # Xlow = np.asarray(Xlow)
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
    # X = np.concatenate((Xclose,Xval2),axis=1)
    X = np.concatenate((Xclose,Xval2,Xclose*Xval2),axis=1)
    P = []
    # for i in range(closepvt.shape[0]-prd1-prd2):
        # profi = openpvt[range(i+prd1+1,i+prd1+prd2+1),:].mean(axis=0)-openpvt[i+prd1,None,:]
        # profi = openpvt[range(i+prd1+1,i+prd1+prd2+1),:].mean(axis=0)/openpvt[i+prd1,None,:]
        # P.append(profi)
    # P = np.concatenate(P,axis=0)
    for i in range(closepvt.shape[0]-prd1-prd2):
        close0 = np.exp(closepvt[range(i+prd1-1,i+prd1+prd2-1),:])-1
        open1 = np.exp(openpvt[range(i+prd1,i+prd1+prd2),:])-1
        close1 = np.exp(closepvt[range(i+prd1,i+prd1+prd2),:])-1
        high1 = np.exp(highpvt[range(i+prd1,i+prd1+prd2),:])-1
        low1 = np.exp(lowpvt[range(i+prd1,i+prd1+prd2),:])-1
        roi1 = close1+0
        roi1[0,:] = roi1[0,:]/open1[0,:]
        roi1[1:,:] = roi1[1:,:]/close1[range(4),:]
        roi1[high1==low1] = np.where(roi1[high1==low1]>1,1,roi1[high1==low1])
        P.append(roi1.prod(axis=0))
    P = np.asarray(P)
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
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=0.3, random_state=seed)
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
            wwz = 1
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
            votei.append(Z2)
            # votei.append(((-Z2).argsort(axis=1)))
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

class updatecodes:
    def __init__(self):
        jg_date = []
        jg_codes = []
        with open(f'data/code_jg.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                jg_date.append(line.split(',')[0])
                jg_codes.append(','.join(line.split(',')[1:]).replace('\n',''))
        qs_date = []
        qs_codes = []
        with open(f'data/code_qs.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                qs_date.append(line.split(',')[0])
                qs_codes.append(','.join(line.split(',')[1:]).replace('\n',''))
        jg2_date = []
        jg2_codes = []
        with open(f'data/code_jg2.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                jg2_date.append(line.split(',')[0])
                jg2_codes.append(','.join(line.split(',')[1:]).replace('\n',''))
        qs2_date = []
        qs2_codes = []
        with open(f'data/code_qs.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                qs2_date.append(line.split(',')[0])
                qs2_codes.append(','.join(line.split(',')[1:]).replace('\n',''))
        jg_date = np.asarray(jg_date)
        qs_date = np.asarray(qs_date)
        jg_codes = np.asarray(jg_codes)
        qs_codes = np.asarray(qs_codes)
        jg2_date = np.asarray(jg2_date)
        qs2_date = np.asarray(qs2_date)
        jg2_codes = np.asarray(jg2_codes)
        qs2_codes = np.asarray(qs2_codes)
        self.jg_date = jg_date
        self.jg_codes = jg_codes
        self.qs_date = qs_date
        self.qs_codes = qs_codes
        self.jg2_date = jg2_date
        self.jg2_codes = jg2_codes
        self.qs2_date = qs2_date
        self.qs2_codes = qs2_codes
        self.tradedates = [d.strftime('%Y%m%d') for d in np.ravel(ak.tool_trade_date_hist_sina())]
    def getdates(self,date0):
        date1 = (np.asarray(self.tradedates)!=str(date0)).argsort()[0]
        return(np.asarray(self.tradedates)[range(date1,date1+3)].tolist())
    def getcodes(self,datei):
        raws = []
        raws.append(self.jg_codes[self.jg_date==str(datei)].tolist()[0].split(','))
        raws.append(self.qs_codes[self.qs_date==str(datei)].tolist()[0].split(','))
        raws.append(self.jg2_codes[self.jg2_date==str(datei)].tolist()[0].split(','))
        raws.append(self.qs2_codes[self.qs2_date==str(datei)].tolist()[0].split(','))
        raws = dict(zip(['jg','qs','jg2','qs2'],raws))
        return(raws)

def train2(X, Y, Z, seed, hidden_dim, latent_dim, dropout_rate, l2_reg, lr, early_tol, patience, patience2):
    weights = [math.log(math.ceil((i+1)/20)) for i in range(X.shape[0])]
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, W_train, W_test = train_test_split(X, Y, Z, weights, test_size=0.3, random_state=seed)
    X_dim = X_train.shape[1]
    Y_dim = Y_train.shape[1]
    Z_dim = Z_train.shape[1]
    # Model 0
    m = 0
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    sampler = WeightedRandomSampler(W_train,len(W_train))
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
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
            wwz = 1
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
            printlog(f'Model {seed}.{m} training, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
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
    printlog(f"Model {seed}.{m} stop, Epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
    return(model)

##########################################################################################
# Trail
##########################################################################################

note = 'dg0'
codelist = updatecodes()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seeds = [303,101,602,603,4,7,10,11,14,49]
hidden_dim = 1024
latent_dim = 128
dropout_rate = 0.5
l2_reg = 0.01
num_epochs = 10000
lr = 0.001
early_tol = 1.1
patience = 20
patience2 = 10
num_robots = 1000
if note=='dg0':
    process = process1
else:
    process = process2

prd1 = 40
prd2 = 5
range0 = 1
prop_votes = 0.05
prop_robots = 0.1

#Loaddata
date0 = codelist.tradedates[(np.asarray(codelist.tradedates)!=codelist.jg_date[len(codelist.jg_date)-2]).argsort()[0]]
date0,date1,date2 = codelist.getdates(date0)
print(f'know @ {date0}, buy @ {date1}, valid @ {date2}')
codes = list(set([elem for sublist in list(codelist.getcodes(date1).values())[:2] for elem in sublist]))
codes2 = list(set([elem for sublist in list(codelist.getcodes(date1).values())[2:] for elem in sublist]))
raw = loaddata(date0,codes)
for i in raw.columns.tolist()[1:6]:
    raw[i] = np.log(raw[i]+1)

#Modeling
datasets,X,Y,Z,X2,Zscaler,raws = process(raw,prd1,prd2,seeds)
votess = []
for trail in range(1):
    print(f'trail {trail} @ {datetime.datetime.now()}')
    models = []
    for i in range(len(seeds)):
        model = train2(X, Y, Z, seeds[i], hidden_dim, latent_dim, dropout_rate, l2_reg, lr, early_tol, patience, patience2)
        models.append(model)
    votes = roboting(num_robots*len(models),models)
    votess.append(votes)

pf2test = np.asarray(pd.concat((raws['close'].iloc()[-(votes.shape[2]-2):,:],raws['close'].iloc()[-1:,:]),axis=0))/np.asarray(raws['open'].iloc()[-(votes.shape[2]-1):,:])
pf2test = (pf2test-1)/2+pf2test
np.savez(f'model/{note}_{date1}.npz',votes=np.asarray(votess),codes=np.ravel(raws['close'].columns).tolist(),codes2=codes2,pf2test=pf2test)

#Voting
file = np.load(f'model/{note}_{date1}.npz',allow_pickle=True)
codes = file['codes']
codes2 = file['codes2']
pf2test = file['pf2test']
for i in range(file['votes'].shape[0]):
    votes = np.concatenate(file['votes'][i],axis=0)
    rois = []
    fvotes = []
    for i in range(votes.shape[0]):
        votesi = votes[i]
        votesi = (votesi >= np.quantile(votesi,q=1-prop_votes,axis=1,keepdims=True))
        rois.append((votesi[range(prd2),:]*pf2test).sum(axis=1)/(votesi[range(prd2),:]).sum(axis=1))
        fvotes.append(votesi[-1,:])
    fvotes = np.asarray(fvotes)
    rois = (np.asarray(rois))
    w = np.asarray(np.ravel(range(prd2))).reshape(1,prd2)
    w = w/np.sum(w)
    rois = ((rois*w).sum(axis=1))
    rlt = votes[rois>=np.quantile(rois,1-prop_robots),prd2,:]
    rlt = pd.DataFrame({'code':codes,
        'mean':rlt.mean(axis=0),'sd':rlt.std(axis=0),
        'count':(rlt >= np.quantile(rlt,1-prop_votes,axis=1,keepdims=True)).sum(axis=0)}).sort_values(['count','mean'],ascending=False)
    rlt1 = rlt
    rlt = rlt[rlt['count']>=0]
    rlt['count'] = rlt.apply(lambda row: row['count'] if row['code'] in file['codes2'] else 0, axis=1)
    rlt2 = rlt
    for i in range(len(codes)):
        if (codes[i] not in codes2):
            votes[:,:,i] = -1
    rois = []
    fvotes = []
    for i in range(votes.shape[0]):
        votesi = votes[i]
        votesi = (votesi >= np.quantile(votesi,q=1-prop_votes,axis=1,keepdims=True))
        rois.append((votesi[range(prd2),:]*pf2test).sum(axis=1)/(votesi[range(prd2),:]).sum(axis=1))
        fvotes.append(votesi[-1,:])
    fvotes = np.asarray(fvotes)
    rois = (np.asarray(rois))
    w = np.asarray(np.ravel(range(prd2))).reshape(1,prd2)
    w = w/np.sum(w)
    rois = ((rois*w).sum(axis=1))
    rlt = votes[rois>=np.quantile(rois,1-prop_robots),prd2,:]
    rlt = pd.DataFrame({'code':codes,
        'mean':rlt.mean(axis=0),'sd':rlt.std(axis=0),
        'count':(rlt >= np.quantile(rlt,1-prop_votes,axis=1,keepdims=True)).sum(axis=0)}).sort_values(['count','mean'],ascending=False)
    rlt3 = rlt

rlt3['buydate'] = rlt2['buydate'] = rlt1['buydate'] = date1
rlt3['valdate'] = rlt2['valdate'] = rlt1['valdate'] = date2
rlt1['vote'] = 'rlt1'
rlt2['vote'] = 'rlt2'
rlt3['vote'] = 'rlt3'
rlt = pd.concat((rlt1,rlt2,rlt3),axis=0)
rlt = rlt.merge(rlt.groupby('vote').agg(index=('count', lambda x: np.quantile(x, 0.95))).reset_index(),on='vote')
rlt['index'] = rlt['count']/rlt['index']
rlt = rlt[(rlt['index']>1.5)&(rlt['mean']>0)]

#Resulting

temp = rlt[(rlt['index']>1.5)&(rlt['mean']>0)]
temp = temp.merge(temp.groupby('vote').apply(lambda x: (x['count'].sum())).reset_index(name='share'),on='vote')
temp['share'] = temp['count']/temp['share']
print(temp)

##########################################################################################
# Rolling
##########################################################################################

date0id = (np.asarray(codelist.tradedates)!='20230330').argsort()[0]
date1id = (np.asarray(codelist.tradedates)!='20230725').argsort()[0]

#Modeling Back

for date0 in codelist.tradedates[date0id:date1id]:
    date0,date1,date2 = codelist.getdates(date0)
    print(f'know @ {date0}, buy @ {date1}, valid @ {date2}')
    codes = list(set([elem for sublist in list(codelist.getcodes(date1).values())[:2] for elem in sublist]))
    codes2 = list(set([elem for sublist in list(codelist.getcodes(date1).values())[2:] for elem in sublist]))
    raw = loaddata(date0,codes)
    for i in raw.columns.tolist()[1:6]:
        raw[i] = np.log(raw[i]+1)
    #Modeling
    datasets,X,Y,Z,X2,Zscaler,raws = process(raw,prd1,prd2,seeds)
    votess = []
    for trail in range(1):
        print(f'trail {trail} @ {datetime.datetime.now()}')
        models = []
        for i in range(len(seeds)):
            model = train2(X, Y, Z, seeds[i], hidden_dim, latent_dim, dropout_rate, l2_reg, lr, early_tol, patience, patience2)
            models.append(model)
        votes = roboting(num_robots*len(models),models)
        votess.append(votes)
    pf2test = np.asarray(pd.concat((raws['close'].iloc()[-(votes.shape[2]-2):,:],raws['close'].iloc()[-1:,:]),axis=0))/np.asarray(raws['open'].iloc()[-(votes.shape[2]-1):,:])
    pf2test = (pf2test-1)/2+pf2test
    np.savez(f'model/{note}_{date1}.npz',votes=np.asarray(votess),codes=np.ravel(raws['close'].columns).tolist(),codes2=codes2,pf2test=pf2test)

 #Modeling Evaluation

rlts = []
roi0 = pd.DataFrame({'vote': ['rlt' + str(i) for i in range(1, 4)], 'roi0': 1})
for date0 in codelist.tradedates[date0id:date1id]:
    date0,date1,date2 = codelist.getdates(date0)
    print(f'know @ {date0}, buy @ {date1}, valid @ {date2}')
    codes = list(set([elem for sublist in list(codelist.getcodes(date1).values())[:2] for elem in sublist]))
    codes2 = list(set([elem for sublist in list(codelist.getcodes(date1).values())[2:] for elem in sublist]))
    # raw = loaddata(date0,codes)
    # for i in raw.columns.tolist()[1:6]:
    #     raw[i] = np.log(raw[i]+1)
    # datasets,X,Y,Z,X2,Zscaler,raws = process(raw,prd1,prd2,seeds)
    #Voting
    prd2 = 5
    file = np.load(f'model/{note}_{date1}.npz',allow_pickle=True)
    codes = file['codes']
    codes2 = file['codes2']
    pf2test = file['pf2test']
    for i in range(file['votes'].shape[0]):
        votes = np.concatenate(file['votes'][i],axis=0)
        rois = []
        fvotes = []
        for i in range(votes.shape[0]):
            votesi = votes[i]
            votesi = (votesi >= np.quantile(votesi,q=1-prop_votes,axis=1,keepdims=True))
            rois.append((votesi[range(prd2),:]*pf2test).sum(axis=1)/(votesi[range(prd2),:]).sum(axis=1))
            fvotes.append(votesi[-1,:])
        fvotes = np.asarray(fvotes)
        rois = (np.asarray(rois))
        w = np.asarray(np.ravel(range(prd2))).reshape(1,prd2)
        w = w/np.sum(w)
        rois = ((rois*w).sum(axis=1))
        rlt = votes[rois>=np.quantile(rois,1-prop_robots),prd2,:]
        rlt = pd.DataFrame({'code':codes,
            'mean':rlt.mean(axis=0),'sd':rlt.std(axis=0),
            'count':(rlt >= np.quantile(rlt,1-prop_votes,axis=1,keepdims=True)).sum(axis=0)}).sort_values(['count','mean'],ascending=False)
        rlt1 = rlt
        rlt = rlt[rlt['count']>=0]
        rlt['count'] = rlt.apply(lambda row: row['count'] if row['code'] in file['codes2'] else 0, axis=1)
        rlt2 = rlt
        for i in range(len(codes)):
            if (codes[i] not in codes2):
                votes[:,:,i] = -1
        rois = []
        fvotes = []
        for i in range(votes.shape[0]):
            votesi = votes[i]
            votesi = (votesi >= np.quantile(votesi,q=1-prop_votes,axis=1,keepdims=True))
            rois.append((votesi[range(prd2),:]*pf2test).sum(axis=1)/(votesi[range(prd2),:]).sum(axis=1))
            fvotes.append(votesi[-1,:])
        fvotes = np.asarray(fvotes)
        rois = (np.asarray(rois))
        w = np.asarray(np.ravel(range(prd2))).reshape(1,prd2)
        w = w/np.sum(w)
        rois = ((rois*w).sum(axis=1))
        rlt = votes[rois>=np.quantile(rois,1-prop_robots),prd2,:]
        rlt = pd.DataFrame({'code':codes,
            'mean':rlt.mean(axis=0),'sd':rlt.std(axis=0),
            'count':(rlt >= np.quantile(rlt,1-prop_votes,axis=1,keepdims=True)).sum(axis=0)}).sort_values(['count','mean'],ascending=False)
        rlt3 = rlt
    rlt3['buydate'] = rlt2['buydate'] = rlt1['buydate'] = date1
    rlt3['valdate'] = rlt2['valdate'] = rlt1['valdate'] = date2
    rlt1['vote'] = 'rlt1'
    rlt2['vote'] = 'rlt2'
    rlt3['vote'] = 'rlt3'
    ref = pd.concat((rlt1,rlt2,rlt3),axis=0)[['code','buydate','valdate']].drop_duplicates()
    rois = []
    for i in range(ref.shape[0]):
        codei = ref['code'][i]
        date1 = ref['buydate'][i]
        date2 = ref['valdate'][i]
        rawi = ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=int(date1), end_date=int(date2), adjust="qfq")
        if len(rawi)==0:
            rois.append(1)
        else:
            rawi.columns = ['date','open','close','high','low','pricechp','pricech','vol','val2','var','val']
            roii = np.ravel(rawi.iloc()[:,[1,2]])
            roii = roii[len(roii)-1]/roii[0]
            rois.append(roii)
    ref['roi'] = rois
    rlt = pd.concat((rlt1,rlt2,rlt3),axis=0).merge(ref, on=['code', 'buydate', 'valdate'])
    rlts.append(rlt)
    rlt = rlt.merge(rlt.groupby('vote').agg(index=('count', lambda x: np.quantile(x, 0.95))).reset_index(),on='vote')
    rlt['index'] = rlt['count']/rlt['index']
    roi = rlt[(rlt['index']>1.5)&(rlt['mean']>1)]
    roi = roi.groupby('vote').apply(lambda x: (x['roi'] * x['count']).sum() / x['count'].sum()).reset_index(name='roi')
    roi0 = roi0.merge(roi,on='vote',how='left')
    roi0['roi'] = roi0['roi'].fillna(1)
    roi0['roi0'] = roi0['roi'] * roi0['roi0']
    print(roi0)
    roi0 = roi0[['vote','roi0']]

rlts = pd.concat(rlts,axis=0)
rlts.to_csv(f'test_{note}.csv')

temp = rlts.merge(rlts.groupby(['buydate','vote']).agg(index=('count', lambda x:np.quantile(x,0.95))).reset_index(),on=['vote','buydate'])
temp['index'] = temp['count']/temp['index']
temp['count'] = 1 # np.sqrt(temp['count'])
temp = temp[(temp['index']>1.5) & (temp['mean']>1)]
temp = temp.groupby(['buydate','vote']).apply(lambda x: (x['roi']*x['count']).sum()/x['count'].sum()).reset_index(name='roi')
temp.groupby('vote').agg(roi=('roi',lambda x: np.prod(x)))


