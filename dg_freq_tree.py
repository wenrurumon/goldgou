
import sklearn.linear_model
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

def fill_missing_values(row):
    return row.fillna(method='ffill')

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
        with open(f'data/code_qs2.txt','r') as file:
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

def loaddata(codes):
    raw = []
    for codei in codes:
        rawi = ak.stock_zh_a_hist_min_em(symbol=codei,start_date='2022-09-01 09:30:00',end_date=datetime.datetime.now(),adjust='qfq')
        rawi.columns = ['date','open','close','high','low','pricechp','pricech','vol','val','var','change']
        rawi['code'] = codei
        raw.append(rawi)
    raw = pd.concat(raw,axis=0)
    raw['date'] = pd.to_datetime(raw['date'])
    raw['date'] = raw['date'].dt.strftime('%y%m%d%H%M')
    raw = raw.drop_duplicates()
    rawsel = raw.groupby('code')['date'].min().reset_index(name='mindate')
    mindate = rawsel['mindate'].mode().iloc[0]
    codesel = np.ravel(rawsel[rawsel['mindate'] <= mindate]['code'])
    raw = raw[raw['code'].isin(codesel) & (raw['date'] >= mindate)]
    raws = []
    for i in ['open', 'close', 'high', 'low', 'pricechp', 'pricech', 'vol', 'val', 'var', 'change']:
        rawi = pd.pivot_table(raw, values=i, index=['date'], columns=['code'])
        if i == 'close':
            rawi = rawi.apply(fill_missing_values, axis=0)
        else:
            rawi = rawi.fillna(0)
        rawi['date'] = rawi.index
        rawi = pd.melt(rawi, id_vars='date', var_name='code', value_name='value')
        rawi['variable'] = i
        raws.append(rawi)
    raw = pd.concat(raws, axis=0)
    raw['time'] = raw['date']
    raw['date'] = raw['time'].astype(int) // 10000
    raw['time'] = raw['time'].astype(int) // 1
    raw = pd.pivot_table(raw, values='value', index=['date', 'code', 'time'], columns='variable').reset_index()
    raw.loc[raw['val'] == 0, ['open', 'high', 'low']] = raw.loc[raw['val'] == 0, 'close']
    raw = pd.melt(raw, id_vars=['date', 'code', 'time'])
    return(raw)

def getXY(i,time0=1130,prd1=5,prd2=2):
    date0 = datelist[i]
    date1 = datelist[i + prd1]
    date2 = datelist[i + prd1 + prd2]
    X = raw[(raw['time'] >= date0 * 10000 + time0) & (raw['time'] < date1 * 10000 + time0)]
    X = pd.merge(X, X.groupby(['code', 'variable'])['value'].mean().reset_index(), on=['code', 'variable'])
    X['value_y'] = np.where(X['value_y'] == 0, 1, np.abs(X['value_y']))
    X['value'] = X['value_x'] / X['value_y']
    X = X.drop(['value_x', 'value_y'], axis=1)
    X['date'] = X['date'].apply(
        lambda x: datelist[i:i + prd1 + prd2 + 1].tolist().index(x) if x in datelist[i:i + prd1 + prd2 + 1] else None)
    X['time'] = X.apply(lambda row: row['date'] * 10000 + row['time'] % 10000, axis=1)
    X['variable_time'] = X['variable'] + '@' + X['time'].astype(str)
    X = pd.pivot_table(X, values='value', index='code', columns='variable_time')
    X['date0'] = date0
    X['date1'] = date1
    X['date2'] = date2
    Y = raw[
        (raw['time'] >= date1 * 10000 + time0) & (raw['time'] < date2 * 10000 + time0) & (raw['variable'] == 'close')]
    Y = pd.merge(Y, Y.groupby(['code', 'variable'])['value'].mean().reset_index(), on=['code', 'variable'])
    Y['value'] = Y['value_x'] / Y['value_y']
    Y = Y.drop(['value_x', 'value_y'], axis=1)
    Y['date'] = Y['date'].apply(
        lambda x: datelist[i:i + prd1 + prd2 + 1].tolist().index(x) if x in datelist[i:i + prd1 + prd2 + 1] else None)
    Y['time'] = Y.apply(lambda row: row['date'] * 10000 + row['time'] % 10000, axis=1)
    Y['variable_time'] = Y['variable'] + '@' + Y['time'].astype(str)
    Y = pd.pivot_table(Y, values='value', index='code', columns='variable_time')
    Y['date0'] = date0
    Y['date1'] = date1
    Y['date2'] = date2
    return(X.reset_index(),Y.reset_index())

def getX(i,time0=1130,prd1=5,prd2=2):
    date0 = datelist[i]
    date1 = datelist[i + prd1]
    date2 = datelist[i + prd1 + prd2]
    X = raw[(raw['time'] >= date0 * 10000 + time0) & (raw['time'] < date1 * 10000 + time0)]
    X = pd.merge(X, X.groupby(['code', 'variable'])['value'].mean().reset_index(), on=['code', 'variable'])
    X['value_y'] = np.where(X['value_y'] == 0, 1, np.abs(X['value_y']))
    X['value'] = X['value_x'] / X['value_y']
    X = X.drop(['value_x', 'value_y'], axis=1)
    X['date'] = X['date'].apply(
        lambda x: datelist[i:i + prd1 + prd2 + 1].tolist().index(x) if x in datelist[i:i + prd1 + prd2 + 1] else None)
    X['time'] = X.apply(lambda row: row['date'] * 10000 + row['time'] % 10000, axis=1)
    X['variable_time'] = X['variable'] + '@' + X['time'].astype(str)
    X = pd.pivot_table(X, values='value', index='code', columns='variable_time')
    X['date0'] = date0
    X['date1'] = date1
    X['date2'] = date2
    return(X.reset_index())

def getY(i,time0=1130,prd1=5,prd2=2):
    date0 = datelist[i]
    date1 = datelist[i + prd1]
    date2 = datelist[i + prd1 + prd2]
    Y = raw[(raw['time'] >= date1 * 10000 + time0) & (raw['time'] < date2 * 10000 + time0) & (raw['variable'] == 'close')]
    Y = pd.merge(Y, Y.groupby(['code', 'variable'])['value'].mean().reset_index(), on=['code', 'variable'])
    Y['value'] = Y['value_x'] / Y['value_y']
    Y = Y.drop(['value_x', 'value_y'], axis=1)
    Y['date'] = Y['date'].apply(
        lambda x: datelist[i:i + prd1 + prd2 + 1].tolist().index(x) if x in datelist[i:i + prd1 + prd2 + 1] else None)
    Y['time'] = Y.apply(lambda row: row['date'] * 10000 + row['time'] % 10000, axis=1)
    Y['variable_time'] = Y['variable'] + '@' + Y['time'].astype(str)
    Y = pd.pivot_table(Y, values='value', index='code', columns='variable_time')
    Y['date0'] = date0
    Y['date1'] = date1
    Y['date2'] = date2
    return(Y.reset_index())

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

################################################
# Loaddata
################################################

codelist = updatecodes()
date0 = codelist.tradedates[(np.asarray(codelist.tradedates)!=codelist.jg_date[len(codelist.jg_date)-2]).argsort()[0]]
date0,date1,date2 = codelist.getdates(date0)
codes = list(set([elem for sublist in list(codelist.getcodes(date1).values())[:2] for elem in sublist]))
codes2 = list(set([elem for sublist in list(codelist.getcodes(date1).values())[2:] for elem in sublist]))
raw = loaddata(codes)
raw = raw[raw['variable'].isin(['change', 'close'])]
datelist = np.sort(np.unique(raw.date))
datelist = np.asarray(codelist.tradedates).astype(float)[np.where(np.asarray(codelist.tradedates).astype(float)==min(raw.date)+20000000)[0][0]:]
datelist = datelist - 20000000

time0 = 930
prd1 = 3
prd2 = 2

Xs = []
Ys = []
Maps = []
for i in range(31-prd1-prd2):
    Xi,Yi = getXY(i,time0,prd1,prd2)
    Maps.append(Xi[['code','date0','date1','date2']])
    Xi = Xi.iloc[:, 1:(Xi.shape[1] - 3)]
    Xs.append(Xi)
    Yi = Yi.iloc[:, 1:(Yi.shape[1]-3)]
    Ys.append(Yi)

for i in range(i,31):
    Xi = getX(i,time0,prd1,prd2)
    Maps.append(Xi[['code','date0','date1','date2']])
    Xi = Xi.iloc[:, 1:(Xi.shape[1] - 3)]
    Xs.append(Xi)

Xs = np.asarray(pd.concat(Xs,axis=0))
Xs = np.nan_to_num(Xs, nan=1)
Ys = np.asarray(pd.concat(Ys,axis=0))
Maps = np.asarray(pd.concat(Maps,axis=0))

X = Xs
Y = (Ys[:,-1]/Ys[:,0])[:,np.newaxis]

#Tree

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
Xscaler = StandardScaler()
Xscaler.fit(X)
X = Xscaler.transform(X)
Yscaler = StandardScaler()
Yscaler.fit(Y)
Y = Yscaler.transform(Y)
Y = Y > np.quantile(Y,0.5)
X2 = X[range(Y.shape[0]),:]

models = []
for seedi in range(1000):
    X_train, X_test, Z_train, Z_test = train_test_split(X2, Y, test_size=0.3, random_state=seedi)
    model = DecisionTreeClassifier(random_state=seedi,max_depth=int(X_train.shape[1]*0.1), min_samples_leaf=int(X_train.shape[0]*0.01))
    model.fit(X_train, Z_train)
    print(datetime.datetime.now(),np.mean(model.predict(X_train)==np.ravel(Z_train)),np.mean(model.predict(X_test)==np.ravel(Z_test)))
    models.append(model)

#Voting

buydate = 230707
rlts = []
for model in models:
    rlts.append(model.predict(X[Maps[:,2]==buydate,:]))

rlt = pd.DataFrame({
    'code':Maps[Maps[:,2]==buydate,0],
    'win':np.asarray(rlts).sum(axis=0),
    'buydate':buydate*10000+930
    }).sort_values('win',ascending=False)

rlt['win2'] = rlt['win']
rlt.loc[~rlt['code'].isin(codes2), 'win2'] = 0
rlt['tval'] = (rlt['win']-np.mean(rlt['win']))/np.std(rlt['win'])
rlt1 = rlt[rlt['tval']>2]
rlt1['share'] = rlt1['win']/np.sum(rlt1['win'])
rlt['tval2'] = (rlt['win2']-np.mean(rlt['win2']))/np.std(rlt['win2'])
rlt = rlt[rlt['win']==rlt['win2']]
rlt2 = rlt[(rlt['tval2']>2)&(rlt['tval']>1)]
rlt2['share'] = rlt2['win2']/np.sum(rlt2['win2'])
