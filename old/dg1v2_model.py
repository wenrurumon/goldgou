
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
import akshare as ak

##########################################################################################
#Module
##########################################################################################

def processdata(arg1,prd1,seeds):
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
    Xclose = []
    for i in range(closepvt.shape[0]-(prd1-1)):
        xi = closepvt[range(i, i + (prd1-1)), :] / closepvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xclose.append(np.ravel(xi.T))
    Xclose = np.asarray(Xclose)
    Xvol = []
    for i in range(volpvt.shape[0]-(prd1-1)):
        xi = volpvt[range(i, i + (prd1-1)), :] / volpvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xvol.append(np.ravel(xi.T))
    Xvol = np.asarray(Xvol)
    X = np.concatenate((Xclose,Xvol),axis=1)
    Z = []
    for i in range(closepvt.shape[0]-prd1-5):
        profi = openpvt[range(i+prd1+1,i+prd1+6),:].mean(axis=0)/openpvt[i+prd1,None,:]
        Z.append(profi)
    Z = np.concatenate(Z,axis=0)
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
    life = []
    for i in range(10, closepvt.shape[0]):
        data = closepvt[(i-9):(i+1), :]
        col_mean = np.mean(data, axis=0)
        life.append(col_mean)
    life = np.array(life)
    profit = (closepvt/openpvt)[-5:,:]
    back = (lowpvt/openpvt)[-5:,:]
    life = life[-5:,:]/life[-6:-1]
    datasets = []
    for seed in seeds:
        np.random.seed(seed)
        samples = np.random.permutation(np.ravel(range(X.shape[0])))
        samples = np.ravel((samples%5).tolist())
        for s in range(5):
            X_train,Y_train,Z_train,X_test,Y_test,Z_test=X[samples!=s,:],Y[samples!=s,:],Z[samples!=s,:],X[samples==s,:],Y[samples==s,:],Z[samples==s,:]
            datasets.append([X_train,Y_train,Z_train,X_test,Y_test,Z_test])
    return(datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes,closepvt,openpvt)

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

def train(modeli,hidden_dim,latent_dim,dropout_rate,l2_reg,lr,early_tol,patience,patience2,momentum):
    X_train,Y_train,Z_train,X_test,Y_test,Z_test = datasets[modeli]
    X_dim = X_train.shape[1]
    Y_dim = Y_train.shape[1]
    Z_dim = Z_train.shape[1]
    # Model 0
    m = 0
    num_epochs = 10000
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    counter2 = 0
    best_loss = np.inf
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=lr*0.1)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            lossz = criterion(ztr, zhat)
            lossy = criterion(ytr, yhat)
            wwz = epoch/1000
            wz = wwz*lossz / (wwz*lossz+lossy)
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
        # if epoch==0:
            # printlog(f'Model {modeli}.{m} training start, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
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
                # printlog(f'Model {modeli}.{m}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2
                optimizer.param_groups[0]['momentum'] = optimizer.param_groups[0]['momentum'] * 0.9
                counter2 += 1
                model.load_state_dict(best_model_state_dict)
            if counter2 > patience2:
                model.load_state_dict(best_model_state_dict)
                with torch.no_grad():
                    yhate, zhate, l2_losse = model(X_test)
                    vlossz = criterion(Z_test, zhate)
                    vlossy = criterion(Y_test, yhate)
                    vloss = wy*vlossy + wz*vlossz + l2_losse
                # printlog(f"Model {modeli}.{m} training stop at epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
                break    
        # if (epoch+1)%1000 == 0:
            # printlog(f"Model {modeli}.{m} training at epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
    # Model 1
    X_train,Y_train,Z_train,X_test,Y_test,Z_test = X[-200:,:],Y[-200:,:],Z[-200:,:],X[-100:,:],Y[-100:,:],Z[-100:,:]
    m = 1
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    counter2 = 0
    best_loss = np.inf
    # printlog(f"Model {modeli}.{m} training start")
    for epoch in range(epoch,num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            lossz = criterion(ztr, zhat)
            lossy = criterion(ytr, yhat)
            wz = 5*lossz / (5*lossz+lossy)
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
        if epoch>0:
            if vloss < best_loss*early_tol:
                if vloss < best_loss:
                    best_loss = vloss
                    best_model_state_dict = model.state_dict()
                    counter = 0
                else:
                    counter += .1
            else:
                counter += 1
            if counter >= patience:
                # printlog(f'Model {modeli}.{m}, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
                counter = 0
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2
                optimizer.param_groups[0]['momentum'] = optimizer.param_groups[0]['momentum'] * 0.9
                counter2 += 1
                model.load_state_dict(best_model_state_dict)
            if counter2 > patience2:
                model.load_state_dict(best_model_state_dict)
                with torch.no_grad():
                    yhate, zhate, l2_losse = model(X_test)
                    vlossz = criterion(Z_test, zhate)
                    vlossy = criterion(Y_test, yhate)
                    vloss = wy*vlossy + wz*vlossz + l2_losse
                # printlog(f"Model {modeli}.{m} training stop at epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
                models.append(model)
                break   
            # if (epoch+1)%1000 == 0:
                # printlog(f"Model {modeli}.{m} training at epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]") 
    return(model)

def roboting(num_robots,models):
    votes = []
    for modeli in models:
        votei = []
        for s in range(int(num_robots/len(models))):
            torch.manual_seed(s)
            Y2, Z2, _ = modeli(X2)
            Z2 = Zscaler.inverse_transform(Z2.cpu().detach().numpy())
            votei.append((-Z2).argsort(axis=1))
        votes.append(np.asarray(votei))
    votes = np.asarray(votes)
    return(votes)

def voting(votes,prop_votes,prop_robots,w,hat_inv):
    votes = votes.reshape((np.prod(votes.shape[0:2]), 6, len(codes)))[:,:,range(int(prop_votes * len(codes)))]
    w = (w/np.sum(w)).reshape(5,1)
    scores = []
    for votei in votes:
        scorei = []
        for i in range(5):
            scorei.append([np.mean(profit[i,votei[i]]),np.mean(life[i,votei[i]]),np.mean(back[i,votei[i]])])
        scorei = np.asarray(scorei)
        scorei = np.append((scorei * w).sum(axis=0),np.min(scorei[:,2]))
        scorei = np.append(scorei,(scorei[range(2)])/(scorei[2]))
        scores.append(scorei)
    scores = pd.DataFrame(np.asarray(scores))
    scores.columns = ['profit','life','back','minback','avgprofit','avglife']
    scores['pl'] = scores['profit'] * scores['back']
    scores['apl'] = scores['avgprofit'] * scores['avglife']
    scores = pd.DataFrame((-np.asarray(scores)).argsort(axis=0))
    scores.columns = ['profit','life','back','minback','avgprofit','avglife','pl','apl']
    scores = scores.head(int(votes.shape[0]*prop_robots))
    votes2 = []
    for i in range(scores.shape[0]):
        votes2.append(codes[votes[scores['life'][i]][5]])
    votes2 = np.ravel(votes2)
    rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes2)), orient='index', columns=['count']).sort_values('count',ascending=False)
    rlt['codes'] = rlt.index
    rlt['date'] = arg1
    rlt['idx'] = rlt['count']/len(scores) * hat_inv
    rlt = rlt[np.cumsum(rlt['idx'])<1]
    rlt['share'] = rlt['count']/np.sum(rlt['count'])
    return(rlt)

##########################################################################################
#Arguments
##########################################################################################

# arg1 = '20230428'
arg1 = int(sys.argv[1])
# prd1 = 40
prd1 = int(sys.argv[2])
# note = 'test'
note = datetime.datetime.now().strftime("%y%m%d%H%M")

if(note=='test'):
  def printlog(x):
    print(datetime.datetime.now(), x)
else:
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s %(message)s', 
        filename=f'log/dg1v2_{arg1}_{note}.log',  
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
#Modeling
##########################################################################################

#Parametering

seeds = [303,777]
hidden_dim = 1024
latent_dim = 128
dropout_rate = 0.5
l2_reg = 0.01
lr = 0.01
early_tol = 1.1
patience = 10
patience2 = 10
momentum = 0.9
prop_votes = 0.05
prop_robots = 0.1
w = [1,2,3,4,5]
hat_inv = 0.1

#Process data

datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes,closepvt,openpvt = processdata(arg1,prd1,seeds)

#Modeling

models = []
for i in range(len(datasets)):
    printlog(f'Model training @ {i}')
    modeli = train(i,hidden_dim,latent_dim,dropout_rate,l2_reg,lr,early_tol,patience,patience2,momentum)
    models.append(modeli)

#Voting

votes = roboting(10000,models)
np.savez(f'rlt/dg1v2_{arg1}_{prd1}_{note}.npz',votes=votes)
trans = voting(votes[:,range(50),:,:],prop_votes,prop_robots,w,hat_inv)
printlog([seeds,hidden_dim,latent_dim,dropout_rate,l2_reg,lr,early_tol,patience,patience2,momentum,prop_votes,prop_robots,w,hat_inv])
printlog(trans)

##########################################################################################
#Validation
##########################################################################################

#Validate

device = torch.device("cpu")
prop_votes = 0.05
prop_robots = 0.1
hat_inv = 0.1
w = [1,2,3,4,5]

rlts = []
rltfiles = np.sort(os.listdir('rlt'))
for rltfile in rltfiles:
    if 'dg1v2_' in rltfile:
        method,arg1,prd1,note = rltfile.split('.')[0].split('_')
        print(arg1)
        datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes,closepvt,openpvt = processdata(int(arg1),int(prd1),[1])
        votes = np.load(f'rlt/{rltfile}',allow_pickle=True)['votes']
        rlt = voting(votes[:,range(50),:,:],prop_votes,prop_robots,w,hat_inv)
        rlts.append(rlt)

trans = pd.concat(rlts,axis=0)

ref = []
for i in np.unique(trans.codes.tolist()):
    codei = str(i+1000000)[-6:]
    refi = ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=min(trans.date), end_date=max(trans.date), adjust="")
    refi = refi.iloc()[:,range(3)]
    refi.columns = ['date','open','close']
    refi['codes'] = i
    ref.append(refi)

ref = pd.concat(ref,axis=0)
ref['date'] = pd.to_datetime(ref['date']).dt.strftime('%Y%m%d')
ref['did'] = ref['date'].rank(method='dense').astype(int)

trans = pd.merge(trans,ref,on=['date','codes'])
trans['did'] = trans['date'].rank(method='dense').astype(int)+1
trans = pd.merge(trans,ref.loc[:,['did','codes','open']].rename(columns={'open':'open1'}),on=['did','codes'],how='left')
trans = trans.loc[:,['codes','date','share','open','close','open1']]
trans['open1'] = np.where(trans['open1'].isna(), trans['close'], trans['open1'])

rlt = pd.merge(
    trans.groupby('date').apply(lambda x:(x['close']/x['open']*x['share']).sum()).reset_index(name='today')[['date', 'today']],
    trans.groupby('date').apply(lambda x:(x['open1']/x['close']*x['share']).sum()).reset_index(name='overnite')[['date', 'overnite']]
)
rlt['profit'] = rlt.today * rlt.overnite
rlt['cumprofit'] = np.cumprod(rlt.profit)
[prop_votes,prop_robots,hat_inv]
rlt

##########################################################################################
#Rolling
##########################################################################################

#Rolling

import os
import numpy as np
files = []
for i in np.sort(os.listdir('data')):
    if 'raw' in i:
        files.append(i.replace('raw','').replace('.csv',''))

for arg1 in files:
    syntax = f'python dg1v2_model.py {arg1} 40'
    print(syntax)
    os.system(syntax)
