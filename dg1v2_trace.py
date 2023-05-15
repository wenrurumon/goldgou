
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
#Modules
##########################################################################################

def processdata(arg1,prd1,seeds):
    # printlog(f'load data: data/raw{arg1}.csv')
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
    Xvol = []
    Xvol2 = []
    for i in range(volpvt.shape[0]-(prd1-1)):
        xi = volpvt[range(i, i + (prd1)), :] 
        xi = xi / (np.nan_to_num(xi, nan=0).mean(axis=1)).reshape(prd1, 1)
        # xi = volpvt[range(i, i + (prd1-1)), :] 
        # xi = volpvt[range(i, i + (prd1-1)), :] / volpvt[i + (prd1-1), None, :]
        xi = np.nan_to_num(xi,nan=-1)
        Xvol2.append(np.ravel(xi.T))
        xi = xi[-10:,:]
        Xvol.append(np.ravel(xi.T))
    Xvol = np.asarray(Xvol)
    Xvol2 = np.asarray(Xvol2)
    X = np.concatenate((Xclose,Xvol2),axis=1)
    P = []
    B = []
    for i in range(closepvt.shape[0]-prd1-5):
        profi = openpvt[range(i+prd1+1,i+prd1+6),:].mean(axis=0)/openpvt[i+prd1,None,:]
        backi = lowpvt[range(i+prd1+1,i+prd1+6),:].min(axis=0)/openpvt[i+prd1,None,:]
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
    profit = (closepvt/openpvt)[-5:,:]
    back = (lowpvt/openpvt)[-5:,:]
    life = life[-5:,:]/life[-6:-1]
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

def train(modeli, X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg, lr, early_tol, patience, patience2, momentum):
    X_train,Y_train,Z_train,X_test,Y_test,Z_test = datasets[modeli]
    # Model 0
    m = 0
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    counter2 = 0
    best_loss = np.inf
    model = Autoencoder(X_dim, Y_dim, Z_dim, hidden_dim, latent_dim, dropout_rate, l2_reg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    # optimizer = optim.Adam(model.parameters(), lr=lr*0.1)
    for epoch in range(num_epochs):
        for xtr, ytr, ztr in train_loader:
            xtr = xtr.float()
            ytr = ytr.float()
            yhat, zhat, l2_loss = model(xtr)
            lossz = criterion(ztr, zhat)
            lossy = criterion(ytr, yhat)
            wz = .5*lossz / (.5*lossz+lossy)
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
            # printlog(f'Model {modeli}.{m} start, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
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
                # printlog(f'Model {modeli}.{m} training, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
                X_train,Y_train,Z_train,X_test,Y_test,Z_test = X[-200:,:],Y[-200:,:],Z[-200:,:],X[-40:,:],Y[-40:,:],Z[-40:,:]
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
                break    
    # printlog(f"Model {modeli}.{m} stop, Epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
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
    return(votes)

def voting1(votes,prop_votes,prop_robots,hat_inv):
    # votes = np.load(f'rlt/{filei}')['votes']
    w = [1,2,3,4,5]
    votes = votes.reshape((np.prod(votes.shape[0:2]), 6, len(codes)))[:,:,range(int(prop_votes * len(codes)))]
    scores = []
    for votei in votes:
        scorei = []
        for i in range(5):
            scorei.append([np.mean(profit[i,votei[i]]),np.mean(life[i,votei[i]]),np.mean(back[i,votei[i]])])
        scorei = np.asarray(scorei)
        scorei = (scorei * np.asarray(w).reshape(5,1)).sum(axis=0)/sum(w)
        scores.append(scorei)
    scores = pd.DataFrame(np.asarray(scores))
    scores.columns = ['profit','life','back']
    score = np.ravel(scores['life'])# * np.ravel(scores['profit'])
    votes = votes[score>=np.quantile(score,1-prop_robots),5,:]
    rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes)), orient='index', columns=['count']).sort_values('count',ascending=False)
    rlt['codes'] = codes[rlt.index]
    rlt['date'] = arg1
    rlt['prop'] = rlt['count']/votes.shape[0]*hat_inv
    rlt['cumprop'] = np.cumsum(rlt['prop'])
    rlt = rlt[rlt['cumprop']<1]
    rlt['share'] = rlt['count']/sum(rlt['count'])
    return(rlt.iloc()[:,[0,1,2,5]])

def voting(votes,prop_votes,prop_robots,hat_inv):
    votes = votes.reshape((np.prod(votes.shape[0:2]), 6, len(codes)))[:,:,range(int(prop_votes * len(codes)))]
    scores = []
    for j in range(votes.shape[0]):
        votei = votes[j]
        today = closepvt[-5:,:]/openpvt[-5:,:]
        overnite = openpvt[-5:,:]/closepvt[-6:-1,:]
        overnite[0,:] = 1
        scorei = []
        for i in range(5):
            scorei.append([np.mean(today[i,votei[i]]-1),np.mean(overnite[i,votei[i]]-1)])
        scorei = np.asarray(scorei)
        scorei = np.concatenate((scorei,scorei.sum(axis=1,keepdims=True)),axis=1)
        scorei = np.append(np.append(scorei.mean(axis=0),scorei.min(axis=0)),np.prod(scorei[:,2]+1)-1)
        scores.append(scorei)
    scores = np.asarray(scores)
    scores = np.concatenate((scores,(scores[:,6,np.newaxis]+1)/(scores[:,5,np.newaxis]+1)),axis=1)
    scores = pd.DataFrame(scores)
    scores.columns = ['avgtoday','avgovernite','avgtotal','mintoday','minovernite','mintotal','accum','accumprisk']
    score = np.ravel(scores['accumprisk'])
    votes = votes[score>=np.quantile(score,1-prop_robots),5,:]
    rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes)), orient='index', columns=['count']).sort_values('count',ascending=False)
    rlt['codes'] = codes[rlt.index]
    rlt['date'] = arg1
    rlt['prop'] = rlt['count']/votes.shape[0]*hat_inv
    rlt['cumprop'] = np.cumsum(rlt['prop'])
    rlt = rlt[rlt['cumprop']<1]
    rlt['share'] = rlt['count']/sum(rlt['count'])
    return(rlt.iloc()[:,[0,1,2,5]])

def match(a, b):
    return [i for i in range(len(b)) if b[i] in a]

##########################################################################################
#Arguments
##########################################################################################

# arg1 = '20230511'
# prd1 = 40
note = 'test'

# arg1 = int(sys.argv[1])
# prd1 = int(sys.argv[2])
# note = datetime.datetime.now().strftime("%y%m%d%H%M")

if(note=='test'):
    logfile = 'Terminal'
    def printlog(x):
        print(datetime.datetime.now(), x)
else:
    logfile = f'log/dg1v3_{arg1}_{prd1}_{note}.log'
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s %(message)s', 
        filename=logfile,  
        filemode='a'  
    )
    printlog = logging.debug

if torch.cuda.is_available():
    printlog("GPU is available")
    device = torch.device("cuda")
else:
    printlog("GPU is not available, using CPU instead")
    device = torch.device("cpu")

##########################################################################################
#Modeling
##########################################################################################

#Parameter

printlog(f'Parameting, logging at {logfile}')

seeds = [101,777,303,602]
hidden_dim = 1024
latent_dim = 128
dropout_rate = 0.5
l2_reg = 0.01
num_epochs = 10000
lr = 0.01
early_tol = 1.1
patience = 10
patience2 = 5
momentum = 0.99
num_robots = 10000
prop_votes = 0.1
prop_robots = 0.05
hat_inv = 0.1
target = 'life'
w = [1,2,3,4,5]

printlog([seeds,hidden_dim,latent_dim,dropout_rate,l2_reg,num_epochs,lr,early_tol,patience,patience2,momentum,num_robots,prop_votes,prop_robots,hat_inv])

#Process

printlog(f'Process data/raw{arg1}.csv with period = {prd1}')

datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes,closepvt,openpvt = processdata(arg1,prd1,seeds=seeds)
X_dim = X.shape[1]
Y_dim = Y.shape[1]
Z_dim = Z.shape[1]

#Modeling

models = []
for i in range(len(datasets)):
    printlog(f'Model {i} training')
    modeli = train(i,X_dim,Y_dim,Z_dim,hidden_dim,latent_dim,dropout_rate,l2_reg,lr,early_tol,patience,patience2,momentum)
    models.append(modeli)

#Voting

votes = roboting(num_robots,models)
np.savez(f'rlt/dg1v3_{arg1}_{prd1}_{note}.npz',votes=votes)
printlog(voting(np.asarray(votes),prop_votes,prop_robots,hat_inv))

##########################################################################################
#Backdata
##########################################################################################

#Parameter

model = 'dg1'
prop_votes = 0.05
prop_robots = 0.1
hat_inv = 0.1
vote = 'voting1'

if vote == 'voting1':
    votemodel = voting1
else:
    votemodel = voting

#Transaction Generator

files = []
for arg1 in np.sort(os.listdir('rlt')).tolist():
    if f'{model}_' in arg1:
        files.append(arg1)

trans = []
for filei in files:
    arg1 = filei.split('_')[1]
    print(arg1)
    datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes,closepvt,openpvt = processdata(arg1,10,seeds=[1])
    votei = np.load(f'rlt/{filei}')['votes'][:,:,:,:]
    rlti = votemodel(votei,prop_votes,prop_robots,hat_inv)
    rlti['robot'] = model
    rlti['vote'] = vote
    rlti['hat'] = hat_inv
    trans.append(rlti)

trans = pd.concat(trans,axis=0)

#Reference of Codes

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

#Calculation

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

[model,prop_votes,prop_robots,hat_inv,vote]
rlt
