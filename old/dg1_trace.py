
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
#Parametering
##########################################################################################

arg1 = 'trace0512'
prd1 = 40
# note = 'trace0512'

# arg1 = int(sys.argv[1])
# prd1 = int(sys.argv[2])
note = datetime.datetime.now().strftime("%y%m%d%H%M")

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
num_robots = 1000
prop_votes = 0.05
prop_robots = 0.1
prop_codes = 0.95
w0 = [1,2,3,4,5]
w = (w0/np.sum(w0)).reshape(5,1)
target = 'life'

if(note=='test'):
  def printlog(x):
    print(datetime.datetime.now(), x)
else:
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s %(message)s', 
        filename=f'log/dg1_{arg1}_{prd1}_{note}.log',  
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
#Modules
##########################################################################################

def processdata(arg1,prd1,seeds=seeds):
    printlog(f'load data: data/raw{arg1}.csv')
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
        xi = volpvt[range(i, i + (prd1-1)), :] / volpvt[i + (prd1-1), None, :]
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
    return(datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes)

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
        if epoch==0:
            printlog(f'Model {modeli}.{m} start, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
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
                printlog(f'Model {modeli}.{m} training, Epoch:[{epoch+1}|{num_epochs}|{patience2-counter2}], Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]')
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
    printlog(f"Model {modeli}.{m} stop, Epoch {epoch+1}, Loss:[{loss:.4f}|{lossy:.4f}|{lossz:.4f}], Validate:[{vloss:.4f}|{vlossy:.4f}|{vlossz:.4f}]")
    return(model)

def voting(num_robots,prop_votes,prop_robots,prop_codes,w,models):
    num_votes = int(len(codes)*prop_votes)
    votes = []
    for modeli in models:
        for s in range(int(num_robots/len(models))):
            torch.manual_seed(s)
            Y2, Z2, _ = modeli(X2)
            Z2 = Zscaler.inverse_transform(Z2.cpu().detach().numpy())
            votes.append(((-Z2).argsort(axis=1)))
    allvotes = np.asarray(votes)
    votes = allvotes[:,:,range(num_votes)]
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
    scores = scores.head(int(num_robots*prop_robots))
    votes2 = []
    for i in range(scores.shape[0]):
        votes2.append(codes[votes[scores['life'][i]][5]])
    votes2 = np.ravel(votes2)
    rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes2)), orient='index', columns=['count']).sort_values('count',ascending=False)
    rlt['codes'] = rlt.index
    rlt['date'] = arg1
    rlt['idx'] = rlt['count']/(num_robots)/(num_votes/len(codes))
    rlt = rlt[rlt['idx']>=np.quantile(rlt['idx'],prop_codes)]
    rlt['share'] = rlt['count']/sum(rlt['count'])
    return(rlt,allvotes)

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

def voting(votes,num_robots,prop_votes,prop_robots,prop_codes,target,w):
    printlog([len(votes),num_robots,prop_votes,prop_robots,prop_codes,target,w])
    num_votes = int(len(codes)*prop_votes)
    votes2 = []
    for votei in votes:
        votes2.append(votei[range(int(num_robots/np.concatenate(votes,axis=0).shape[0]*votei.shape[0])),:])
    votes = np.concatenate(votes2,axis=0)[:,:,range(num_votes)]
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
    scores = scores.head(int(num_robots*prop_robots))
    votes2 = []
    for i in range(scores.shape[0]):
        votes2.append(codes[votes[scores[target][i]][5]])
    votes2 = np.ravel(votes2)
    rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes2)), orient='index', columns=['count']).sort_values('count',ascending=False)
    rlt['codes'] = rlt.index
    rlt['date'] = arg1
    rlt['idx'] = rlt['count']/(num_robots)/(num_votes/len(codes))
    rlt = rlt[rlt['idx']>=np.quantile(rlt['idx'],prop_codes)]
    rlt['share'] = rlt['count']/sum(rlt['count'])
    return(rlt)

##########################################################################################
#Modeling
##########################################################################################

# #Process
# datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes = processdata(arg1,prd1,seeds=seeds)
# X_dim = X.shape[1]
# Y_dim = Y.shape[1]
# Z_dim = Z.shape[1]

# #Modeling
# models = []
# for i in range(len(datasets)):
#     modeli = train(i,X_dim,Y_dim,Z_dim,hidden_dim,latent_dim,dropout_rate,l2_reg,lr,early_tol,patience,patience2,momentum)
#     models.append(modeli)

# #Voting

# votes = roboting(10000,models)
# np.savez(f'rlt/dg1_{arg1}_{prd1}_{note}.npz',votes=votes)
# rlt = voting(votes,num_robots,prop_votes,prop_robots,prop_codes,target,w)
# printlog(rlt)
# printlog(voting(votes,num_robots=10000,prop_votes=0.05,prop_robots=0.1,prop_codes=0.9,target='life',w=w))

##########################################################################################
#Backdat
##########################################################################################

files = []
for arg1 in np.sort(os.listdir('data')).tolist():
    if 'raw' in arg1:
        files.append(int(arg1.replace('.csv','').replace('raw','')))

rlts = []
for arg1 in files:
    torch.cuda.empty_cache()
    #Data Processing
    datasets,life,profit,back,X,Y,Z,X2,Zscaler,codes = processdata(arg1,prd1,seeds=seeds)
    X_dim = X.shape[1]
    Y_dim = Y.shape[1]
    Z_dim = Z.shape[1]
    #Modeling
    models = []
    for i in range(len(datasets)):
        modeli = train(i,X_dim,Y_dim,Z_dim,hidden_dim,latent_dim,dropout_rate,l2_reg,lr,early_tol,patience,patience2,momentum)
        models.append(modeli)
    #Voting
    votes = roboting(10000,models)
    np.savez(f'rlt/dg1_{arg1}_{prd1}_{note}.npz',votes=votes)
    rlt = voting(votes,num_robots,prop_votes,prop_robots,prop_codes,target,w)
    rlts.append(rlt)
