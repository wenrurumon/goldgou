
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import datetime
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import rankdata
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import logging

#######################################################################################################################################
# Parameter
#######################################################################################################################################

arg1 = 20230511
prd1 = 40
note = 'test'

# arg1 = sys.argv[1]
# prd1 = sys.argv[2]
# note = datetime.datetime.now().strftime("%y%m%d%H%M")

if(note=='test'):
  def printlog(x):
    print(datetime.datetime.now(), x)
else:
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s %(message)s', 
        filename=f'log/bmodel1_{arg1}_{prd1}_{note}.log',  
        filemode='a'  
    )
    printlog = logging.debug

if torch.cuda.is_available():
    printlog("GPU is available")
    device = torch.device("cuda")
else:
    printlog("GPU is not available, using CPU instead")
    device = torch.device("cpu")

printlog([arg1,note,hidden_dim,latent_dim,dropout_rates,lrs,l2_regs,num_epochss,early_tols,batch_sizes,patiences,patience2s,prelosss])

#######################################################################################################################################
# Module
#######################################################################################################################################

def processdata(arg1,prd1):
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
    codes = np.ravel(raws['close'].columns)
    closepvt = np.asarray(raws['close'])
    openpvt = np.asarray(raws['open'])
    volpvt = np.asarray(raws['vol'])
    lowpvt = np.asarray(raws['low'])
    highpvt = np.asarray(raws['high'])
    datemap = raw[['did', 'date']].drop_duplicates().sort_values('date')
    datemap = datemap[datemap['date']>='2021-01-01']
    dates = np.sort(np.ravel(datemap['date']))
    Z = []
    for i in range(closepvt.shape[0]-(prd1-1)):
        closei = closepvt[range(i, i + (prd1-1)), :] / closepvt[i + (prd1-1), None, :]
        closei = np.nan_to_num(closei,nan=-1)
        voli = volpvt[range(i, i + prd1), :]
        voli = voli / (np.nan_to_num(voli, nan=0).mean(axis=1)).reshape(prd1, 1)
        voli = np.nan_to_num(voli,nan=0)
        refi = np.concatenate((closei,voli),axis=0)
        Z.append(np.ravel(refi.T))
    Z = np.asarray(Z)
    Y = []
    Y2 = []
    for i in range(closepvt.shape[0]-prd1-5):
        profi = openpvt[range(i+prd1+2,i+prd1+6),:].mean(axis=0)/openpvt[i+prd1+1,None,:]
        backi = lowpvt[range(i+prd1+2,i+prd1+6),:].min(axis=0)/openpvt[i+prd1+1,None,:]
        Y.append(profi)
        Y2.append(backi)
    Y = np.concatenate(Y)
    Y = np.nan_to_num(Y,nan=0.9)
    Y2 = np.concatenate(Y2)
    Y2 = np.nan_to_num(Y2,nan=0.5)
    X = Z[range(Y.shape[0]),:]
    Z = Z[X.shape[0]:,:]
    life = []
    for i in range(10, closepvt.shape[0]):
        data = closepvt[(i-9):(i+1), :]
        col_mean = np.mean(data, axis=0)
        life.append(col_mean)
    life = np.array(life)
    P = closepvt/openpvt
    P = P[-5:,:]
    L = life[-5:,:]/life[-6:-1]
    B = lowpvt/openpvt
    B = B[-5:,:]
    dates = dates[-(X.shape[0]+Z.shape[0]):]
    return(X,Y,Y2,Z,P,L,B,codes,dates)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, dropout_rate, l2_reg):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + output_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self.predictor2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self.regularization = nn.ModuleList()
        self.regularization.append(nn.Linear(hidden_dim, hidden_dim))
        self.regularization.append(nn.Linear(latent_dim, latent_dim))
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.predictor(z)
        y2_pred = self.predictor2(z)
        concat = torch.cat((z, y_pred, y2_pred), dim=1)
        x2_pred = self.decoder(concat)
        reg_loss = 0.0
        for layer in self.regularization:
            reg_loss += torch.norm(layer.weight)
        reg_loss *= self.l2_reg
        return x2_pred, y_pred, y2_pred, reg_loss

def train():
    models = []
    for s in range(len(datasets)):
        seed = s
        X_train,X2_train,Y_train,Y2_train,X_test,X2_test,Y_test,Y2_test = datasets[s]
        best_loss = np.inf
        for k in range(3):
            # k = 0
            dropout_rate = dropout_rates[k]
            lr = lrs[k]
            l2_reg = l2_regs[k]
            num_epochs = num_epochss[k]
            batch_size = batch_sizes[k]
            patience = patiences[k]
            patience2 = patience2s[k]
            early_tol = early_tols[k]
            preloss = prelosss[k]
            preloss = (np.ravel(preloss)/np.sum(preloss)).tolist()
            counter = 0
            counter2 = 0
            #
            printlog(f"Model {k} start training with seed @ {seed}")
            #
            model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).to(device)
            if k>0:
                model.load_state_dict(best_model_state_dict)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            #
            train_dataset = TensorDataset(X_train, Y_train, X2_train, Y2_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            #
            for epoch in range(num_epochs):
                for xtr, ytr, x2tr, y2tr in train_loader:
                    xtr = xtr.float()
                    ytr = ytr.float()
                    x2tr = x2tr.float()
                    y2tr = y2tr.float()
                    x2hat, yhat, y2hat, l2_loss = model(xtr)
                    loss1 = criterion(x2hat, x2tr)
                    loss2 = criterion(yhat, ytr)
                    loss3 = criterion(y2hat, y2tr)
                    loss = preloss[0]*criterion(x2tr, x2hat) + preloss[1]*criterion(ytr, yhat) + preloss[2]*criterion(y2tr, y2hat) + l2_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    x2hat2, yhat2, y2hat2, _ = model(X_test)
                    vloss1 = criterion(x2hat2, X2_test)
                    vloss2 = criterion(yhat2, Y_test)
                    vloss3 = criterion(y2hat2, Y2_test)
                    vloss = preloss[0]*vloss1 + preloss[1]*vloss2 + preloss[2]*vloss3 + _
                if epoch>0:
                    if vloss < best_loss*early_tol:
                        if vloss < best_loss:
                            best_loss = vloss
                            best_model_state_dict = model.state_dict()
                            counter = 0
                        else:
                            counter += 0.5
                    else:
                        counter += 1
                    if counter >= patience:
                        printlog(f'Epoch:[{epoch+1}|{num_epochs}], Loss:[{loss:.4f}|{loss1:.4f}|{loss2:.4f}|{loss3:.4f}], Validate:[{vloss:.4f}|{vloss1:.4f}|{vloss2:.4f}|{vloss3:.4f}], Learning:[{counter:.2f}|{counter2:.2f}]')
                        counter = 0
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                        model.load_state_dict(best_model_state_dict)
                        counter2 += 1
                    if counter2 > patience2:
                        printlog(f"Early stopping at epoch {epoch+1}")
                        break               
                else:
                    if vloss < best_loss:
                            best_loss = vloss
                            best_model_state_dict = model.state_dict()
                    if (epoch+1)%100 == 0:
                        printlog(f'Epoch:[{epoch+1}|{num_epochs}], Loss:[{loss:.4f}|{loss1:.4f}|{loss2:.4f}|{loss3:.4f}], Validate:[{vloss:.4f}|{vloss1:.4f}|{vloss2:.4f}|{vloss3:.4f}]')
        model.load_state_dict(best_model_state_dict) 
        with torch.no_grad():
            x2hat2, yhat2, y2hat2, _ = model(X_test)
            vloss1 = criterion(x2hat2, X2_test)
            vloss2 = criterion(yhat2, Y_test)
            vloss3 = criterion(y2hat2, Y2_test)
            vloss = preloss[0]*vloss1 + preloss[1]*vloss2 + preloss[2]*vloss3 + _
        printlog(f'Loss:[{loss:.4f}|{loss1:.4f}|{loss2:.4f}|{loss2:.4f}], Validate:[{vloss:.4f}|{vloss1:.4f}|{vloss2:.4f}|{vloss3:.4f}]]')
        models.append(model)
    return(models)

def vote(x, votelen):
    return x>=-np.sort(-x)[votelen]

#######################################################################################################################################
# Modeling
#######################################################################################################################################

#Parameter
hidden_dim = 512
latent_dim = 32
dropout_rates = [0.5, 0.5, 0.5]
lrs = [0.001, 0.001, 0.001]
l2_regs = [0.01, 0.01, 0.01]
num_epochss = [5000,5000,5000]
early_tols = [1.1,1.1,1.05]
batch_sizes = [32,32,32]
patiences = [20,25,30]
patience2s = [15,20,30]
prelosss = [[7,2,1],[3,4,3],[1,7,2]]
sampleset = [777,603]
w = [1,2,3,4,5]

prop_votes = 0.03
num_votes = int(len(codes)*prop_votes)
num_robots = 10000
prop_robots = 0.2

#Data Process
X,Y,Y2,Z,P,L,B,codes,dates = processdata(arg1,prd1)
XZ = np.concatenate((X,Z))
XZscaler = StandardScaler()
XZscaler.fit(XZ)
X = XZscaler.transform(X)
Z = XZscaler.transform(Z)
Yscaler = StandardScaler()
Yscaler.fit(Y)
Y = Yscaler.transform(Y)
Y2scaler = StandardScaler()
Y2scaler.fit(Y2)
Y2 = Y2scaler.transform(Y2)
X = torch.tensor(X).float().to(device)
Y = torch.tensor(Y).float().to(device)
Y2 = torch.tensor(Y2).float().to(device)
Z = torch.tensor(Z).float().to(device)
X2 = torch.cat((X,Z),dim=0)[-X.shape[0]:,:]
input_dim = X.shape[1]
output_dim = Y.shape[1]

#Modelfile
datasets = []
for samplei in sampleset:
    np.random.seed(samplei)
    samples = np.random.permutation(np.ravel(range(X.shape[0])))
    samples = samples//91
    for s in range(5):
        datasets.append([X[samples!=s,:],X2[samples!=s,:],Y[samples!=s,:],Y2[samples!=s,:],X[samples==s,:],X2[samples==s,:],Y[samples==s,:],Y2[samples==s,:]])

#Training
models = train()

#Voting

Zi = Z
Ypreds = []

for model in models:
    for s in range(int(num_robots/len(models))):    
        torch.manual_seed(s)
        Ypredi = (model(Zi.to(device))[1].cpu().detach().numpy())
        Ypredi = Yscaler.inverse_transform(Ypredi)
        Ypreds.append(Ypredi)

Yvotes = []
Ppreds = []
Lpreds = []
Bpreds = []
for Ypredi in Ypreds:
    Ypredi = np.apply_along_axis(vote, 1, Ypredi[range(5),:], num_votes)
    Yvotes.append(Ypredi)    
    Ppreds.append(Ypredi*P)
    Lpreds.append(Ypredi*L)
    Bpreds.append(Ypredi*B)

robot_scores = []
for i in range(len(Ypreds)):
    robot_scores.append(np.ravel(np.concatenate((np.nansum(Ppreds[i],axis=1),np.nansum(Bpreds[i],axis=1),np.nansum(Lpreds[i],axis=1)),axis=0)))

Ypreds = np.asarray(Ypreds)
test = np.asarray(robot_scores)/num_votes
weight = np.asarray((w*3)).reshape(1,15)
weight = weight/sum(w)
test = (test*weight)
test1 = test[:,range(5)].sum(axis=1)
test2 = test[:,range(5,10)].sum(axis=1)
test3 = test[:,range(10,15)].sum(axis=1)
test = pd.DataFrame({'score1':test1,'score2':-test2,'score3':test3})

robots = rankdata(-test,axis=0)
robots = np.ravel((robots<int(prop_robots*num_robots)).mean(axis=1))
robots = np.ravel(robots>0.5).tolist()
robots = np.ravel(range(len(Ypreds)))[robots].tolist()
rlt = []
for i in range(6):
    logging.debug(i)
    Ypredi = Ypreds[robots,i,:]
    Yvotei = np.apply_along_axis(vote, 1, Ypredi, 20)
    out = pd.DataFrame(dict({'day':i-5,'code':codes,'vote':Yvotei.sum(axis=0),'mean':Ypredi.mean(axis=0),'std':Ypredi.std(axis=0)})).sort_values('vote',ascending=False)
    out['cumvote'] = np.cumsum(out['vote']/len(robots)*10)
    out = out[out['cumvote']<100]
    out['share'] = out['vote']/np.sum(out['vote'])
    printlog(out)
    if(i<5):
        printlog(np.sum(P[i,np.ravel(out.index)] * out['share']))
    rlt.append(out)

printlog([arg1,note,hidden_dim,latent_dim,dropout_rates,lrs,l2_regs,num_epochss,early_tols,batch_sizes,patiences,patience2s,prelosss])
pd.concat(rlt,axis=0).to_csv(f'rlt/{arg1}_{note}.csv')
os.system(f'rm data/{arg1}.npz')
