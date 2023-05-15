
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

# HyperParameter

arg1 = sys.argv[1]
prd1 = sys.argv[2]
note = datetime.datetime.now().strftime("%y%m%d%H%M")
os.system(f'python proc1.py {arg1} {prd1}')
arg1 = f'model1_{arg1}_{prd1}'

# arg1 = 'model1_20230420_40'
# note = 'test'

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s %(message)s', 
    filename=f'log/bmodel1_{arg1}_{note}.log',  
    filemode='a'  
)
printlog = logging.debug
# printlog = print

if torch.cuda.is_available():
    printlog("GPU is available")
    device = torch.device("cuda")
else:
    printlog("GPU is not available, using CPU instead")
    device = torch.device("cpu")

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
# patience2s = [1,2,3]
prelosss = [[7,2,1],[3,4,3],[1,7,2]]
sampleset = [777,603]

printlog([arg1,note,hidden_dim,latent_dim,dropout_rates,lrs,l2_regs,num_epochss,early_tols,batch_sizes,patiences,patience2s,prelosss])

# Data Load and Processing

raw = np.load(f'data/{arg1}.npz',allow_pickle=True)
printlog(f'load data: data/{arg1}.npz')
X = raw['X']
Y = raw['Y']
Y2 = raw['Y2']
Z = raw['Z']
P = raw['P']
L = raw['L']
B = raw['B']
codes = raw['codes']
dates = raw['dates']
# Y = np.where(Y < 1, 1 + (Y - 1) * 2, Y)

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

# Tensoring

X = torch.tensor(X).float().to(device)
Y = torch.tensor(Y).float().to(device)
Y2 = torch.tensor(Y2).float().to(device)
Z = torch.tensor(Z).float().to(device)
X2 = torch.cat((X,Z),dim=0)[-X.shape[0]:,:]

input_dim = X.shape[1]
output_dim = Y.shape[1]

# Model Setup

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

# Resulting

models = []
datasets = []
for samplei in sampleset:
    np.random.seed(samplei)
    samples = np.random.permutation(np.ravel(range(X.shape[0])))
    samples = samples//91
    for s in range(5):
        datasets.append([X[samples!=s,:],X2[samples!=s,:],Y[samples!=s,:],Y2[samples!=s,:],X[samples==s,:],X2[samples==s,:],Y[samples==s,:],Y2[samples==s,:]])

# Modeling

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
    models.append(best_model_state_dict)
    torch.save(best_model_state_dict, f'model/{arg1}_{note}_{s}.pt')

printlog(f"{arg1}_{note}_{s} training finished @ {datetime.datetime.now()}")

# Voting

# models = []
# for i in range(10):
#     models.append(torch.load(f'model/{arg1}_{note}_{i}.pt'))

def vote(x, votelen):
    return x>-np.sort(-x)[votelen]

k = 2
Zi = Z
Ypreds = []
Yvotes = []
Ppreds = []
Lpreds = []
Bpreds = []

for coef in models:
    model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).to(device)
    model.load_state_dict(coef)
    for s in range(int(2000/len(models)*5)):    
        torch.manual_seed(s)
        Ypredi = (model(Zi.to(device))[1].cpu().detach().numpy())
        Ypredi = Yscaler.inverse_transform(Ypredi)
        Ypreds.append(Ypredi)
        Ypredi = np.apply_along_axis(vote, 1, Ypredi[range(5),:], 20)
        Yvotes.append(Ypredi)    
        Ppreds.append(Ypredi*P)
        Lpreds.append(Ypredi*L)
        Bpreds.append(Ypredi*B)

robot_scores = []
for i in range(len(Ypreds)):
    robot_scores.append(np.ravel(np.concatenate((Ppreds[i].sum(axis=1),Bpreds[i].sum(axis=1),Lpreds[i].sum(axis=1)),axis=0)))

Ypreds = np.asarray(Ypreds)

test = np.asarray(robot_scores)/20
weight = np.asarray(([1,2,3,4,5]*3)).reshape(1,15)
weight = weight/sum([1,2,3,4,5])
test = (test*weight)
test1 = test[:,range(5)].sum(axis=1)
test2 = test[:,range(5,10)].sum(axis=1)
test3 = test[:,range(10,15)].sum(axis=1)
test = pd.DataFrame({'score1':test1,'score2':-test2,'score3':test3})

robots = rankdata(-test,axis=0)
robots = np.ravel((robots<2000).mean(axis=1))
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

# Voting 2

# k = 2
# Zi = Z
# Ypreds = []
# Yvotes = []
# Ppreds = []
# Lpreds = []
# Bpreds = []

# for coef in models:
#     model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rates[k], l2_regs[k]).to(device)
#     model.load_state_dict(coef)
#     for s in range(int(1000)):    
#         torch.manual_seed(s)
#         Ypredi = (model(Zi.to(device))[1].cpu().detach().numpy())
#         Ypredi = Yscaler.inverse_transform(Ypredi)
#         Ypreds.append(Ypredi)
#         Ypredi = np.apply_along_axis(vote, 1, Ypredi[range(5),:], 20)
#         Yvotes.append(Ypredi)    
#         Ppreds.append(Ypredi*P)
#         Lpreds.append(Ypredi*L)
#         Bpreds.append(Ypredi*B)

# robot_scores = []
# for i in range(len(Ypreds)):
#     robot_scores.append(np.ravel(np.concatenate((Ppreds[i].sum(axis=1),Bpreds[i].sum(axis=1),Lpreds[i].sum(axis=1)),axis=0)))

# Ypreds = np.asarray(Ypreds)
# Bpreds = np.asarray(Bpreds)

# test = np.asarray(robot_scores)/20
# weight = np.asarray(([1,2,3,4,5]*3)).reshape(1,15)
# weight = weight/sum([1,2,3,4,5])
# test = (test*weight)
# test1 = test[:,range(5)].sum(axis=1)
# test2 = test[:,range(5,10)].sum(axis=1)
# test3 = test[:,range(10,15)].sum(axis=1)
# test = pd.DataFrame({'score1':test1,'score2':-test2,'score3':test3})

# #half votes

# rlt = []
# for i in range(10):
#     np.random.seed(i*111)
#     samples = np.random.permutation(np.ravel(range(test.shape[0])))
#     samples = samples[range(int(len(samples)/10))]
#     testi = test.iloc()[samples,:]
#     Ypredsi = Ypreds[samples,:,:]
#     Bpredsi = Bpreds[samples,:,:]
#     robots = rankdata(-testi,axis=0)
#     robots = np.ravel((robots<int(test.shape[0]*0.2)).mean(axis=1))
#     robots = np.ravel(robots>0.5).tolist()
#     robots = np.ravel(range(len(Ypredsi)))[robots].tolist()
#     Ypredi = Ypredsi[robots,5,:]
#     Yvotei = np.apply_along_axis(vote, 1, Ypredi, 20)
#     out = pd.DataFrame(dict({'day':i,'code':codes,'vote':Yvotei.sum(axis=0),'mean':Ypredi.mean(axis=0),'std':Ypredi.std(axis=0)})).sort_values('vote',ascending=False)
#     out['cumvote'] = np.cumsum(out['vote']/len(robots)*10)
#     out = out[out['cumvote']<100]
#     out['share'] = out['vote']/np.sum(out['vote'])
#     rlt.append(out)

# rlt = pd.concat([df for df in rlt])
# rlt = rlt.groupby('code').agg({'vote':'sum','mean':'mean','std':'mean'})
# rlt['risk'] = rlt['mean']-rlt['std']
# rlt['share'] = rlt['vote']/np.sum(rlt['vote'])
# rlt = rlt.sort_values('share',ascending=False)
# rlt

