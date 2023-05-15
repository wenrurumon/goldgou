
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import os 
from collections import Counter
device = torch.device("cpu")

printlog = print

def processdata(arg1,prd1,seeds):
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
    return(closepvt,openpvt,codes)

############################################################################################################

files = np.sort(os.listdir('rlt'))
rlts = []
for votefile in files:
    prop_votes = 0.05
    closepvt,openpvt,codes = processdata(votefile.split('_')[1],prd1=40,seeds=[1])
    votes = np.load(f'rlt/{votefile}',allow_pickle=True)['votes']
    # votes = votes[:,range(50),:,:]
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
    votes = votes[score>=np.quantile(score,0.95),5,:]
    rlt = pd.DataFrame.from_dict(Counter(np.ravel(votes)), orient='index', columns=['count']).sort_values('count',ascending=False)
    rlt['codes'] = codes[rlt.index]
    rlt['date'] = votefile.split('_')[1]
    rlt['prop'] = rlt['count']/votes.shape[0]*0.1
    rlt['cumprop'] = np.cumsum(rlt['prop'])
    rlt = rlt[rlt['cumprop']<1]
    rlt['share'] = rlt['count']/sum(rlt['count'])
    rlt.iloc()[:,[0,1,2,5]]
    rlts.append(rlt)

rlts = pd.concat(rlts,axis=0)
rlts.to_csv('rlt/dg1_test.csv')
