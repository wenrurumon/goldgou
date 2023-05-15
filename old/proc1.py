import pandas as pd
import numpy as np
import sys

##########################################################################################
#Load Raw Data
##########################################################################################

arg1 = sys.argv[1]
# arg1 = '0404'
print(f'load data: data/raw{arg1}.csv')

raw = pd.read_csv(f'data/raw{arg1}.csv')
raw =  raw.drop_duplicates()
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values(['code','date'])
raw['did'] = raw['date'].rank(method='dense').astype(int) - 1

def fill_missing_values(row):
    return row.fillna(method='ffill')

raws = []
for i in ['open','close','vol']:
    rawi = pd.pivot_table(raw, values=i, index=['did'], columns=['code'])
    rawi = rawi.apply(fill_missing_values,axis=0)
    raws.append(rawi[rawi.index>=np.min(raw['did'][raw['date']>'2021-01-01'])])

raws = dict(zip(['open','close','vol'],raws))

##########################################################################################
#Reference data Z
##########################################################################################

codes = np.ravel(raws['close'].columns)
closepvt = np.asarray(raws['close'])
openpvt = np.asarray(raws['open'])
volpvt = np.asarray(raws['vol'])

Z = []
for i in range(closepvt.shape[0]-39):
    closei = closepvt[range(i, i + 39), :] / closepvt[i + 39, None, :]
    closei = np.nan_to_num(closei,nan=-1)
    voli = volpvt[range(i, i + 40), :]
    voli = voli / (np.nan_to_num(voli, nan=0).mean(axis=1)).reshape(40, 1)
    voli = np.nan_to_num(voli,nan=0)
    refi = np.concatenate((closei,voli),axis=0)
    Z.append(np.append(i,np.ravel(refi.T)))

Z = np.asarray(Z)

##########################################################################################
#Load Raw Data
##########################################################################################

Y = []
for i in range(closepvt.shape[0]-45):
    profi = np.append(i,openpvt[range(i+42,i+46),:].mean(axis=0)/openpvt[i+41,None,:])
    Y.append(profi)

Y = np.asarray(Y)
Y = np.nan_to_num(Y,nan=0.9)

datemap = raw[['did', 'date']].drop_duplicates().sort_values('date')
datemap = datemap[datemap['date']>='2021-01-01']
dates = np.ravel(datemap['date'])[39:]
X = Z[np.searchsorted(Z[:,0], Y[:,0]),:]
Z = Z[X.shape[0]:,:]

#Output

np.savez(f'data/model1_{arg1}.npz',X=X,Y=Y,Z=Z,codes=codes,dates=dates)
print(f'save data: data/model1_{arg1}.npz')
