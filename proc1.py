
import pandas as pd
import numpy as np
import sys

##########################################################################################
#Load Raw Data
##########################################################################################

arg1 = sys.argv[1]
period = int(sys.argv[2])
# arg1 = '20230413'
# period = 40
print(f'load data: data/raw{arg1}.csv')

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

##########################################################################################
#Reference data Z
##########################################################################################

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
for i in range(closepvt.shape[0]-(period-1)):
    closei = closepvt[range(i, i + (period-1)), :] / closepvt[i + (period-1), None, :]
    closei = np.nan_to_num(closei,nan=-1)
    voli = volpvt[range(i, i + period), :]
    voli = voli / (np.nan_to_num(voli, nan=0).mean(axis=1)).reshape(period, 1)
    voli = np.nan_to_num(voli,nan=0)
    refi = np.concatenate((closei,voli),axis=0)
    Z.append(np.ravel(refi.T))

Z = np.asarray(Z)

##########################################################################################
#Load Raw Data
##########################################################################################

Y = []
Y2 = []
for i in range(closepvt.shape[0]-period-5):
    profi = openpvt[range(i+period+2,i+period+6),:].mean(axis=0)/openpvt[i+period+1,None,:]
    backi = lowpvt[range(i+period+2,i+period+6),:].min(axis=0)/openpvt[i+period+1,None,:]
    Y.append(profi)
    Y2.append(backi)

Y = np.concatenate(Y)
Y = np.nan_to_num(Y,nan=0.9)
Y2 = np.concatenate(Y2)
Y2 = np.nan_to_num(Y2,nan=0.5)

X = Z[range(Y.shape[0]),:]
Z = Z[X.shape[0]:,:]

##########################################################################################
#Life Curve Evaluation
##########################################################################################

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

##########################################################################################
#Output
##########################################################################################

dates = dates[-(X.shape[0]+Z.shape[0]):]
np.savez(f'data/model1_{arg1}_{period}.npz',X=X,Y=Y,Y2=Y2,Z=Z,P=P,L=L,B=B,codes=codes,dates=dates)
print(f'save data: data/model1_{arg1}_{period}.npz')

