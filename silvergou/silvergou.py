
import pandas as pd
import datetime
import numpy as np
import sklearn.linear_model
from sklearn.metrics import r2_score
import akshare as ak
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

#Get Dates

tradedates = [d.strftime('%Y%m%d') for d in np.ravel(ak.tool_trade_date_hist_sina())]
def getdates(date0,f=420):
    date1 = (np.asarray(tradedates) != str(date0)).argsort()[0]
    out = []
    out.append(np.asarray(tradedates)[date1-f])
    for i in np.asarray(tradedates)[range(date1, date1 + 3)].tolist():
        out.append(i)
    return(out)

#Get Codes

codelist = pd.concat((ak.stock_sh_a_spot_em(),ak.stock_sz_a_spot_em()),axis=0)
codelist = codelist[~codelist["名称"].str.contains(r'退|ST')]
codes = np.ravel(codelist.iloc()[:,1])

date0 = datetime.datetime.today().strftime('%Y%m%d')
# date0 = 20230814
date00, date0, date1, date2 = getdates(date0)
print(f'know @ {date00}~{date0}, buy @ {date1}, valid @ {date2}')

#Get Transaction
# raw = []
# for codei in codes[1:]:
#     print(codei)
#     rawi = ak.stock_zh_a_hist(symbol=codei, period="daily", start_date=date00, end_date=date0,adjust="qfq")
#     rawi['code'] = codei
#     raw.append(rawi)
#
# raw = pd.concat(raw, axis=0)
# raw.to_csv(f'data/data_{date1}.csv')
raw = pd.read_csv(f'data/data_{date1}.csv',index_col=0)
raw.columns = ['date', 'open', 'close', 'high', 'low', 'vol', 'val', 'var', 'chp', 'chv', 'turn', 'code']
codes = raw.groupby('code')['date'].min().reset_index()
raw = raw[raw['code'].isin(codes[codes.date<='2023-01-01'].code)]
raw = raw[raw['date']>'2023-01-01']
dates = np.sort(np.unique(raw['date']))
raw = raw.drop_duplicates()
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values(['code', 'date'])
raw['did'] = raw['date'].rank(method='dense').astype(int) - 1
dates = np.sort(np.unique(raw['date']))

#Process Modelfile
prd1 = 2
prd2 = 20

datasets = []
for datei in range(prd1+prd2,len(dates)-1):
    date00 = dates[range(max(0,datei-prd1-prd2),datei)]
    date1 = np.asarray(dates[datei]).reshape(1,)
    date2 = np.asarray(dates[datei+1]).reshape(1,)
    rawi = raw[raw["date"].isin(np.concatenate([date00,date1,date2]))]
    rawi = rawi[['did','date','code','close','open','high','low','turn']]
    rawi = rawi[rawi['code'].isin(rawi[rawi['date']==np.max(date00)]['code'])]
    raws = []
    for i in rawi.columns[3:]:
        rawii = pd.pivot_table(rawi, values=i, index=['did'], columns=['code'])
        if i=='close':
            def fill_missing_values(row):
                return row.fillna(method='ffill')
            rawii = rawii.apply(fill_missing_values, axis=0)
            rawii = rawii.fillna(method='bfill')
        elif i=='turn':
            rawii = rawii.fillna(0)
        raws.append(rawii)
    for i in range(1,len(raws)):
        raws[i] = raws[i].fillna(raws[0])
    raws = dict(zip(rawi.columns[3:], raws))
    closei = np.asarray(raws['close'])
    openi = np.asarray(raws['open'])[-(prd1+2):,:]
    highi = np.asarray(raws['high'])[-(prd1+2):,:]
    lowi = np.asarray(raws['low'])[-(prd1+2):,:]
    turni = np.asarray(raws['turn'])[-(prd1+2):,:]
    close20i = []
    for i in range(prd2,closei.shape[0]):
        close20i.append(closei[range(i-prd2+1,i+1),:].mean(axis=0))
    close20i = np.asarray(close20i)
    closei = closei[-(prd1+2):,:]
    dtodayi = (closei/openi)[range(prd1),:]
    dtonitei = (openi[1:,:]/closei[range(closei.shape[0]-1),:])[range(prd1-1),:]
    upi = np.where(openi>closei,openi,closei)
    downi = np.where(openi<closei,openi,closei)
    dhighi = ((highi/upi)/(upi/downi))[range(prd1),:]
    dlowi = ((lowi/downi)/(upi/downi))[range(prd1),:]
    dturni = (turni[1:,:]/(turni[range(turni.shape[0]-1),:]+1e-3))[range(prd1-1),:]
    roii = closei[prd1+1,None,:]/openi[prd1,None,:]
    riski = lowi[prd1:].min(axis=0).reshape(1,-1)/openi[prd1,None,:]
    dclose20i = (closei/close20i)[range(prd1),:]
    xi = np.vstack((dtodayi,dtonitei,dhighi,dlowi,dturni,dclose20i)).T
    yi = np.vstack((roii,riski)).T
    codesi = raws['close'].columns.tolist()
    mapi = pd.DataFrame({'code':codesi})
    _, date0, date1, date2 = getdates(np.datetime_as_string(np.max(date00), unit='D').replace('-', ''))
    mapi['know'] = date0
    mapi['buy'] = date1
    mapi['sell'] = date2
    mapi = pd.concat((mapi.reset_index().iloc()[:,1:],pd.DataFrame(yi,columns=['roi','risk']).reset_index().iloc()[:,1:]),axis=1)
    datasets.append((xi,mapi))

for datei in range(len(dates)-1,len(dates)+1):
    date00 = dates[range(max(0,datei-prd1-prd2),datei)]
    _,date0,date1,date2 = getdates(np.datetime_as_string(np.max(date00), unit='D').replace('-',''))
    rawi = raw[raw["date"].isin(np.concatenate([date00]))]
    rawi = rawi[['did','date','code','close','open','high','low','turn']]
    rawi = rawi[rawi['code'].isin(rawi[rawi['date']==np.max(date00)]['code'])]
    raws = []
    for i in rawi.columns[3:]:
        rawii = pd.pivot_table(rawi, values=i, index=['did'], columns=['code'])
        if i=='close':
            def fill_missing_values(row):
                return row.fillna(method='ffill')
            rawii = rawii.apply(fill_missing_values, axis=0)
            rawii = rawii.fillna(method='bfill')
        elif i=='turn':
            rawii = rawii.fillna(0)
        raws.append(rawii)
    for i in range(1,len(raws)):
        raws[i] = raws[i].fillna(raws[0])
    raws = dict(zip(rawi.columns[3:], raws))
    closei = np.asarray(raws['close'])
    openi = np.asarray(raws['open'])[-(prd1):,:]
    highi = np.asarray(raws['high'])[-(prd1):,:]
    lowi = np.asarray(raws['low'])[-(prd1):,:]
    turni = np.asarray(raws['turn'])[-(prd1):,:]
    close20i = []
    for i in range(prd2,closei.shape[0]):
        close20i.append(closei[range(i-prd2+1,i+1),:].mean(axis=0))
    close20i = np.asarray(close20i)
    closei = closei[-(prd1):,:]
    dtodayi = (closei/openi)[range(prd1),:]
    dtonitei = (openi[1:,:]/closei[range(closei.shape[0]-1),:])[range(prd1-1),:]
    upi = np.where(openi>closei,openi,closei)
    downi = np.where(openi<closei,openi,closei)
    dhighi = ((highi/upi)/(upi/downi))[range(prd1),:]
    dlowi = ((lowi/downi)/(upi/downi))[range(prd1),:]
    dturni = (turni[1:,:]/(turni[range(turni.shape[0]-1),:]+1e-3))[range(prd1-1),:]
    dclose20i = (closei/close20i)[range(prd1),:]
    xi = np.vstack((dtodayi,dtonitei,dhighi,dlowi,dturni,dclose20i)).T
    yi = [np.nan]
    codesi = raws['close'].columns.tolist()
    mapi = pd.DataFrame({'code':codesi})
    mapi['know'] = date0
    mapi['buy'] = date1
    mapi['sell'] = date2
    mapi['roi'] = np.nan
    mapi['risk'] = np.nan
    datasets.append((xi,mapi))

Xs = []
Maps = []
for i in datasets:
    Xs.append(i[0])
    Maps.append(i[1])

Xs = np.concatenate(Xs,axis=0)
Maps = pd.concat(Maps,axis=0)

#Model

rlt = []
for date0 in np.sort(np.unique(Maps['know']))[range(100,len(np.sort(np.unique( Maps['know'])))-2)]:
# date0 = 20230804
    date00,date0,date1,date2 = getdates(date0,0)
    print(date00,date0,date1,date2)
    mapi = Maps[(Maps['know']>=date00)&(Maps['know']<=date0)]
    xi = Xs[(Maps['know']>=date00)&(Maps['know']<=date0),:]
    xi = np.hstack((np.where(xi>1,xi,0),np.where(xi<1,xi,0))) + 1e-8
    yi = np.asarray(mapi[['roi','risk']])
    mapi2 = Maps[Maps['know']==date2]
    xi2 = Xs[Maps['know']==date2,:]
    xi2 = np.hstack((np.where(xi2>1,xi2,0),np.where(xi2<1,xi2,0))) + 1e-8
    yi2 = np.asarray(mapi2[['roi','risk']])
    p = []
    for seed in range(100):
        x_train, x_test, y_train, y_test = train_test_split(xi, yi, test_size=0.5, random_state=seed)
        model = sklearn.linear_model.LinearRegression()
        model.fit(np.log(x_train),(np.log(y_train)))
        p.append(np.exp(model.predict(np.log(xi2))))
    p = np.asarray(p)
    pmean = p.mean(axis=0)[:,0]
    pmin = p.min(axis=0)[:,1]
    p = pd.concat((mapi2.reset_index().iloc()[:,1:],pd.DataFrame({'mean': pmean,'min': pmin})),axis=1).sort_values(['mean'],ascending=False)
    rlt.append(p)

#Evaluation

rlt2 = []
for p in rlt:
    sel = 5
    p['score'] = p['mean'] * p['min']
    p = p.sort_values(['score'],ascending=False)
    rlt2.append([np.unique(p['buy'])[0],np.mean(p['roi'][range(sel)]), np.mean(p['roi']),np.mean(p['roi'][range(sel)])/np.mean(p['roi'])])

rlt2 = pd.DataFrame(rlt2,columns=['buydate','roi','ref','index'])
rlt2 = rlt2.iloc()[range(rlt2.shape[0]-2),:]
rlt2.iloc()[:,1:].prod(axis=0)
np.mean(rlt2['index']>1)

#Result

rlt2 = pd.concat(rlt,axis=0).sort_values(['know','score'],ascending=False)
rlt2[(rlt2['know']==np.max(rlt2['know']))].iloc()[range(sel),:]
# pd.concat(rlt,axis=0).to_csv('wangbaba.csv')

######################################################################

def create_xxi(xi):
    n, p = xi.shape
    xxi = np.zeros((n, p*p))
    for i in range(p):
        for j in range(i,p):
            col_index = i * p + j
            xxi[:, col_index] = xi[:, i] * xi[:, j]
    return np.hstack((xi,xxi))

date0 = '20230808'
date00, date0, date1, date2 = getdates(date0, 0)
print(date00, date0, date1, date2)
mapi = Maps[(Maps['know'] >= date00) & (Maps['know'] <= date0)]
xi = Xs[(Maps['know'] >= date00) & (Maps['know'] <= date0), :]
xi = np.hstack((np.where(xi >= 1, xi, 0), np.where(xi <= 1, xi, 0)))
xxi = create_xxi(xi)
yi = np.ravel(mapi['roi'])
mapi2 = Maps[Maps['know'] == date2]
xi2 = Xs[Maps['know'] == date2, :]
xi2 = np.hstack((np.where(xi2 >= 1, xi2, 0), np.where(xi2 <= 1, xi2, 0)))
xxi2 = create_xxi(xi2)
yi2 = np.ravel(mapi2['roi'])

model = sklearn.linear_model.LinearRegression()
model.fit(np.log(xxi+ 1e-8),np.log(yi))
spearmanr(np.log(yi),model.predict(np.log(xxi+ 1e-8)))
spearmanr(np.log(yi2),model.predict(np.log(xxi2+ 1e-8)))



mapi2 = Maps[Maps['know'] == date2]
xi2 = Xs[Maps['know'] == date2, :]
xi2 = np.hstack((np.where(xi2 > 1, xi2, 0), np.where(xi2 < 1, xi2, 0))) + 1e-8
yi2 = np.asarray(mapi2[['roi', 'risk']])
p = []
for seed in range(100):
    x_train, x_test, y_train, y_test = train_test_split(xi, yi, test_size=0.5, random_state=seed)
    model = sklearn.linear_model.LinearRegression()
    model.fit(np.log(x_train), (np.log(y_train)))
    p.append(np.exp(model.predict(np.log(xi2))))
p = np.asarray(p)
pmean = p.mean(axis=0)[:, 0]
pmin = p.min(axis=0)[:, 1]
p = pd.concat((mapi2.reset_index().iloc()[:, 1:], pd.DataFrame({'mean': pmean, 'min': pmin})), axis=1).sort_values(
    ['mean'], ascending=False)
rlt.append(p)
