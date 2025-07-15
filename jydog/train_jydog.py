
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import random
from joblib import Parallel, delayed
import time
import akshare as ak
import datetime

################################################################################
# Process Sina Data
################################################################################

start_time = time.time()
print(f"启动数据处理")

# 读取数据
max_file = max([f for f in os.listdir('data') if 'sinadata' in f])
rawsina = pd.read_csv(os.path.join('data', max_file))

# 数据筛选和处理
rawsina = rawsina[~rawsina['code'].str.startswith('bj')]
rawsina = rawsina[['code', 'date', 'open', 'close', 'high', 'low', 'amount']]
rawsina.rename(columns={'amount': 'val'}, inplace=True)
rawsina['code'] = rawsina['code'].str[2:8].astype(int)
rawsina['date'] = rawsina['date'].str.replace('-', '').astype(int)
rawsina = rawsina[rawsina['date'] >= 20210000]

# 处理交易日期
datemap = ak.tool_trade_date_hist_sina()
datemap['trade_date'] = datemap['trade_date'].astype(str).str.replace('-', '')
datemap = datemap['trade_date'].astype(int).reset_index(drop=True)
datemap = pd.DataFrame({'date': datemap, 'did': np.arange(1, len(datemap) + 1)})

obsmap = pd.DataFrame({
    'obs': datemap['date'],
    'buy': np.append(datemap['date'][1:], np.nan),
    'sell': np.append(datemap['date'][2:], [np.nan, np.nan])
})

# 票清单
# stock_list = ak.stock_zh_a_spot()
# stock_list = stock_list[['代码', '名称', '最新价', '昨收', '今开', '最低', '最高', '成交额', '成交额', '时间戳']]
# stock_list.rename(columns={
#     '代码': 'code',
#     '名称': 'name',
#     '最新价': 'new',
#     '昨收': 'close1',
#     '今开': 'open0',
#     '最低': 'low0',
#     '最高': 'high0',
#     '成交额': 'value',
#     '时间戳': 'time'
# }, inplace=True)
# stock_list['code'] = stock_list['code'].str[2:8].astype(int)
# stock_list['date'] = pd.to_datetime('today').date()

################################################################################
# Process Jiyao
################################################################################

# 导入数据

rawjg_his = pd.read_csv('data/jglist_his.csv', index_col=0)
rawjg_new = pd.read_csv('data/jglist_new.csv', index_col=0)
rawjg = pd.concat([rawjg_his, rawjg_new], ignore_index=True)
rawjg['date'] = rawjg['date'].str.replace('-', '').astype(int)
rawjg['time'] = rawjg['time'].str.replace(':', '').astype(int)
rawjg = rawjg[~rawjg['code'].isin([30024, 688981, 858, 300152])]
rawjg = rawjg[~rawjg['name'].str.contains('证券|ST|国泰海通|申万宏源|国联民生')]
rawjg = rawjg.drop_duplicates()

# 处理机构数据
jglist = rawjg.copy()

# 每个 jgid 的投票次数，并计算权重 1/n
vote_counts = jglist.groupby('jgid').size().reset_index(name='vote')
jglist = jglist.merge(vote_counts, on='jgid')
jglist['vote'] = 1 / jglist['vote']

# 早于 9:15 的交易归于前一日
jglist['date2'] = np.where(jglist['time'] <= 915, jglist['date'] - 1, jglist['date'])

# 映射到最近的观测日 obs
def find_max_obs(date2):
    return obsmap.loc[obsmap['obs'] <= date2, 'obs'].max()

jglist['obs'] = jglist['date2'].apply(find_max_obs)

# 标记盘中交易（是否为观测日内且时间介于 9:20 到 15:00）
jglist['indayjy'] = ((jglist['date2'] == jglist['obs']) &
                     (jglist['time'] >= 920) &
                     (jglist['time'] <= 1500)).astype(int)

# 汇总每个 obs 和 code 的投票权重 vote
vote_df = jglist.groupby(['obs', 'code'])['vote'].sum().reset_index()

# 汇总非盘中交易的投票权重 onvote
onvote_df = jglist[jglist['indayjy'] == 0].groupby(['obs', 'code'])['vote'].sum().reset_index()
onvote_df.rename(columns={'vote': 'onvote'}, inplace=True)

# 合并 vote 和 onvote
jg = vote_df.merge(onvote_df, on=['obs', 'code'], how='left')
jg['onvote'] = jg['onvote'].fillna(0)

# 合并 obsmap 中的 buy 日期
jg = jg.merge(obsmap, on='obs', how='left')
jg.rename(columns={'buy': 'date'}, inplace=True)

# 按日期归一化 vote 和 onvote
daily_total = jg.groupby('date')[['vote', 'onvote']].sum().reset_index()
daily_total.rename(columns={'vote': 'vote2', 'onvote': 'onvote2'}, inplace=True)

jg = jg.merge(daily_total, on='date', how='left')
jg['vote'] = jg['vote'] / jg['vote2']
jg['onvote'] = jg['onvote'] / jg['onvote2']
jg = jg[['date','code','vote','onvote']]

################################################################################
# Keysets
################################################################################

# 构建宽表
ddata = pd.merge(rawsina, jg, on=['date', 'code'], how='left')
ddata['vote'] = ddata['vote'].fillna(0)
ddata['onvote'] = ddata['onvote'].fillna(0)

# 构建宽表
close = ddata.pivot(index='date', columns='code', values='close')
open_ = ddata.pivot(index='date', columns='code', values='open')
val = ddata.pivot(index='date', columns='code', values='val')
high = ddata.pivot(index='date', columns='code', values='high')
low = ddata.pivot(index='date', columns='code', values='low')
jg = ddata.pivot(index='date', columns='code', values='vote')
jg2 = ddata.pivot(index='date', columns='code', values='onvote')

# 处理缺失值和异常值
val = val.fillna(0)
close = close.where(close >= 0, np.nan)
open_ = open_.where(open_ >= 0, np.nan)

################################################################################
# Datasets
################################################################################

# 获取交集的日期
common_dates = sorted(set(jglist['date']).intersection(set(ddata['date'])))
common_dates = common_dates[100:]

# 初始化结果列表
results = []

# 遍历每个 buyi
for buyi in common_dates:
    # print(buyi)
    # buyi = 20241112
    # 确定观测市场
    i = close.index.get_loc(buyi)
    yidx = list(range(i + 1, i + 21))
    xidx = list(range(i - 1, i - 21, -1))
    # 确保索引在范围内
    yidx = [idx for idx in yidx if idx < len(close)]
    # 获取对应的日期
    yidx_dates = [close.index[idx] for idx in yidx]
    xidx_dates = [close.index[idx] for idx in xidx]
    # 获取 X 数据
    xopen = open_.loc[xidx_dates[:5]].T
    xopen.columns = [f'open{i}' for i in range(1, 6)]
    xclose = close.loc[xidx_dates[:5]].T
    xclose.columns = [f'close{i}' for i in range(1, 6)]
    xlow = low.loc[xidx_dates[:5]].T
    xlow.columns = [f'low{i}' for i in range(1, 6)]
    xhigh = high.loc[xidx_dates[:5]].T
    xhigh.columns = [f'high{i}' for i in range(1, 6)]
    xval = val.loc[xidx_dates[:5]].T
    xval.columns = [f'val{i}' for i in range(1, 6)]
    xjg = jg.loc[xidx_dates[:5]].T
    xjg.columns = [f'jg{i}' for i in range(1, 6)]
    xjg2 = jg2.loc[xidx_dates[:5]].T
    xjg2.columns = [f'ojg{i}' for i in range(1, 6)]
    # 获取 Y 数据
    if len(yidx) == 0:
        jg10 = jg.loc[xidx_dates[:10]].mean(axis=0, skipna=True)
        open0 = open_.loc[buyi]
        price_out = np.nan
    else:
        jg10 = jg.loc[xidx_dates[:10]].mean(axis=0, skipna=True)
        open0 = open_.loc[buyi]
        price_out = close.loc[yidx_dates[0]]
    # 创建数据表
    data = pd.DataFrame({
        'code': xopen.index.astype(int),
        'buy': int(buyi),
        **xopen.to_dict(orient='list'),
        **xclose.to_dict(orient='list'),
        **xlow.to_dict(orient='list'),
        **xhigh.to_dict(orient='list'),
        **xval.to_dict(orient='list'),
        **xjg.to_dict(orient='list'),
        **xjg2.to_dict(orient='list'),
        'jg10': jg10,
        'open0': open0,
        'price_out': price_out
    })
    # 计算 ztrate 和 xjj
    data['ztrate'] = np.where(np.floor(data['code'] / 10000).isin([30, 68]), 1.198, 1.098)
    data['roi'] = np.where(data['open0'] > data['close1'] * data['ztrate'], 1, data['price_out'] / data['open0'])
    # 保存结果
    results.append(data)

# 合并结果
rawdata = pd.concat(results, ignore_index=True)

# 移除所有含缺失值的样本行
datasets = rawdata.dropna().copy()

# 计算按 buy 聚合的统计指标

datasets['btoday1'] = datasets['close1'] / datasets['open1']
datasets['bonite0'] = datasets['open0'] / datasets['close1']
datasets['broi1'] = datasets['close1'] / datasets['open2']
datasets['broi5'] = datasets['close1'] / datasets['open5']

# 按 buy 聚合后取平均与标准差
buy_stats = datasets.groupby('buy').agg({
    'btoday1': 'mean',
    'bonite0': 'mean',
    'broi1': 'mean',
    'broi5': 'mean',
    'roi': ['mean', 'std']
}).reset_index()

buy_stats.columns = ['buy', 'btoday1', 'bonite0', 'broi1', 'broi5', 'roi_mean', 'roi_std']
datasets = datasets.drop(columns=['btoday1', 'bonite0', 'broi1', 'broi5'])
datasets = datasets.merge(buy_stats, on='buy', how='left')

# 计算 z 分数与分类编码
datasets['z'] = (datasets['roi'] - datasets['roi_mean']) / datasets['roi_std']
datasets['class'] = (
    (datasets['btoday1'] > 1).astype(int) * 1000 +
    (datasets['bonite0'] > 1).astype(int) * 100 +
    (datasets['broi1'] > 1).astype(int) * 10 +
    (datasets['broi5'] > 1).astype(int)
)

#Training Data

mfiles_train = []
for i in datasets['class'].unique():
    datai = datasets[datasets['class'] == i].copy()
    datai['ztrate2'] = (datai['high1'] / datai['close2'] >= datai['ztrate']).astype(int)
    datai['ztrate3'] = (datai['low1'] / datai['close2'] >= datai['ztrate']).astype(int)
    # 特征变换
    datai['open1'] = datai['open1'] / datai['open0']
    datai['close1'] = datai['close1'] / datai['open0']
    datai['high1'] = datai['high1'] / datai['open0']
    datai['low1'] = datai['low1'] / datai['open0']
    datai['jg35'] = datai['jg3'] + datai['jg4'] + datai['jg5']
    datai['closegr'] = datai['close1'] / (datai['close2'] + datai['close3'] + datai['close4'] + datai['close5']) * 4
    datai['y'] = np.log(datai['roi'])
    datai['onite0'] = 1 / datai['close1']
    # 选择输出列
    feature_cols = ['open1', 'onite0', 'high1', 'low1', 'jg1', 'jg2', 'jg35', 'jg10', 'ojg1',
                    'closegr', 'ztrate', 'ztrate2', 'ztrate3',
                    'btoday1', 'bonite0', 'broi1', 'broi5']
    label_cols = ['y', 'z', 'class']
    xi = datai[feature_cols].to_numpy()
    yi = datai[['buy', 'code'] + label_cols].to_numpy()
    # 拼接成训练数据
    combined = np.hstack([xi, yi])
    combined_df = pd.DataFrame(combined, columns=feature_cols + ['buy', 'code', 'y', 'z', 'class'])
    mfiles_train.append(combined_df)

end_time = time.time()
print(f"完成数据处理，耗时：{end_time - start_time:.2f} 秒")

################################################################################################
# Training Model
################################################################################################

#参数设置

sample_rate = 0.1
seeds = np.random.RandomState(1).choice(10000, 500, replace=False)

def train_model(seed, mfile, sample_rate):
    np.random.seed(seed)
    # print(seed)
    idx = np.random.choice(
        mfile.shape[0],
        int(sample_rate * mfile.shape[0]),
        replace=False
    )
    sampled = mfile.iloc[idx].copy()
    X = sampled.drop(columns=['buy', 'code', 'z', 'class', 'y'])
    Y = sampled[['y', 'z']]
    model = DecisionTreeRegressor(ccp_alpha=0.001)
    model.fit(X, Y)
    return model


# 全量训练开始

models = []
start_time = time.time()
print(datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))
for mfile in mfiles_train:
    cls = mfile['class'].iloc[0]
    print(f"训练 class={cls} @ {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
    # 并行训练多个随机子模型
    modeli = Parallel(n_jobs=-1, backend='loky')(
        delayed(train_model)(seed, mfile, sample_rate) for seed in seeds
    )
    models.append({'class': cls, 'models': modeli})

end_time = time.time()
print(f"训练耗时：{end_time - start_time:.2f} 秒")


