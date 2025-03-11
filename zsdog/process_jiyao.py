
import pandas as pd
import numpy as np
import re
import os
import itertools
import pickle
import akshare as ak

# 读取所有股票代码
allcodes = pd.read_csv('data/allcodes.csv').iloc[:, 1:]
allcodes['namecn'] = allcodes['name'].str.replace(r'\*| ', '', regex=True)

# 获取所有纪要
files = [f"data/{file}" for file in os.listdir("data") if file.endswith(".bin")]
df = pd.DataFrame({
    "files": files,
    "month": [int(re.sub(r'data/jy|\.bin', '', file)) for file in files]
})
filtered_files = df[df["month"] < 2503]["files"].tolist()
filtered_files = sorted(filtered_files)

raw = [pickle.load(open(file, "rb")) for file in filtered_files]
raw = sorted(list(set(itertools.chain.from_iterable(raw))))

# 处理所有纪要，匹配股票代码

jglist = [
    [idx, lines[1]] + sorted([name for name in allcodes['namecn'] if name in item])
    for idx, item in enumerate(raw) if (lines := item.split("\n")) and len(lines) > 1
]
jglist = [
    [line[0], line[1].split(" ")[0], line[1].split(" ")[1], name]
    for line in jglist for name in line[2:]
]

jglist = pd.DataFrame(jglist, columns=['jgid', 'date', 'time', 'namecn'])
jglist = jglist.merge(allcodes, left_on='namecn', right_on='namecn', how='left').drop(columns=['namecn']).sort_values(by=['date', 'code'], ascending=[True, True])

#读取股票数据

ddata = pd.read_csv('data/exedata.csv').iloc[:, 1:]
dateid = pd.to_datetime(sorted(pd.unique(ddata['日期'])))

jglist['date'] = pd.to_datetime(jglist['date'])
jglist['time'] = pd.to_datetime(jglist['time'], format='%H:%M').dt.time  # 转换时间格式
date_indices = np.searchsorted(dateid, jglist['date'], side='right')
next_dates = pd.Series(pd.NaT, index=jglist.index)  # 先创建 NaT 的 Series
valid_indices = date_indices < len(dateid)  # 仅选择有效索引
next_dates.loc[valid_indices] = dateid[date_indices[valid_indices]]  # 仅对有效索引赋值

jglist['buy'] = np.where(
    jglist['date'].isin(dateid) & (jglist['time'] < pd.to_datetime('15:00', format='%H:%M').time()),
    jglist['date'],  # 在 dateid 且 time < 15:00，则 date2 = date
    next_dates  # 否则，取比 date 大的下一个 dateid，超出范围返回 NaT
)

jglist = jglist.sort_values(by=['jgid'])
jglist.to_csv('data/jglist_till2502.csv')
