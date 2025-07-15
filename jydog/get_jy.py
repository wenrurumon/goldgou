
###############################################################
# 调出页面
###############################################################

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import akshare as ak
import pickle
import time
import datetime
import pandas as pd
import numpy as np
import re
import os
import itertools
import json

def refreshslides(nslides):
    for i in range(nslides):
        print(i,datetime.datetime.now())
        time.sleep(1)
        d.execute_script("window.scrollTo(0, document.body.scrollHeight);")

def savedata(filename):
    elements = d.find_elements(By.XPATH, "//div[@watermark='main']")
    len(elements)
    outs = []
    for element in elements:
        text = element.text
        outs.append(text)
    with open(filename, "wb") as file:
        pickle.dump(outs, file)

def savedata2(filename):
    outs = pickle.load(open(filename, "rb"))
    elements = d.find_elements(By.XPATH, "//div[@watermark='main']")
    for element in elements:
        text = element.text
        outs.append(text)
    outs = sorted(list(set(outs)))
    with open(filename, "wb") as file:
        pickle.dump(outs, file)

#当需要扫码登录时，进入服务器获取cookies

service = Service('/Users/huzixin/Downloads/chromedriver-mac-x64/chromedriver')
# d = webdriver.Chrome(service=service)
# d.get('https://wx.zsxq.com/login')

# cookies = d.get_cookies()
# d.quit()

# with open('data/cookies.json', 'w') as f:
#     json.dump(cookies, f)

#导入cookies，以禁用图片模式进入浏览器

chrome_options = Options()
prefs = {"profile.managed_default_content_settings.images": 2}  # 2 = 禁用图片
chrome_options.add_experimental_option("prefs", prefs)
d = webdriver.Chrome(service=service,options=chrome_options)
d.get('https://wx.zsxq.com/login')

with open('data/cookies.json', 'r') as f:
    cookies = json.load(f)

for cookie in cookies:
    d.add_cookie(cookie)

d.get('https://wx.zsxq.com/login')

################################################################
#更新页面
###############################################################

#加载原先的bin

filename = 'data/jy2507.bin'

#刷新页面

refreshslides(30)
savedata2(filename)

################################################################
#更新jglist
###############################################################

# 得到codelist

# allcodes = pd.concat([ak.stock_sh_a_spot_em(), ak.stock_sz_a_spot_em()], ignore_index=True)
# allcodes = allcodes[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
# allcodes = allcodes.drop_duplicates()
# allcodes['namecn'] = allcodes['name'].str.replace(r'\*| ', '', regex=True)
# allcodes.to_csv('/Users/huzixin/Documents/goldgou/data/todayscodes.csv',index=False)
allcodes2 = pd.read_csv('data/todayscodes.csv')
allcodes2['code'] = (allcodes2['code'].astype(int) + 1000000).astype(str).str[-6:]
allcodes = allcodes2

# 获取所有纪要

files = [f"data/{file}" for file in os.listdir("data") if file.endswith(".bin")]
df = pd.DataFrame({
    "files": files,
    "month": [int(re.sub(r'data/jy|\.bin', '', file)) for file in files]
})
df = df.sort_values(by="month", ascending=False)

# 更新纪要数据

filtered_files = df[df['month']==max(df["month"])]['files'].tolist()
raw = [pickle.load(open(file, "rb")) for file in filtered_files]
raw = sorted(list(set(itertools.chain.from_iterable(raw))))
raw = [text for text in raw if "复盘" not in text and "策略" not in text]

jglist = [
    [idx, lines[1]] + sorted([name for name in allcodes['namecn'] if name in item])
    for idx, item in enumerate(raw) if (lines := item.split("\n")) and len(lines) > 1
]

#输出

jglist = [
    [line[0], line[1].split(" ")[0], line[1].split(" ")[1], name]
    for line in jglist for name in line[2:]
]
jglist = pd.DataFrame(jglist, columns=['jgid', 'date', 'time', 'namecn'])
jglist = jglist.merge(allcodes, left_on='namecn', right_on='namecn', how='left').drop(columns=['namecn']).sort_values(by=['date', 'code'], ascending=[True, True])
jglist.to_csv('data/jglist_new.csv')

###############################################################
#全量页面刷新

#刷页面

# refreshslides(400)
# refreshslides(100)

# #读页面

# savedata('data/jy2507.bin')



