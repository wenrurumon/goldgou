
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import time
import datetime

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

#打开浏览器

d = webdriver.Chrome()
d.get('https://wx.zsxq.com/login')#找王爸爸扫码

#刷页面

refreshslides(300)
# refreshslides(100)

#读页面

savedata('data/jy2208.bin')
