
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(stringr)
library(purrr)
library(randomForest)

ak <- import('akshare')

setwd('/Users/huzixin/Documents/goldgou/')

########################################################################################################################
# Get Historical Data

updatedata <- function(start_date){
  
  start_date <- 20180101
  
  allcodes <- rbind(
    ak$stock_sh_a_spot_em(),
    ak$stock_sz_a_spot_em()
  ) %>%
    select(code=2,name=3)
  write.csv(allcodes,'data/allcodes.csv')
  # allcodes <- fread('data/allcodes.csv')[,-1]
  
  #Get His Data
  
  getcodei <- function(codei){
    x <- ak$stock_zh_a_hist(symbol=codei,
                            period='daily',
                            start_date=start_date,
                            end_date=gsub('-','',Sys.Date()), adjust='hfq') %>%
      mutate(code=codei)
    x[[1]] <- sapply(x[[1]],paste)
    x
  }
  
  system.time(
    hisdata <- lapply(allcodes$code,function(codei){
      print(codei)
      try(getcodei(codei))
    })
  )
  
  hisdata <- do.call(rbind,hisdata[which(sapply(hisdata,is.data.frame))])
  
  write.csv(hisdata,paste0('data/hisdata.csv'))
  
}

updateusdata <- function(){
  
  # start_date <- 20180101
  
  uscodes <- ak$stock_us_spot_em() %>%
    filter(grepl('中国',`名称`),grepl('ETF',`名称`)) %>%
    select(name=`名称`,code=`代码`) %>%
    mutate(code=substr(code,5,nchar(code)))
  
  getcodei <- function(codei){
    x <- ak$stock_us_daily(symbol=codei,adjust='qfq') %>% mutate(code=codei)
    x
  }
  
  system.time(
    hisdata <- lapply(uscodes$code,function(codei){
      print(codei)
      try(getcodei(codei))
    })
  )
  
  hisdata <- do.call(rbind,hisdata[which(sapply(hisdata,is.data.frame))])
  
  write.csv(hisdata,'data/usdata.csv')
  
}

#Full Data

# system.time(updatedata(20180101))
# updateusdata()

################################################################################
#导入原始数据

#纪要数据

jglist <- fread('data/rawjglist.csv')[,-1] %>%
  mutate(date=as.numeric(gsub('-','',date)),time=as.numeric(gsub(':','',time))) %>%
  filter(!code%in%c(30024,688981,858,300152),!grepl('证券|ST',name))

#东财数据

allcodes <- fread('data/allcodes.csv')[,-1]

ddata <- fread('data/hisdata.csv')[,-1] %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  unique()

datemap <- sort(unique(ddata$date))
datemap <- data.table(date=datemap,did=1:length(datemap))

obsmap <- data.table(
  obs=datemap$date,
  buy=c(datemap$date[-1],NA),
  sell=c(datemap$date[-1:-2],rep(NA,2))
) 

#整合数据

ddata <- ddata %>% filter(code%in%jglist$code)
jglist <- jglist %>% filter(code%in%ddata$code)

jglist <- jglist %>%
  merge(
    jglist %>%
      group_by(jgid) %>%
      summarise(vote=n())    
  ) %>%
  mutate(vote=1/vote)

################################################################################
#处理数据

close <- ddata %>% acast(date~code,value.var='close')
open <- ddata %>% acast(date~code,value.var='open')
val <- ddata %>% acast(date~code,value.var='val')
high <- ddata %>% acast(date~code,value.var='high')
low <- ddata %>% acast(date~code,value.var='low')
val[is.na(val)] <- 0
close[close<0] <- NA
open[open<0] <- NA
roi <- close[-1,]/open[-nrow(open),]

#处理机构数据

jglist <- jglist %>%
  mutate(date2=ifelse(time<=915,date-1,date))

jglist$obs <- sapply(jglist$date2,function(i){
  max(obsmap$obs[obsmap$obs<=i])
})

jglist <- jglist %>% select(-date2)

jglist2 <- jglist %>%
  group_by(obs,code) %>%
  summarise(vote=sum(vote)) %>%
  acast(obs~code,value.var='vote',fill=0)

jg<- t(t(jglist2)/rowSums(jglist2))

################################################################################
#测试

test <- melt(roi) %>%
  select(obs=1,code=2,roi=3) %>%
  filter(obs%in%rownames(jg)) %>%
  merge(
    melt(jg) %>%
      select(obs=1,code=2,vote=3) %>%
      filter(vote>0),
    all.x=T
  ) %>%
  mutate(vote=ifelse(is.na(vote),0,vote))

(test %>%
    group_by(obs,jg=vote>0) %>%
    summarise(roi=mean(roi-1,na.rm=T)/2+1) %>%
    dcast(obs~jg) %>%
    mutate(idx=`TRUE`/`FALSE`))$idx %>% summary

test %>%
  group_by(obs,jg=vote>0) %>%
  summarise(roi=mean(roi-1,na.rm=T)/2+1) %>%
  dcast(obs~jg) %>%
  mutate(idx=`TRUE`/`FALSE`) %>%
  group_by(floor(obs/10000)) %>%
  summarise(mean(idx),sd(idx),mean(idx>=1),n())

################################################################################
#策略

system.time(
  
  datasets <- lapply(unique(jglist$date)[which(unique(jglist$date) %in% obsmap$obs)][-1:-100],function(buyi){
    
    # buyi <- 20241112
    buyi <- paste(buyi)
    print(buyi)
    
    #确定观测市场
    
    i <- which(rownames(close)==buyi)
    yidx <- i+1:20
    xidx <- i-1:20
    yidx <- yidx[yidx<=nrow(close)]
    
    yidx <- rownames(close)[yidx]
    xidx <- rownames(close)[xidx]
    
    if(length(yidx)==0){
      return(NULL)
    }
    
    #确定一下要看的池子
    
    jgcode <- names(which(jg[xidx[1],]>0))
    data.table(
      jg=mean(roi[xidx[1],jgcode],na.rm=T),
      bench=mean(roi[xidx[1],],na.rm=T)
    ) %>%
      mutate(uplift=jg/bench)
    
    #银狗
    
    Y <- data.table(
      code=as.numeric(colnames(close)),
      buy=as.numeric(buyi),
      open0 = open[buyi,],
      open1 = open[xidx[1],],
      open2 = open[xidx[2],],
      open5 = open[xidx[5],],
      high1 = high[xidx[1],],
      close1 = close[xidx[1],],
      close2 = close[xidx[2],],
      close5 = close[xidx[5],],
      val1 = val[xidx[1],],
      val2 = colMeans(val[xidx[1:2],],na.rm=T),
      val5 = colMeans(val[xidx[1:5],],na.rm=T),
      jg1=jg[xidx[1],],
      jg2=colMeans(jg[xidx[2],,drop=F],na.rm=T),
      jg5=colMeans(jg[xidx[3:5],],na.rm=T),
      jg10=colMeans(jg[xidx[6:10],],na.rm=T),
      xjj=close[yidx[1],]/open[buyi,]
    ) %>%
      mutate(
        zt1=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198 * close2, 1.098 * close2),
        zt0=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198 * close1, 1.098 * close1),
        xjj=ifelse(open0>=zt0,1,xjj)
      ) %>%
      mutate(
        code = as.numeric(code),
        onite0 = open0/close1,
        today1 = close1/open1,
        oday0 = open0/open1,
        ztouch1 = high1/zt1,
        zend1 = close1/zt1,
        valg = val1/val5,
        roi1 = sqrt(close1/open2),
        roi5 = (close1/open5)^(1/5)
      )
    
    #输出
    
    Y
    
  })
 
)

datasets <- rbindlist(datasets[which(!sapply(datasets,is.null))])
datasets <- datasets[rowSums(is.na(datasets))==0,]

bench <- datasets %>%
  group_by(buy) %>%
  summarise(
    bench.xjj = mean(xjj,na.rm=T),
    bench.today1 = sum(today1*val1,na.rm=T)/sum(val1,na.rm=T),
    bench.roi1 = sum(roi1*val2,na.rm=T)/sum(val2,na.rm=T),
    bench.roi5 = sum(roi5*val5,na.rm=T)/sum(val5,na.rm=T),
  )

mfile <- datasets %>% merge(bench) %>% mutate(score=xjj/bench.xjj)

#策略

mfile %>%
  group_by(
    # bench.roi1=bench.roi1>1,
    bench.today1=bench.today1>1,
    bench.roi5=bench.roi5>1,
    jg1=jg1 > 0,
    jg2=jg2 > 0,
    jg5=jg5 > 0,
    jg10=jg10 > 0,
    onite0=onite0 > 1,
    today1=today1 > 1,
    oday0=oday0 > 1,
    ztouch1=ztouch1 > 1
  ) %>%
  summarise(
    mean=mean(score),sd=sd(score),n=n()
  ) %>%
  filter(
    n>0
  ) %>%
  arrange(
    desc(mean)
  ) %>%
  filter(bench.today1,bench.roi5) %>%
  filter(mean>1) %>%
  head(40) %>%
  as.data.frame

#对于1日和5日大盘收益都正的天，
#取obs当天有纪要但6-10天没纪要
#优先2-5天也没有纪要
#优先昨天触发过涨停
#优先低开

temp <- mfile %>%
  group_by(
    bench.today1=bench.today1>1,
    bench.roi5=bench.roi5>1,
    jg1=jg1 > 0,
    jg2=jg2 > 0,
    jg5=jg5 > 0,
    jg10=jg10 > 0,
    jgs10 = jg1+jg2+jg5+jg10>0,
    jgs10_1 = jg2+jg5+jg10>0,
    onite0=onite0 > 1,
    today1=today1 > 1,
    oday0=oday0 > 1,
    ztouch1=ztouch1 > 1
  ) %>%
  summarise(
    mean=mean(score)
    # ,sd=sd(score)
    ,n=n()
  ) %>%
  filter(
    n>0
  ) %>%
  arrange(
    desc(mean)
  ) %>%
  filter(bench.today1,bench.roi5,ztouch1) %>%
  filter(mean>1) 

plot.ts(
  sapply(3:(ncol(temp)-2),function(i){
    cumsum(temp[[i]]*temp$n)/cumsum(temp$n)
  })
)

mfile %>%
  filter(bench.roi1>1,bench.roi5>1) %>%
  filter(ztouch1>0.95,jg1>0,jg2+jg5+jg10==0,today1>1,onite0<1) %>%
  summarise(mean(score,na.rm=T))

#暴跌行情

temp <- mfile %>%
  group_by(
    bench.today1=bench.today1>1,
    bench.roi5=bench.roi5>1,
    jg1=jg1 > 0,
    jg2=jg2 > 0,
    jg5=jg5 > 0,
    jg10=jg10 > 0,
    jgs10 = jg1+jg2+jg5+jg10>0,
    jgs10_1 = jg2+jg5+jg10>0,
    onite0=onite0 > 1,
    today1=today1 > 1,
    oday0=oday0 > 1,
    ztouch1=ztouch1 > 1
  ) %>%
  summarise(
    mean=mean(score)
    # ,sd=sd(score)
    ,n=n()
  ) %>%
  filter(
    n>0
  ) %>%
  arrange(
    desc(mean)
  ) %>%
  filter(!bench.today1,!bench.roi5) %>%
  filter(mean>1) %>%
  as.data.frame()
  # tail()

################################################################
# Data for XGboost

system.time(
  
  models <- lapply(1:100,function(i){
    
    print(i)
    
    set.seed(i)
    
    sel <- sample(1:nrow(mfile),1000)
    
    train <- mfile[sel,] %>%
      select(
        bench.roi1,
        bench.roi5,
        jg1,
        jg2,
        jg5,
        jg10,
        zt1,
        onite0,
        today1,
        oday0,
        ztouch1,
        zend1,
        valg,
        y=score
      ) %>%
      mutate(y=log(y)) %>%
      mutate(y=as.factor(y>0))
    
    modeli <- randomForest(
      y~.,
      data=train,
      ntree = 500,        # 增加树的数量
      mtry = 5,           # 每棵树随机选择的特征数（尝试多个）
      importance = TRUE
    )
  
    modeli
    
  })
  
)

outs <- sapply(models,function(modeli){
  
  predict(modeli,
          newdata=mfile %>%
            filter(buy>=20240901) %>%
            select(
              bench.roi1,
              bench.roi5,
              jg1,
              jg2,
              jg5,
              jg10,
              zt1,
              onite0,
              today1,
              oday0,
              ztouch1,
              zend1,
              valg
            )
          )
  
})



