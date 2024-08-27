
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(xgboost)
library(caret)
library(parallel)
# use_condaenv(condaenv='/Users/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
# setwd('/Users/huzixin/Documents/jingou')
use_condaenv(condaenv=system('which python', intern = TRUE),required=TRUE)
setwd('/home/huzixin/documents/goldgou')

########################################################################################################################

#Get Historical Data

system.time(system('Rscript /home/huzixin/documents/goldgou/gethisdata.R'))

hisdata <- fread('hisdata210101.csv')[,-1]
dates <- sort(unique(as.numeric(gsub('-','',unique(hisdata$日期)))))

########################################################################################################################

rlts <- lapply(220:0,function(datei){
  #Sample Data
  print(paste(datei,Sys.time()))
  raw <- hisdata %>% 
    select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
    mutate(date=gsub('-','',date)) %>%
    filter(date%in%dates[length(dates)-0:28-datei]) 
  raw$did <- match(raw$date,sort(unique(raw$date)))
  #ROI Matrix
  test <- raw %>%
    select(codei=code,did,date0=date,open0=open,close0=close,val0=val) %>%
    as.data.frame 
  test <- test %>%
    filter(val0>=quantile(test$val0,0.8),close0/open0>=max(mean(test$close0/test$open0),1))
  raw1 <- unique(raw[, .(codej = code, did = did - 1, date1 = date, open1 = open)])
  raw2 <- unique(raw[, .(codej = code, did = did - 2, date2 = date, close2 = close)])
  system.time(merged_raw <- merge(raw1, raw2, by = c('did','codej'), allow.cartesian = TRUE))
  merged_raw[, roi := close2 / open1]
  system.time(roimat <- merge(test, merged_raw, by = "did") %>% as.data.table)
  system.time(
    roimat <- roimat[, .(
      roi = mean(close2 / open1),
      n = sum(close2 > open1),
      N = .N
    ), by = .(codei, codej)]
  )
  roimat[, pval := pbinom(n, N, prob = 0.5, lower.tail = FALSE)]
  #Reference
  ref <- raw %>%
    select(date0=date,code,did,open0=open) %>%
    merge(
      raw %>%
        mutate(did=did-1) %>%
        select(date1=date,code,did,close1=close)
    ) %>%
    mutate(roi=close1/open0) %>%
    arrange(did) %>%
    group_by(code) %>%
    summarise(
      periodroi=close1[did==max(did)]/open0[did==min(did)],
      prodroi=prod(roi),refroi=mean(roi),refprob=mean(roi>1)
    )
  #Calculate
  test <- raw %>%
    filter(date==max(raw$date)) %>%
    select(codei=code,did,date0=date,open0=open,close0=close,val0=val) 
  test <- test %>%
    filter(val0>=quantile(test$val0,0.8),close0/open0>=max(mean(test$close0/test$open0),1))
  test <- test %>%
    merge(
      roimat
    ) %>%
    group_by(obsday=max(raw$date),code=codej) %>%
    summarise(roi=sum(N*roi)/sum(N),n=sum(n),N=sum(N),prob=n/N) %>%
    merge(
      ref
    ) %>%
    mutate(pval=pbinom(n,N,0.5,lower.tail=F)) %>%
    merge(
      hisdata %>%
        select(code,name) %>%
        unique()
    ) %>%
    mutate(uplift=prob/refprob) 
  test
  # test %>%
  #   filter(pval<0.05,refprob>0.6,ttroi>1) %>%
  #   arrange(desc(roi*prob)) %>%
  #   head()
  # test %>%
  #   filter(pval<0.05,refprob>0.6,refroi>1) %>%
  #   arrange(desc(prob*roi)) %>%
  #   head
})

lapply(rlts,function(x){
  x %>%
    select(code,name,obsday,roi,prob,pval,uplift,refprob,ttroi) %>%
    filter(roi>1,prob>0.6,pval<0.05,prob>refprob) %>%
    arrange(desc(roi)) %>%
    head()
})

rlt <- do.call(rbind,rlts)
write.csv(rlt,'trace_nnn2.csv')

#Check

tracedata <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=gsub('-','',date)) %>%
  filter(as.numeric(date)>=min(rlt$obsday))

tracedata$did <- match(tracedata$date,sort(unique(tracedata$date)))

rlt %>%
  merge(
    tracedata %>%
      select(did,code,obsday=date) %>%
      merge(
        tracedata %>%
          mutate(did=did-1) %>%
          select(did,code,buyday=date,open)
      ) %>%
      merge(
        tracedata %>%
          mutate(did=did-1) %>%
          select(did,code,sellday=date,close)
      ) %>%
      mutate(realroi=close/open)
  ) %>%
  select(code,name,obsday,roi,prob,pval,uplift,realroi,refprob) %>%
  filter(roi>1,prob>0.6,pval<0.05,prob>refprob) %>%
  arrange(desc(roi)) %>%
  group_by(obsday) %>%
  summarise(realroi=mean(realroi[1:5])) %>%
  group_by(substr(obsday,1,6)) %>%
  summarise((prod(realroi)-1)/2+1)
