
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)

# use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
# setwd('/home/huzixin/documents/goldgou')

setwd('/Users/huzixin/Documents/goldgou/')

########################################################################################################################
# Get Historical Data

#Load Data

allcode <- fread('data/allcodes.csv')[,-1]

jgp <- read.csv('data/raw_grade.csv')[,-1] %>%
  filter(grade%in%c('增持','买入'),ex.price!='-') %>%
  mutate(ex.price=as.numeric(ex.price)) %>%
  group_by(code,date) %>%
  summarise(eprice=min(ex.price)) %>%
  filter(eprice!='-')

#Merge Data

allcode <- allcode %>% filter(code%in%jgp$code)
jgp <- jgp %>% filter(code%in%allcode$code)

ddata <- fread('data/hisdata.csv')[,-1] %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  filter(code%in%allcode$code)

datemap <- sort(unique(ddata$date))
datemap <- data.table(date=datemap,did=1:length(datemap))

obsmap <- data.table(
  obs=datemap$date,
  buy=c(datemap$date[-1],NA),
  sell=c(datemap$date[-1:-2],rep(NA,2))
)

################################################################################
#Daily Data Model

#Big Table

close <- ddata %>% acast(date~code,value.var='close')
open <- ddata %>% acast(date~code,value.var='open')
val <- ddata %>% acast(date~code,value.var='val')
high <- ddata %>% acast(date~code,value.var='high')
low <- ddata %>% acast(date~code,value.var='low')
val[is.na(val)] <- 0

for(i in 2:nrow(close)){
  close[i,] <- ifelse(is.na(close[i,]),close[i-1,],close[i,])
  open[i,] <- ifelse(is.na(open[i,]),open[i-1,],open[i,])
  high[i,] <- ifelse(is.na(high[i,]),high[i-1,],high[i,])
  low[i,] <- ifelse(is.na(low[i,]),low[i-1,],low[i,])
}

close <- apply(close,2,function(x){
  x[which(!is.na(x))[1]+0:5] <- NA
  x
})
open <- apply(open,2,function(x){
  x[which(!is.na(x))[1]+0:5] <- NA
  x
})
val <- apply(val,2,function(x){
  x[which(!is.na(x))[1]+0:5] <- NA
  x
})

################################################################################
#Module

ogou <- function(iminus,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05){
  # iminus <- 1000
  # b <- 29
  # k <- 2
  # codek0 <- 15
  # codek2 <- 10
  # thres_roi <- 0.05
  # thres_pop <- 0.05
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
  pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
  if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}
  i <- nrow(roi)-iminus
  startdid0 <- datemap$date[which(datemap$date==rownames(roi)[i])-codek0-k]
  enddid0 <- datemap$date[which(datemap$date==rownames(roi)[i])-k]
  jgpi0 <- jgp %>% filter(date>startdid0,date<=enddid0)
  codei0 <- unique(jgpi0$code) %>% paste
  roi0 <- roi[i-(b:0+k),codei0]
  pop0 <- pop[i-(b:0+k),codei0]
  roi1 <- roi[i-(b:0),codei0]
  startdid2 <- datemap$date[which(datemap$date==rownames(roi)[i])-codek2]
  enddid2 <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi2 <- jgp %>% filter(date>startdid2,date<=enddid2)
  codei2 <- unique(jgpi2$code) %>% paste
  roi2 <- roi[i-(b:0),codei2]
  pop2 <- pop[i-(b:0),codei2]
  x.ref <- (rbind(roi=roi0,pop=pop0))
  x.obs <- (rbind(roi=roi2,pop=pop2))
  if((ncol(x.obs)==0)|(ncol(roi1)==0)){
    return(NULL)
  }
  y.ref <- data.table(
    code=colnames(roi1),
    obs=as.numeric(rownames(roi)[i-2]),
    buy=as.numeric(rownames(roi)[i-1]),
    sell=as.numeric(rownames(roi)[i-1+k]),
    roi=roi1[nrow(roi1),]
  )
  droi <- apply(roi2,2,function(x){colSums((x-roi0)^2)}) %>%
    log %>% scale %>% pnorm %>%
    melt() %>%
    select(code0=1,code2=2,droi=3) 
  dpop <- apply(pop2,2,function(x){colSums((x-pop0)^2)}) %>%
    log %>% scale %>% pnorm %>%
    melt() %>%
    select(code0=1,code2=2,dpop=3)
  data.table(
    obs=as.numeric(rownames(roi)[i]),
    buy=as.numeric(rownames(roi)[i+1]),
    sell=as.numeric(rownames(roi)[i+k]),
    droi,
    dpop=dpop$dpop
  ) %>%
    merge(y.ref %>% mutate(code=as.numeric(code)) %>% select(code0=1,roi)) %>%
    filter(droi<=thres_roi,dpop<=thres_pop) %>%
    group_by(obs,buy,sell,code=code2) %>%
    summarise(mean=mean(roi),win=mean(roi>1)) %>%
    filter(mean>1,win>0.5) %>%
    arrange(desc(mean)) 
}

ygou <- function(iminus){
  
  i <- nrow(close)-iminus
  
  yg1 <- data.table(
    # T-1日收盘价一阶导大于T-1日收盘价一阶导20日均线
    (close[i,]-close[i-1,])>(colMeans(close[i-19:0,]-close[i-20:1,])),
    # T-1日收盘价一阶导大于0.01
    close[i,]-close[i-1,]>0.01,
    # T-1日成交额大于T-1日成交额20日均线
    val[i,]>(colMeans(val[i-19:0,])),
    # T-1日收盘价小于最高价
    close[i,]<high[i,],
    # T-1日开盘价小于收盘价
    close[i,]>open[i,],
    # T-1日收盘价一阶导大于T-2日收盘价一阶导
    (close[i,]-close[i-1,])>(close[i-1,]-close[i-2,]),
    # T-1日成交额大于T-2日成交额
    val[i,]>val[i-1,],
    # T-1日收盘价大于T-2日收盘价
    close[i,]>close[i-1,],
    # T-2日收盘价一阶导大于T-2日收盘价一阶导20日均线*2
    (close[i-1,]-close[i-2,])>(colMeans(close[i-20:1,]-close[i-21:2,])*2),
    # T-1日K线实体部分*2>上下引线部分之和
    (abs(open[i,]-close[i,])*2)>
      (abs(ifelse(open[i,]>close[i,],open[i,],close[i,])-high[i,])+
         abs(ifelse(open[i,]<close[i,],open[i,],close[i,])-low[i,]))
  )
  
  yg2 <- data.table(
    # T-1日收盘价一阶导大于T-1日收盘价一阶导20日均线
    (close[i,]-close[i-1,])>(colMeans(close[i-19:0,]-close[i-20:1,])),
    # T-1日成交额大于T-1日成交额20日均线*2
    val[i,]>(colMeans(val[i-19:0,])*2),
    # T-1日开盘价小于收盘价
    close[i,]>open[i,],
    # T-1日收盘价大于T-2日收盘价
    close[i,]>close[i-1,],
    # T-2日收盘价一阶导大于T-2日收盘价一阶导20日均线
    (close[i-1,]-close[i-2,])>(colMeans(close[i-20:1,]-close[i-21:2,])),
    # T-2日涨幅超过9.8%
    close[i-1,]/close[i-2,]>1.098,
    # T-1日K线实体部分*2>上下引线部分之和
    (abs(open[i,]-close[i,])*2)>
      abs(ifelse(open[i,]>close[i,],open[i,],close[i,])-high[i,])+abs(ifelse(open[i,]<close[i,],open[i,],close[i,])-low[i,])
  )

  yg3 <- data.table(
    # T-1日涨幅超过5%
    close[i,]/close[i-1,]>1.05,
    # T-2日涨幅小于0
    close[i-1,]-close[i-2]>0,
    # T-3日涨幅超过5%
    close[i-2,]/close[i-3,]>1.05,
    # T-1日成交额大于T-2日
    val[i,]>val[i-1,],
    # T-3日成交额大于T-2日
    val[i-2,]>val[i-1,],
    # T-1日上引线不超过2.5%
    high[i,]/ifelse(open[i,]>close[i,],open[i,],close[i,])<1.025
  )
  
  idx <- close[i,]-2*colMeans(close[i-19:0,])
  
  obsmap %>%
    merge(
      data.table(
        obs=as.numeric(rownames(close)[i]),
        code=as.numeric(colnames(close)),
        mean=close[i,]-2*colMeans(close[i-19:0,]),
        yg1=rowMeans(yg1),yg2=rowMeans(yg2),yg3=rowMeans(yg3)
      ) %>% 
        mutate(
          win=((yg1==1)+(yg2==1)+(yg3==1))/3
        )
    ) %>%
    filter(win>0) %>%
    arrange(desc(mean)) %>%
    select(obs,buy,sell,code,mean,win)
  
}

ngou <- function(iminus,b=29,k=2,thres_valp=0.2,thres_roi=0.05,thres_pop=0.05){
  
  # 参数输入
  iminus <- 2
  b <- 5
  k <- 2
  thres_valp <- 0.2
  thres_roi <- 0.05
  thres_pop <- 0.05
  
  print(paste(iminus,Sys.time()))
  
  #准备数据
  today <- close[k:nrow(close),]/open[k:nrow(close),] 
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  valp <- t(apply(val[-1:-(k-1),],1,function(x){x/sum(x)}))
  pop <- val[-1:-(k-1),]
  
  #准备池子
  
  i <- nrow(roi)-iminus
  sel1 <- which((today[i,]>1)&(pop[i,]>pop[i-1,])&(valp[i,]>quantile(valp[i,],thres_valp)))
  roi1 <- roi[i-(b:0),sel1] 
  pop1 <- pop[i-(b:0),sel1] 
  pop1 <- t(t(pop1)/pop1[nrow(pop1),])
  roi1 <- log(roi1)
  pop1 <- log(pop1)
  sel0 <- which((today[i-k,]>1)&(pop[i-k,]>pop[i-k-1,])&(valp[i-k,]>quantile(valp[i-k,],thres_valp)))
  roi0 <- roi[i-(b:0+k),sel0]
  pop0 <- pop[i-(b:0+k),sel0]
  pop0 <- t(t(pop0)/pop0[nrow(pop0),])
  roi0 <- log(roi0)
  pop0 <- log(pop0)
  y0 <- data.table(
    refcode=as.numeric(names(sel0)),
    roi=roi[i,sel0]
  )
  
  #计算距离
  
  dist.roi <- apply(roi1,2,function(x){colMeans((x-roi0)^2)}) %>%
    log %>% scale %>% pnorm %>%
    melt() %>%
    select(refcode=1,code=2,droi=3)
  dist.pop <- apply(pop1,2,function(x){colMeans((x-pop0)^2)}) %>%
    log %>% scale %>% pnorm %>%
    melt() %>%
    select(refcode=1,code=2,dpop=3)
  
  #筛选股票
  
  obsmap %>%
    merge(
      data.table(
        obs=as.numeric(max(rownames(roi1))),
        data.table(
          dist.roi,
          dpop=as.numeric(dist.pop$dpop)
        ) %>%
          merge(y0)  %>%
          filter(droi<thres_roi,dpop<thres_pop) %>%
          group_by(code) %>%
          summarise(mean=mean(roi),win=mean(roi>0),n=n(),sd=sd(roi)) %>%
          filter(win>0.5,mean>1) 
      ),all.y=T
    ) %>%
    arrange(desc(mean-sd))
  
}

validtest <- function(tests,scale=4,k=2,h=5){
  
  # scale <- 8
  # k <- 2
  # h <- 5
  
  trans <- rbindlist(lapply(tests,function(x){head(x,h)})) %>%
    filter(!is.na(sell))
  
  trans <- data.table(
    trans,
    t(
      sapply(1:nrow(trans),function(i){
        c(
          open=open[paste(trans$buy[i]),paste(trans$code[i])],
          close=close[paste(trans$sell[i]),paste(trans$code[i])]
        )
      })
    )
  ) %>%
    mutate(roi=close/open) %>%
    mutate(roi=(roi-1)/k+1) %>%
    group_by(obs,buy,sell) %>%
    summarise(roi=mean(roi))
  
  obsmap %>%
    filter(obs>=min(trans$obs)) %>%
    merge(
      data.table(
        sell=as.numeric(rownames(close)[-1:-k]),
        bench=rowMeans(close[-1:-k,]/open[-(nrow(open)-0:(k-1)),],na.rm=T)
      ) %>%
        mutate(bench=(bench-1)/k+1)
    ) %>%
    merge(trans,all.x=T,by=c('obs','buy','sell')) %>%
    group_by(obs=substr(obs,1,scale)) %>%
    summarise(miss=mean(is.na(roi)),roi=prod(roi,na.rm=T),bench=prod(bench))
  
}

validtest <- function(tests,scale=4,k=2,h=5,gaokai_rate=1.03){
  
  # scale <- 8
  # k <- 2
  # h <- 5
  # gaokai_rate <- 1.03
  
  trans <- rbindlist(lapply(tests,function(x){head(x,h)})) %>%
    filter(!is.na(sell))
  
  trans <- data.table(
    trans,
    t(
      sapply(1:nrow(trans),function(i){
        if(match(paste(trans$sell[i]),rownames(close))+1 < nrow(close)){
          c(
            close0=close[paste(trans$obs[i]),paste(trans$code[i])],
            open1=open[paste(trans$buy[i]),paste(trans$code[i])],
            close1=close[paste(trans$buy[i]),paste(trans$code[i])],
            low1=low[paste(trans$buy[i]),paste(trans$code[i])],
            low2=low[paste(trans$sell[i]),paste(trans$code[i])],
            high2=high[paste(trans$sell[i]),paste(trans$code[i])],
            close2=close[paste(trans$sell[i]),paste(trans$code[i])],
            open3=open[match(paste(trans$sell[i]),rownames(close))+1,paste(trans$code[i])]
          )
        } else {
          c(
            close0=close[paste(trans$obs[i]),paste(trans$code[i])],
            open1=open[paste(trans$buy[i]),paste(trans$code[i])],
            close1=close[paste(trans$buy[i]),paste(trans$code[i])],
            low1=low[paste(trans$buy[i]),paste(trans$code[i])],
            low2=low[paste(trans$sell[i]),paste(trans$code[i])],
            high2=high[paste(trans$sell[i]),paste(trans$code[i])],
            close2=close[paste(trans$sell[i]),paste(trans$code[i])],
            open3=NA
          )
        }
        
      })
    )
  ) %>%
    mutate(
      gaokai = open1/close0>=gaokai_rate,
      huiluo = low1/close0<=gaokai_rate,
      zhangting = ifelse(floor(code/10000)%in%c(30,68),high2>1.198*close1,high2>1.098*close1),
      dieting = ifelse(floor(code/10000)%in%c(30,68),low2<(1-0.198)*close1,low2<(1-0098)*close1),
    ) %>%
    mutate(
      buyprice = ifelse(!huiluo,NA,ifelse(!gaokai,open1,close0*gaokai_rate)),
      sellprice = ifelse(dieting,open3,
                         ifelse(!zhangting,close2,
                                ifelse(floor(code/10000)%in%c(30,68),1.198*close1,1.098*close1)
                         )),
      roi=ifelse(is.na(sellprice/buyprice),1,sellprice/buyprice)
    ) %>%
    mutate(roi=(roi-1)/k+1) %>%
    group_by(obs,buy,sell) %>%
    summarise(roi=mean(roi))
  
  obsmap %>%
    filter(obs>=min(trans$obs)) %>%
    merge(
      data.table(
        sell=as.numeric(rownames(close)[-1:-k]),
        bench=rowMeans(close[-1:-k,]/open[-(nrow(open)-0:(k-1)),],na.rm=T)
      ) %>%
        mutate(bench=(bench-1)/k+1)
    ) %>%
    merge(trans,all.x=T,by=c('obs','buy','sell')) %>%
    group_by(obs=substr(obs,1,scale)) %>%
    summarise(miss=mean(is.na(roi)),roi=prod(roi,na.rm=T),bench=prod(bench))
  
}

darisk <- function(k=5,ncode=50){
  
  # k <- 5
  # ncode <- 50
  
  roik <- (close[-1:-(k-1),]/open[1:(nrow(open)-k+1),]) %>%
    melt() %>%
    select(obs=1,code=2,roik=3)
  
  valk <- t(
    sapply(k:nrow(val),function(i){
      colSums(val[1:k+i-k,],na.rm=T)
    })
  ) %>%
    melt() %>%
    select(obs=1,code=2,valk=3)
  
  roik <- data.table(roik,valk=valk$valk)
  
  roik <- roik %>%
    merge(
      roik %>%
        arrange(obs,desc(valk)) %>%
        group_by(obs) %>%
        summarise(thres_val=valk[ncode]) %>%
        filter(thres_val>0)
    ) %>%
    filter(valk>=thres_val) %>%
    group_by(obs) %>%
    summarise(risk=mean(roik>1)) 
  
  roik
  
}

################################################################################
#Main 

system.time(tests_ogou <- lapply(60:0,ogou,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05))
validtest(tests_ogou,6,2,5)

system.time(tests_ygou <- lapply(1600:0,ygou))
validtest(tests_ygou,1,2,5) %>% as.data.frame

ygou(1)


ygou(1000)
ogou(iminus=1000,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05)

list(
  ngou(3,3,2,0.5,0.05,0.05) %>% 
    merge(allcode,by='code') %>%
    arrange(desc(mean-sd))
) %>%
  validtest(1,2,5)

tests <- lapply(3:0,ngou,b=9,k=2,thres_valp=0.2,thres_roi=0.05,thres_pop=0.05)
validtest(tests,1,2,5)

tests_old <- lapply(60:0,ogou,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05)
validtest(tests_old,1,2,5)

ogou(iminus=0,b=9,k=2,codek0=15,codek2=10,thres_roi=0.1,thres_pop=0.1)
# 
# system.time(
#   tests <- lapply(1600:0,function(i){
#     ogou(iminus=i,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05)
#   })
# )
# 
# validtest(tests,4,2,5,1.03)
# save(tests,file='result/tests1026_29_2_15_10_005_005.rda')
# rbindlist(tests) %>% write.csv('result/test.csv')
