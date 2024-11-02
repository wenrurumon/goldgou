
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

jgp <- jgp %>% filter(code%in%allcode$code)

ddata <- fread('data/hisdata.csv')[,-1] %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=as.numeric(gsub('-','',date))) 

code60 <- ddata %>%
  filter(date<=rev(sort(unique(ddata$date)))[60]) %>%
  group_by(code) %>%
  summarise(n=n_distinct(date))
ddata <- ddata %>% filter(code%in%code60$code)

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

close[close<0] <- NA
open[open<0] <- NA

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

jgou <- function(iminus,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05){
  # iminus <- 32
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
  jgpi0 <- jgp %>% filter(date>startdid0,date<=enddid0,code%in%colnames(close))
  codei0 <- unique(jgpi0$code) %>% paste
  roi0 <- roi[i-(b:0+k),codei0]
  pop0 <- pop[i-(b:0+k),codei0]
  roi1 <- roi[i-(b:0),codei0]
  startdid2 <- datemap$date[which(datemap$date==rownames(roi)[i])-codek2]
  enddid2 <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi2 <- jgp %>% filter(date>startdid2,date<=enddid2,code%in%colnames(close))
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

ygou <- function(iminus,k=2){
  
  # iminus <- 1
  # k <- 2
  
  obs <- nrow(close)-iminus
  buy <- obs+1
  sell <- obs+k
  
  yg0 <- data.table(
    close[obs,] > open[obs,],
    val[obs,] > val[obs-1,],
    high[obs,]/close[obs-1,]>=ifelse(floor(as.numeric(colnames(close))/10000)%in%c(30,68),1.198,1.098)
  )
  
  yg1 <- data.table(
    lower = ifelse((close[obs-1,]+close[obs,])/2>close[obs,]*0.95,close[obs,]*0.95,(close[obs-1,]+close[obs,])/2),
    upper = close[obs,]*1.03
  )
  
  data.table(
    code=as.numeric(colnames(close)),
    obs=as.numeric(rownames(close)[obs]),
    yg0=rowMeans(yg0)==1,
    yg1
  ) %>%
    filter(yg0) %>%
    select(code,obs,lower,upper)

}

valid <- function(tests,k=2,h=5,gaokai_rate=1.03){
  
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
    summarise(roi=mean(roi),code=paste(code,collapse=','))
  
  trans
  
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

risk <- darisk(2,50)
print(ifelse(tail(risk$risk,1)>=0.6,'JGou','YGou'))

list(
  strategy=ifelse(tail(risk$risk,1)>=0.6,'JGou','YGou'),
  jingou=jgou(0,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05) %>% merge(allcode) %>% arrange(desc(mean)),
  yingou=ygou(0,k=2) %>% merge(allcode) %>% filter(!grepl('ST|退',name))
)

# system.time(tests_jg <- lapply(1600:0,function(i){
#   print(paste(i,Sys.time()))
#   jgou(i,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05)
# }))

# system.time(
#   tests_yg <- lapply(1600:3,function(iminus){
#     testi <- ygou(iminus) %>% merge(allcode) %>% merge(obsmap,by='obs')
#     testi$open1 <- open[unique(paste(testi$buy)),paste(testi$code)]
#     testi <- testi %>%
#       filter(!grepl('ST|退',name)) %>%
#       filter(open1>=lower,open1<=upper)
#     testi %>% select(obs,buy,sell,code) %>% mutate(mean=1.1,win=1)
#   })
# )

# save(tests_jg,tests_yg,file='result/test1101_gou.rda')
load('result/test1101_gou.rda')

################################################################################

#Yingou

bench <- obsmap %>%
  merge(
    melt(close[-1,]/open[-nrow(open),]) %>%
      select(sell=1,code=2,bench=3) %>%
      mutate(bench=ifelse(is.na(bench),1,bench))    
  ) %>%
  merge(
    val %>%
      melt() %>%
      select(obs=1,code=2,val=3) %>%
      merge(
        val %>%
          melt() %>%
          select(obs=1,code=2,val=3) %>%
          group_by(obs) %>%
          summarise(valp=quantile(val,1-100/ncol(close),na.rm=T))
      ) %>%
      filter(val>=valp),
    by=c('obs','code')
  ) %>%
  group_by(obs) %>%
  summarise(bench=mean((bench-1)/2+1))

test <- obsmap %>%
  filter(floor(obs/10000)>=2019) %>%
  merge(
    risk,by='obs'
  ) %>%
  merge(
    valid(tests_jg) %>% as.data.frame %>% select(obs,jg=roi),all.x=T
  ) %>%
  merge(
    valid(tests_yg,h=Inf) %>% as.data.frame %>% select(obs,yg=roi),all.x=T
  ) %>%
  merge(bench) %>%
  mutate(roi=ifelse(risk>=0.6,yg,jg)) 

test %>%
  group_by(obs=floor(obs/10000)) %>%
  summarise(miss=mean(is.na(roi)),JinGou=prod(jg,na.rm=T),YinGou=prod(yg,na.rm=T),DualGou=prod(roi,na.rm=T),bench=prod(bench,na.rm=T))

data.table(test[,1:4],apply(test[,-1:-4],2,function(x){
  cumprod(ifelse(is.na(x),1,x))
})) %>%
  select(obs,buy,sell,risk,JinGou=jg,YinGou=yg,DualGou=roi,Benchmark=bench) %>%
  melt(id=1:4) %>%
  mutate(obs=as.Date(strptime(obs,format='%Y%m%d'))) %>%
  ggplot() + 
  geom_line(aes(x=obs,y=value,colour=variable)) +
  theme_bw()
  
