
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(xgboost)
library(caret)
library(parallel)
use_condaenv(condaenv='/Users/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
# use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
ak <- import("akshare")
skl <- import('sklearn.linear_model')
setwd('/Users/huzixin/Documents/jingou')

# allcodes <- unlist(
#   ak$stock_sh_a_spot_em() %>% select(code=代码) %>%
#     rbind(ak$stock_sz_a_spot_em() %>% select(code=代码))
# )

# #Get His Data
# datei <- dates[length(dates)]
# system.time(
#   hisdata <- do.call(rbind,lapply(allcodes,function(codei){
#     print(codei)
#     Sys.sleep(0.01)
#     x <- ak$stock_zh_a_hist(symbol=codei,
#                             period='daily',
#                             start_date=20210101,
#                             end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
#       mutate(code=codei)
#     x[[1]] <- sapply(x[[1]],paste)
#     x
#   }))
# )
# write.csv(hisdata,'hisdata210101.csv')

########################################################################################################################

getpool <- function(...){
  do.call(rbind,lapply(list(...),function(x){
    do.call(rbind,lapply(readLines(x) %>% strsplit(','),function(xi){
      data.table(file=x,date=xi[1],code=xi[-1])
    }))
  }))
}

getmodelfile <- function(raw,ni=10,nj=3){
  ni <- max(ni,20)
  rawi <- raw %>%
    filter(variable%in%c('open','close','val','vol')) %>%
    acast(date~code~variable,value.var='value')
  datasets <- lapply((ni+nj):nrow(rawi),function(i){
    Xi <- rawi[ni:1+i-ni-nj,,]
    rownames(Xi) <- 1:nrow(Xi)
    xi.today <- (Xi[1:5,,'close']/Xi[1:5,,'open']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('today',day)) %>%
      dcast(code~day,value.var='value')
    xi.onite <- (Xi[1:5,,'open']/Xi[2:6,,'close']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('onite',day)) %>%
      dcast(code~day,value.var='value')
    xi.others <- data.table(
      code=colnames(Xi),
      dclose05mat = Xi[1,,'close']/colMeans(Xi[1:5,,'close']),
      dvol05mat = Xi[1,,'vol']/colMeans(Xi[1:5,,'vol']),
      dclose10mat = Xi[1,,'close']/colMeans(Xi[1:10,,'close']),
      dvol10mat = Xi[1,,'vol']/colMeans(Xi[1:10,,'vol']),
      dclose20mat = Xi[1,,'close']/colMeans(Xi[1:20,,'close']),
      dvol20mat = Xi[1,,'vol']/colMeans(Xi[1:20,,'vol']),
      dclose05 = Xi[1,,'close']/Xi[5,,'close'],
      dclose10 = Xi[1,,'close']/Xi[10,,'close']
    )
    zi <- c(rowMeans(Xi[1:5,,'close']/Xi[1:5,,'open']-1,na.rm=T)+1,
            rowSums(Xi[1:4,,'val'])/rowSums(Xi[2:5,,'val']))
    xi <- xi.today %>%
      merge(xi.onite) %>%
      merge(xi.others)
    Yi <- rawi[i-nj+1:nj,,]
    yi <- colMeans(Yi[,,'close'],na.rm=T)/Yi[1,,'open']
    yi <- data.table(
      code=colnames(Yi),
      buy=rownames(Yi)[1],
      sell=rownames(Yi)[nrow(Yi)],
      roi=yi
    )
    list(xi,yi,zi)
  })
  datasets
}

getmodelfile2 <- function(raw,ni=10,nj=3){
  ni <- max(ni,20)
  rawi <- raw %>%
    filter(variable%in%c('open','close','high','low','volp','vol','val')) %>%
    acast(date~code~variable,value.var='value')
  datasets <- lapply((nrow(rawi)-ni-nj)+1:nj,function(i){
    Xi <- rawi[ni:1+i,,]
    yi <- data.table(
      code=colnames(Xi),
      obs=rownames(Xi)[1]
    )
    rownames(Xi) <- 1:nrow(Xi)
    xi.today <- (Xi[1:5,,'close']/Xi[1:5,,'open']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('today',day)) %>%
      dcast(code~day,value.var='value')
    xi.onite <- (Xi[1:5,,'open']/Xi[2:6,,'close']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('onite',day)) %>%
      dcast(code~day,value.var='value')
    xi.others <- data.table(
      code=colnames(Xi),
      dclose05mat = Xi[1,,'close']/colMeans(Xi[1:5,,'close']),
      dvol05mat = Xi[1,,'vol']/colMeans(Xi[1:5,,'vol']),
      dclose10mat = Xi[1,,'close']/colMeans(Xi[1:10,,'close']),
      dvol10mat = Xi[1,,'vol']/colMeans(Xi[1:10,,'vol']),
      dclose20mat = Xi[1,,'close']/colMeans(Xi[1:20,,'close']),
      dvol20mat = Xi[1,,'vol']/colMeans(Xi[1:20,,'vol']),
      dclose05 = Xi[1,,'close']/Xi[5,,'close'],
      dclose10 = Xi[1,,'close']/Xi[10,,'close']
    )
    zi <- c(rowMeans(Xi[1:5,,'close']/Xi[1:5,,'open']-1,na.rm=T)+1,
            rowSums(Xi[1:4,,'val'])/rowSums(Xi[2:5,,'val']))
    xi <- xi.today %>%
      merge(xi.onite) %>%
      merge(xi.others)
    list(xi,yi,zi)
  })
  names(datasets) <- rownames(rawi)[nrow(rawi)-(nj-1):0]
  datasets
}

######################################################################
# Cubes
######################################################################

hisdata <- read.csv('hisdata210101.csv')[,-1] 
# hisdata <- hisdata %>% filter(!`日期`%in%c('2024-04-11','2024-04-10','2024-04-09'))
dates <- sort(unique(as.numeric(gsub('-','',unique(hisdata$日期)))))
datei <- as.numeric(gsub('-','',max(hisdata$日期)))

pool <- getpool('jgp.txt','qs.txt')

#Data Setting

raw <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=gsub('-','',date)) %>%
  filter(date>=dates[length(dates)-250])

raw <- raw %>%
  filter(code%in%(pool %>% filter(date>=min(raw$date)))$code)

test <- melt(raw,id=1:2) %>%
  mutate(value=as.numeric(value)) %>%
  acast(date~code~variable,value.var='value') 

for(i in 2:nrow(test)){
  test[i,,'close'] <- ifelse(is.na(test[i,,'close']),test[i-1,,'close'],test[i,,'close'])
  test[i,,'open'] <- ifelse(is.na(test[i,,'open']),test[i,,'close'],test[i,,'open'])
  test[i,,'high'] <- ifelse(is.na(test[i,,'high']),test[i,,'close'],test[i,,'high'])
  test[i,,'low'] <- ifelse(is.na(test[i,,'low']),test[i,,'close'],test[i,,'low'])
  test[i,,'val'] <- ifelse(is.na(test[i,,'val']),0,test[i,,'val'])
  test[i,,'vol'] <- ifelse(is.na(test[i,,'vol']),0,test[i,,'vol'])
}
raw <- test %>%
  melt() %>%
  select(date=1,code=2,variable=3,value=4) 

raw <- raw %>%
  rbind(
    raw %>%
      filter(variable%in%c('val','vol')) %>%
      merge(
        raw %>%
          group_by(date,variable) %>%
          summarise(idx=sum(value,na.rm=T))
      ) %>%
      mutate(value=value/(idx)) %>%
      mutate(variable=paste0(variable,'p')) %>%
      select(date,code,variable,value)
  ) %>%
  mutate(value=ifelse((variable%in%c('volp','valp'))&(is.na(value)),0,value))

##########################################################################################
# NNdog2
##########################################################################################

ni <- 20
nj <- 2
ob.days <- 40

datasets <- getmodelfile(raw=raw %>% filter(date!=max(raw$date)),ni,nj)
datasets2 <- getmodelfile2(raw=raw %>% filter(date!=max(raw$date)),ni,nj)

# datasets <- getmodelfile(raw=raw,ni,nj)
# datasets2 <- getmodelfile2(raw=raw,ni,nj)

#setup dataset
dataseti.train <- datasets[length(datasets)-1:ob.days+1]
dataseti.obs <- datasets2

dataseti.train <- lapply(dataseti.train,function(x){
  xi <- log(x[[1]][,-1])
  na2 <- max(xi,na.rm=T)*2
  inf2 <- min(xi[xi!=-Inf],na.rm=T)*2
  xi[xi==-Inf] <- inf2
  xi[is.na(xi)] <- na2
  yi <- x[[2]]
  zi <- x[[3]]
  list(x=xi,y=yi,zi=zi)
})

dataseti.obs <- lapply(dataseti.obs,function(x){
  xi <- log(x[[1]][,-1])
  na2 <- max(xi,na.rm=T)*2
  inf2 <- min(xi[xi!=-Inf],na.rm=T)*2
  xi[xi==-Inf] <- inf2
  xi[is.na(xi)] <- na2
  yi <- x[[2]]
  zi <- x[[3]]
  list(x=xi,y=yi,zi=zi)
})

#Pooling

Xtr <- do.call(rbind,lapply(dataseti.train,function(x){x[[1]]}))
Xte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[1]]}))
Xscaler <- apply(Xtr,2,function(x){c(mean=mean(x),sd=sd(x))})
Xidxtr <- sapply(1:ncol(Xtr),function(i){(Xtr[,i]-Xscaler[1,i])/Xscaler[2,i]})
Xidxte <- sapply(1:ncol(Xte),function(i){(Xte[,i]-Xscaler[1,i])/Xscaler[2,i]})

Ytr <- do.call(rbind,lapply(dataseti.train,function(x){x[[2]]}))
Yte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[2]]}))

Ztr <- do.call(rbind,lapply(dataseti.train,function(x){x[[3]]}))
Zte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[3]]}))
Zscaler <- apply(Ztr,2,function(x){c(mean=mean(x),sd=sd(x))})
Zidxtr <- sapply(1:ncol(Ztr),function(i){(Ztr[,i]-Zscaler[1,i])/Zscaler[2,i]})
Zidxte <- sapply(1:ncol(Zte),function(i){(Zte[,i]-Zscaler[1,i])/Zscaler[2,i]})

pool <- (data.table(
  buy=unique(Ytr$buy),
  dist=colMeans((t(Zidxtr)-Zidxte[2,])^2),
  sign=colMeans(t(Ztr-1)*(Zte[2,]-1)>0)
) %>%
    arrange(dist) %>%
    head(20))$buy

#Train and Test set

dataseti.train2 <- dataseti.train[which(sapply(dataseti.train,function(x){x[[2]]$buy[1]})%in%pool)]
Xtr2 <- do.call(rbind,lapply(dataseti.train2,function(x){x[[1]]})) %>% t
Ytr2 <- do.call(rbind,lapply(dataseti.train2,function(x){x[[2]]}))
Xte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[1]]})) %>% t
Yte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[2]]}))

#KNN

system.time(
  outi <- data.table(
    Yte,
    do.call(rbind,
            apply(Xte,2,function(xi){
              xdist <- colMeans((xi-Xtr2)^2)
              data.table(dist=xdist,refroi=Ytr$roi) %>%
                arrange(dist) %>%
                head(30) %>%
                summarise(mean=mean(refroi),median=median(refroi),sd=sd(refroi),prop=mean(refroi>1))
            }))
  )
)

#return

(outi2 <- outi %>% 
    filter(obs==max(outi$obs)) %>%
    mutate(sel=((prop>=0.8)&(mean>1))) %>%
    filter(sel) %>%
    arrange(obs,desc(prop),desc(mean)) %>%
    head(10))

outi2 <- outi2 %>%
  filter(!code%in%(raw %>% filter(date==outi$obs[1],variable=='open',value<=2))$code)

outi2$prop2 <- (log(nrow(outi2))/log(10)/2+0.5)*outi2$prop/sum(outi2$prop)
outi2 %>% select(obs,code,prop,prop2)

outi2 <- outi %>% 
    filter(obs==max(outi$obs)) %>%
    filter(mean>1) %>%
    arrange(obs,desc(prop),desc(mean))

outi2 <- outi2 %>%
  filter(!code%in%(raw %>% filter(date==outi$obs[1],variable=='open',value<=2))$code) %>%
  head(10)

outi2$prop2 <- (log(nrow(outi2))/log(10)/2+0.5)*outi2$prop/sum(outi2$prop)
outi2 %>% select(obs,code,prop,prop2)

##########################################################################################
# NNdog
##########################################################################################

ni <- 20
nj <- 2
ob.days <- 20

datasets <- getmodelfile(raw=raw %>% filter(date!=max(raw$date)),ni,nj)
datasets2 <- getmodelfile2(raw=raw %>% filter(date!=max(raw$date)),ni,nj)

# datasets <- getmodelfile(raw=raw,ni,nj)
# datasets2 <- getmodelfile2(raw=raw,ni,nj)

#setup dataset
dataseti.train <- datasets[length(datasets)-1:ob.days+1]
dataseti.obs <- datasets2

#Train and Test set
Xtr <- do.call(rbind,lapply(dataseti.train,function(x){x[[1]]}))[,-1]
Ytr <- do.call(rbind,lapply(dataseti.train,function(x){x[[2]]}))
Xtr <- Xtr %>% log %>% as.matrix %>% t
na2 <- max(Xtr,na.rm=T)*2
inf2 <- min(Xtr[Xtr!=-Inf],na.rm=T)*2
Xtr[Xtr==-Inf] <- inf2
Xtr[is.na(Xtr)] <- na2
sel <- which((is.na(Ytr$roi))==0)
Xtr <- Xtr[,sel]
Ytr <- Ytr[sel,]
Xte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[1]]}))[,-1]
Yte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[2]]}))
Xte <- Xte %>% log %>% as.matrix %>% t
Xte[Xte==-Inf] <- inf2
Xte[is.na(Xte)] <- na2

#KNN
system.time(
  outi <- data.table(
    Yte,
    do.call(rbind,
            apply(Xte,2,function(xi){
              xdist <- colMeans((xi-Xtr)^2)
              data.table(dist=xdist,refroi=Ytr$roi) %>%
                arrange(dist) %>%
                head(30) %>%
                summarise(mean=mean(refroi),median=median(refroi),sd=sd(refroi),prop=mean(refroi>1))
            }))
  )
)

#return

(outi2 <- outi %>% 
    filter(obs==max(outi$obs)) %>%
    mutate(sel=((prop>=0.8)&(mean>1))) %>%
    filter(sel) %>%
    arrange(obs,desc(prop),desc(mean)) %>%
    head(10))

outi2 <- outi2 %>%
  filter(!code%in%(raw %>% filter(date==outi$obs[1],variable=='open',value<=2))$code)

outi2$prop2 <- (log(nrow(outi2))/log(10)/2+0.5)*outi2$prop/sum(outi2$prop)
outi2 %>% select(obs,code,prop,prop2)

outi2 <- outi %>%
  filter(obs==max(outi$obs)) %>%
  filter(mean>1) %>%
  arrange(obs,desc(prop),desc(mean))

outi2 <- outi2 %>%
  filter(!code%in%(raw %>% filter(date==outi$obs[1],variable=='open',value<=2))$code) %>%
  head(10)

outi2$prop2 <- (log(nrow(outi2))/log(10)/2+0.5)*outi2$prop/sum(outi2$prop)
outi2 %>% select(obs,code,prop,prop2)

##########################################################################################
# Trace
##########################################################################################

#Parameter

ni <- 20
nj <- 2
ob.days <- 20
datasets <- getmodelfile(raw=raw,ni,nj)

#Cubei

system.time(
  test <- lapply(1:(length(datasets)-ni-2),function(i){
    print(paste(i,Sys.time()))
    #setup dataset
    dataseti.train <- datasets[1:ob.days+i-1]
    dataseti.obs <- datasets[ob.days+nj+i-1]
    #Train and Test set
    Xtr <- do.call(rbind,lapply(dataseti.train,function(x){x[[1]]}))[,-1]
    Ytr <- do.call(rbind,lapply(dataseti.train,function(x){x[[2]]}))
    Xtr <- Xtr %>% log %>% as.matrix %>% t
    na2 <- max(Xtr,na.rm=T)*2
    inf2 <- min(Xtr[Xtr!=-Inf],na.rm=T)*2
    Xtr[Xtr==-Inf] <- inf2
    Xtr[is.na(Xtr)] <- na2
    sel <- which((is.na(Ytr$roi))==0)
    Xtr <- Xtr[,sel]
    Ytr <- Ytr[sel,]
    Xte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[1]]}))[,-1]
    Yte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[2]]}))
    Xte <- Xte %>% log %>% as.matrix %>% t
    Xte[Xte==-Inf] <- inf2
    Xte[is.na(Xte)] <- na2
    #KNN
    system.time(
      outi <- data.table(
        Yte,
        do.call(rbind,
                apply(Xte,2,function(xi){
                  xdist <- colMeans((xi-Xtr)^2)
                  data.table(dist=xdist,refroi=Ytr$roi) %>%
                    arrange(dist) %>%
                    head(30) %>%
                    summarise(mean=mean(refroi),sd=sd(refroi),prop=mean(refroi>1))
                }))
      )
    )
    #return
    testi <- outi %>% filter(prop>=0.8,mean>1)
    print(
      data.table(
        buy=outi$buy[1],
        sell=outi$sell[1],
        n=nrow(testi),
        roi=mean(testi$roi),
        ref=mean(outi$roi,na.rm=T)
      )
    )
    return(outi)
  })
)

#Resulting

pool2 <- pool %>%
  select(buy=date,code,file) %>%
  mutate(value=1) %>%
  dcast(buy+code~file,value.var='value')
  
rlt <- do.call(
  rbind,
  lapply(test,function(x){
    xi <- x %>% 
      merge(pool2,all.x=T) %>%
      mutate(sel=((prop>=0.8)&
                    (mean>1)&
                    (!code%in%(raw %>% filter(date==x$buy[1],variable=='open',value<=2))$code))) %>%
      arrange(desc(sel),desc(prop),desc(mean))
    xi$sel[-1:-10] <- F
    xi.buy <- xi %>% filter(sel)
    xi.buyroi <- sum(
      xi.buy$prop/sum(xi.buy$prop)*(log(sum(xi$sel))/log(10)/2+0.5)*xi.buy$roi,
      1*(0.5-log(sum(xi$sel))/log(10)/2)
    )
    xi <- xi %>% 
      group_by(buy,sell) %>%
      summarise(
        n=sum(sel),
        # sel=mean(roi[sel],na.rm=T),
        sel=xi.buyroi,
        ref=mean(roi,na.rm=T),
        jg=mean(roi[!is.na(jgp.txt)]),
        qs=mean(roi[!is.na(qs.txt)]))
  })
) %>%
  mutate(sel=ifelse(n==0,1,sel)) %>%
  mutate(sel=ifelse(is.na(sel),1,sel)) %>%
  mutate(jg=ifelse(is.na(jg),1,jg)) %>%
  mutate(qs=ifelse(is.na(qs),1,qs)) %>%
  mutate(sel=(sel-1)/2+1,jg=(jg-1)/2+1,qs=(qs-1)/2+1,ref=(ref-1)/2+1)

data.table(
  rlt %>% select(buy,sell,n),
  jingou=cumprod(rlt$sel),
  actual=cumprod(rlt$ref),
  jg=cumprod(rlt$jg),
  qs=cumprod(rlt$qs))

data.table(
  id=1:nrow(rlt),
  rlt %>% select(buy,sell,n),
  jingou=cumprod(rlt$sel),
  actual=cumprod(rlt$ref),
  jg=cumprod(rlt$jg),
  qs=cumprod(rlt$qs)) %>%
  melt(id=1:4) %>%
  ggplot() + 
  geom_line(aes(x=id,y=value,colour=variable))

##########################################################################################

load("/Users/huzixin/Documents/jingou/rlt_nndog.rda")
out <- do.call(rbind,lapply(test,function(x){
  x <- x %>% arrange(desc(prop)) %>% mutate(roi=ifelse(is.na(roi),1,roi))
  out <- cumsum(x$roi)/(1:nrow(x))
  out[c((1:5)*10,length(out))]
}))
apply(out,2,prod)
