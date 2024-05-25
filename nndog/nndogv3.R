
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
use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
setwd('/home/huzixin/documents/goldgou')

########################################################################################################################

# ak <- import("akshare")
# skl <- import('sklearn.linear_model')
# 
# allcodes <- ak$stock_sh_a_spot_em() %>% 
#   rbind(ak$stock_sz_a_spot_em()) %>%
#   select(code=2,name=3)
# 
# #Get His Data
# system.time(
#   hisdata <- do.call(rbind,lapply(allcodes$code,function(codei){
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
# 
# hisdata <- hisdata %>% merge(allcodes)
# write.csv(hisdata,'hisdata210101.csv')

########################################################################################################################

getpool <- function(...){
  do.call(rbind,lapply(list(...),function(x){
    do.call(rbind,lapply(readLines(x) %>% strsplit(','),function(xi){
      data.table(file=x,date=xi[1],code=xi[-1])
    }))
  }))
}

getpool2 <- function(){
  nnlist <- readLines('nnlist.txt')[-1]
  do.call(rbind,lapply(strsplit(nnlist,'\t'),function(x){
    data.table(
      buy=x[1],
      pool=x[2],
      code=strsplit(x[[3]],',')[[1]]
    )
  }))
}

getmodelfile_train <- function(raw,ni=10,nj=3){
  # ni <- 20
  # nj <- 3
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
    yi.close <- Yi[,,'close']
    yi.open <- Yi[1,,'open']
    yi.vol <- Yi[,,'vol']
    yi <- colMeans(yi.close,na.rm=T)/yi.open
    yi[yi.vol[1,]==0] <- NA
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

getmodelfile_test <- function(raw,ni=10,nj=3){
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

getmodelfile_train2 <- function(raw,ni=10,nj=3){
  ni <- max(ni,20)
  rawi <- raw %>%
    filter(variable%in%c('open','close','val','vol','high','low')) %>%
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
    xi.dvol <-  (Xi[1:5,,'vol']/Xi[2:6,,'vol']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('dvol',day)) %>%
      dcast(code~day,value.var='value')
    xi.high <- (Xi[1:5,,'high']/Xi[1:5,,'open']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('high',day)) %>%
      dcast(code~day,value.var='value')
    xi.low <- (Xi[1:5,,'low']/Xi[1:5,,'open']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('low',day)) %>%
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
            rowMeans(Xi[1:5,,'open']/Xi[2:6,,'close']-1,na.rm=T)+1,
            rowSums(Xi[1:4,,'val'])/rowSums(Xi[2:5,,'val']))
    xi <- xi.today %>%
      merge(xi.onite) %>%
      merge(xi.dvol) %>%
      merge(xi.high) %>%
      merge(xi.low) %>%
      merge(xi.others)
    Yi <- rawi[i-nj+1:nj,,]
    yi.close <- Yi[,,'close']
    yi.open <- Yi[1,,'open']
    yi.vol <- Yi[,,'vol']
    yi <- colMeans(yi.close,na.rm=T)/yi.open
    yi[yi.vol[1,]==0] <- NA
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

getmodelfile_test2 <- function(raw,ni=10,nj=3){
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
    xi.dvol <-  (Xi[1:5,,'vol']/Xi[2:6,,'vol']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('dvol',day)) %>%
      dcast(code~day,value.var='value')
    xi.high <- (Xi[1:5,,'high']/Xi[1:5,,'open']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('high',day)) %>%
      dcast(code~day,value.var='value')
    xi.low <- (Xi[1:5,,'low']/Xi[1:5,,'open']) %>%
      melt() %>%
      select(day=1,code=2,value=3) %>%
      mutate(day=paste0('low',day)) %>%
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
            rowMeans(Xi[1:5,,'open']/Xi[2:6,,'close']-1,na.rm=T)+1,
            rowSums(Xi[1:4,,'val'])/rowSums(Xi[2:5,,'val']))
    xi <- xi.today %>%
      merge(xi.onite) %>%
      merge(xi.dvol) %>%
      merge(xi.high) %>%
      merge(xi.low) %>%
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
pool <- getpool2()

#Data Setting

raw <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=gsub('-','',date)) %>%
  filter(date>=dates[length(dates)-250]) %>%
  filter()

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
ob.days <- 20
ob.days2 <- 20

# datasets <- getmodelfile_train2(raw=raw %>% filter(date!=max(raw$date)),ni,nj)
# datasets2 <- getmodelfile_test2(raw=raw %>% filter(date!=max(raw$date)),ni,nj)

system.time(datasets <- getmodelfile_train2(raw=raw,ni,nj))
system.time(datasets2 <- getmodelfile_test2(raw=raw,ni,nj))

#setup dataset
dataseti.train <- datasets[length(datasets)-1:ob.days2+1]
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

data.table(
  buy=unique(Ytr$buy),
  dist=colMeans((t(Zidxtr)-Zidxte[2,])^2),
  sign=colMeans(t(Ztr-1)*(Zte[2,]-1)>0)
) %>% 
  mutate(sel=buy%in%pool) %>%
  mutate(buy=match(buy,unique(Ytr$buy))) %>%
  ggplot() + 
  geom_point(aes(x=buy,y=dist,colour=sel),size=2) +
  geom_line(aes(x=buy,y=dist),linetype=2)

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
                summarise(mean=mean(refroi),median=median(refroi),sd=sd(refroi),prop=mean(refroi>max(mean(Ytr2$roi,na.rm=T),1)))
            }))
  )
)

#return

outi2 <- outi %>% 
  mutate(code=as.numeric(code)) %>%
  merge(hisdata %>% select(code,name) %>% unique() %>% mutate(code=as.numeric(code))) %>%
  filter(obs==max(outi$obs)) %>%
  filter(mean>1) %>%
  arrange(obs,desc(prop),desc(mean))

outi2 <- outi2 %>%
  filter(!code%in%(raw %>% filter(date==outi$obs[1],variable=='open',value<=2))$code) %>%
  head(10)

outi2$prop2 <- (log(nrow(outi2))/log(10)/2+0.5)*outi2$prop/sum(outi2$prop)
print(outi2 %>% select(obs,code,name,prop,prop2))

#Test

##########################################################################################
# Trace
##########################################################################################

#Parameter

ni <- 20
nj <- 2
ob.days <- 20
datasets <- getmodelfile_train2(raw=raw,ni,nj)

#Cubei

system.time(
  test <- lapply(1:(length(datasets)-ob.days-1),function(i){
    print(paste(i,Sys.time()))
    #setup dataset
    dataseti.train <- datasets[1:ob.days+i-1]
    dataseti.obs <- datasets[ob.days+nj+i-1]
    dataseti.train <- lapply(dataseti.train,function(x){
      xi <- log(x[[1]][,-1])
      na2 <- max(xi,na.rm=T)*2
      inf2 <- max(abs(xi)[abs(xi)!=Inf],na.rm=T)*2
      xi[is.na(xi)] <- na2
      xi[-xi==-Inf] <- inf2
      xi[xi==-Inf] <- -inf2
      yi <- x[[2]]
      zi <- x[[3]]
      list(x=xi,y=yi,zi=zi)
    })
    dataseti.obs <- lapply(dataseti.obs,function(x){
      xi <- log(x[[1]][,-1])
      na2 <- max(xi,na.rm=T)*2
      inf2 <- max(abs(xi)[abs(xi)!=Inf],na.rm=T)*2
      xi[is.na(xi)] <- na2
      xi[-xi==-Inf] <- inf2
      xi[xi==-Inf] <- -inf2
      yi <- x[[2]]
      zi <- x[[3]]
      list(x=xi,y=yi,zi=zi)
    })
    #Pooling
    Xtr <- do.call(rbind,lapply(dataseti.train,function(x){x[[1]]}))
    Xte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[1]]}))
    Ytr <- do.call(rbind,lapply(dataseti.train,function(x){x[[2]]}))
    Yte <- do.call(rbind,lapply(dataseti.obs,function(x){x[[2]]}))
    #Jigou and QS
    codei <- unique((pool %>% filter(buy%in%dates[match(unique(Yte$buy),dates)-0:200]))$code)
    seltr <- Ytr$code %in% codei
    selte <- Yte$code %in% codei
    Xtr <- Xtr[seltr,]
    Xte <- Xte[selte,]
    Ytr <- Ytr[seltr,]
    Yte <- Yte[selte,]
    #Scaling
    Xscaler <- apply(Xtr,2,function(x){c(mean=mean(x),sd=sd(x))})
    Xidxtr <- sapply(1:ncol(Xtr),function(i){(Xtr[,i]-Xscaler[1,i])/Xscaler[2,i]})
    Xidxte <- sapply(1:ncol(Xte),function(i){(Xte[,i]-Xscaler[1,i])/Xscaler[2,i]})
    # Xtr <- Xtr %>% t
    # Xte <- Xte %>% t
    Xtr <- Xidxtr %>% t
    Xte <- Xidxte %>% t
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
                    summarise(mean=mean(refroi),median=median(refroi),sd=sd(refroi),prop=mean(refroi>max(mean(Ytr$roi,na.rm=T),1)))
                }))
      )
    )
    #return
    testi <- outi %>% filter(prop>0.8,mean>1)
    print(
      data.table(
        buy=outi$buy[1],
        sell=outi$sell[1],
        n=nrow(testi),
        roi=mean(testi$roi),
        ref=mean(outi$roi,na.rm=T)
      ) %>%
        mutate(roi=ifelse(is.na(roi),1,roi))
    )
    return(outi)
  })
)
save(test,file='test.rda')

#Resulting

rlt <- do.call(
  rbind,
  lapply(test,function(x){
    xi <- x %>% 
      mutate(sel=((prop>0.8)&
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
        sel=xi.buyroi,
        ref=mean(roi,na.rm=T))
  })
) %>%
  mutate(sel=ifelse(n==0,1,sel)) %>%
  mutate(sel=ifelse(is.na(sel),1,sel)) %>%
  mutate(sel=(sel-1)/2+1,ref=(ref-1)/2+1)

data.table(
  rlt %>% select(buy,sell,n),
  jingou=cumprod(rlt$sel),
  ref=cumprod(rlt$ref),
  djingou=rlt$sel,
  dref=rlt$ref) 

data.table(
  id=1:nrow(rlt),
  rlt %>% select(buy,sell,n),
  jingou=cumprod(rlt$sel),
  actual=cumprod(rlt$ref)) %>%
  melt(id=1:4) %>%
  ggplot() + 
  geom_line(aes(x=id,y=value,colour=variable))

############################################################
#操作回测
############################################################

#获取日维度数据

load('test.rda')
test <- do.call(rbind,test) %>% mutate(code=as.numeric(code))

back.stocks.data2 <- hisdata %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,shift=涨跌幅) %>%
  mutate(date=gsub('-','',date)) %>%
  mutate(code=as.numeric(code)) %>%
  filter(date>=dates[length(dates)-250])

back.date <- back.stocks.data2$date %>% unique %>% sort

#筛选股票

test2 <- test %>%
  filter(mean>1,prop>=0.8)

test3 <- test2 %>%
  merge(
    test2 %>%
      group_by(buy) %>%
      summarise(threshold=sort(prop,decreasing=T)[3]) %>%
      mutate(threshold=if_else(is.na(threshold),0,threshold))
  ) %>%
  filter(prop>=threshold)

test3 %>% group_by(buy) %>%
  summarise(count=n())

backtest1 <- test3 %>%
  mutate(forecast=buy)

backtest1 <- do.call(rbind,lapply(1:length(unique(backtest1$forecast)),function(i){
  # print(i)
  forecasti <- unique(backtest1$forecast)[i]
  stocki <- backtest1 %>% filter(forecast==forecasti)
  numi <- nrow(stocki)
  backtest1i <- stocki %>%
    mutate(share=1/numi)
}))


#测算

inner <- 0.03
backtest2 <- do.call(rbind,lapply(2:(length(back.date)-1),function(i){
  # print(i)
  forecasti <- back.date[i]
  stocki <- backtest1 %>% filter(forecast==forecasti)
  todayi <- forecasti
  afteri <- back.date[which(back.date==forecasti)+1]
  after2i <- back.date[which(back.date==forecasti)+2]
  beforei <- back.date[which(back.date==forecasti)-1]
  datai_today <- back.stocks.data2 %>% filter(date==todayi)
  datai_afteri <- back.stocks.data2 %>% filter(date==afteri)
  datai_after2i <- back.stocks.data2 %>% filter(date==after2i)
  datai_beforei <- back.stocks.data2 %>% filter(date==beforei)
  stock2i <- stocki %>%
    merge(datai_today,by='code') %>%
    merge(datai_beforei,by='code') %>%
    mutate(open.shift=(open.x-close.y)/close.y,
           low.shift=(low.x-close.y)/close.y,
           inner.point=floor(close.y*(1+inner)*100)/100,
           one.zt.flag=ifelse((high.x==low.x)&(shift.x>0),1,0),
           open.buy.flag=ifelse(open.shift<inner,1,0),
           inner.buy.flag=ifelse((open.shift>=inner)&(low.shift<inner),1,0),
           buyprice=ifelse(one.zt.flag==1,
                           NA,
                           ifelse(open.buy.flag==1,
                                  open.x,
                                  ifelse(inner.buy.flag==1,
                                         inner.point,
                                         NA)))) %>%
    filter(!is.na(buyprice)) %>%
    select(code,forecast,share,buyprice,benchmark=close.x)
  stock3i <- stock2i %>%
    merge(datai_afteri,by='code') %>%
    merge(datai_after2i,by='code') %>%
    mutate(limit.flag=ifelse(substr(code,1,2) %in% c('68','30'),1,2),
           high.shift=(high.x-benchmark)/benchmark,
           one.dt.flag=ifelse((high.x==low.x)&(shift.x<0),1,0),
           sellprice=ifelse(one.dt.flag==1,
                            close.y,
                            ifelse((high.shift>=0.098)&(limit.flag=2),
                                   high.x,
                                   ifelse((high.shift>=0.198)&(limit.flag=1),
                                          high.x,
                                          close.x)))) %>%
    select(code,forecast,share,buyprice,sellprice) %>%
    mutate(profit=(sellprice-buyprice)*share/buyprice)
  data.frame(date=forecasti,
             profit=sum(stock3i$profit)/2)
})) %>% mutate(accu.profit=cumprod(1+profit),profit=profit+1)


ref <- test %>% group_by(date=buy) %>% summarise(ref=sum(roi-1,na.rm=T)/n()+1)
ref$accu.ref <- cumprod(ref$ref)
rlt <- backtest2 %>% merge(ref)
