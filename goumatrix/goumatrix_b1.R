
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(xgboost)
library(caret)
library(parallel)
# use_condaenv(condaenv='/tmp/Rtmp6u7696/rstudio/terminal/python',required=TRUE)
setwd('/home/huzixin/documents/goldgou')

np <- import('numpy')
skl <- import('sklearn')

########################################################################################################################

#Get Historical Data

# system.time(system('Rscript /home/huzixin/documents/goldgou/gethisdata.R'))

hisdata <- fread('hisdata210101.csv')[,-1]
dates <- sort(unique(as.numeric(gsub('-','',unique(hisdata$日期)))))

########################################################################################################################
#骷髅狗
########################################################################################################################

rawhis <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=gsub('-','',date)) %>%
  filter(date%in%dates[length(dates)-0:40]) 
rawhis$did <- match(rawhis$date,sort(unique(rawhis$date)))

dataset <- lapply(1:max(rawhis$did),function(startday){
  keepdays <- 2
  obsdays <- c(2,5,10)
  close <- rawhis %>% acast(did~code,value.var='close')
  open <- rawhis %>% acast(did~code,value.var='open')
  val <- rawhis %>% acast(did~code,value.var='val')
  if(max(obsdays)+startday-1>nrow(close)){
    return(NULL)
  } else {
    vali <- apply(val[1:max(obsdays)+startday-1,],2,rev)
    closei <- apply(close[1:max(obsdays)+startday-1,],2,rev)
    openi <- apply(open[1:max(obsdays)+startday-1,],2,rev)
    pricech <- sapply(obsdays,function(i){(closei[1,]/openi[i,])})
    if(nrow(close)<max(obsdays)+startday+keepdays-1){
      roi <- NA
    } else {
      roi <- data.table(
        code=colnames(close),
        roi=close[max(obsdays)+startday+keepdays-1,]/open[max(obsdays)+startday,]  
      )
    }
    return(list(obs=max(obsdays)+startday-1,code=colnames(close),price=pricech,val=val[1,],roi=cbind(roi)))
  }
})

dataset <- dataset[!sapply(dataset,is.null)]
trainset <- dataset[sapply(dataset,function(x){length(x$roi)})==2]
testset <- dataset[[length(dataset)]]

X <- data.table(code=rownames(testset$price),testset$price,val=testset$val)
X <- X[rowSums(is.na(X))==0,]
X0 <- do.call(rbind,lapply(trainset,function(x){data.table(obs=x$obs,code=rownames(x$price),x$price)}))
Y0 <- do.call(rbind,lapply(trainset,function(x){data.table(obs=x$obs,x$roi)}))
X <- X %>%
  filter(val>=quantile(X$val,0.8,na.rm=T)) %>%
  filter(V1>1.01,V3>1)

Y0 %>%
  merge(
    X0 %>% 
      filter(V1>1.01,V3>1,code%in%X$code) %>%
      group_by(obs) %>%
      summarise(n=n())    
  ) %>%
  group_by(code) %>%
  summarise(mean=sum(roi*n)/sum(n),sd=sd(rep(roi,n)),rate=mean(roi>1),mean2=mean(roi)) %>%
  merge(
    X0 %>%
      group_by(code) %>%
      summarise(benchrate=mean(V1>1),benchroi=prod((V1-1)/2+1))
  ) %>%
  merge(
    unique(hisdata %>% select(code,name))
  ) %>%
  arrange(desc(mean)) %>%
  mutate(obsday=testset$obs,obsdate=sort(unique(rawhis$date))[obsday]) %>%
  filter(mean>1,rate>0.5,rate>benchrate) %>%
  arrange(desc(mean)) %>%
  head() 

########################################################################################################################
#骷髅狗大回测
########################################################################################################################

rawhis <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=gsub('-','',date)) %>%
  filter(date>=20221201)
rawhis$did <- match(rawhis$date,sort(unique(rawhis$date)))

system.time(
  dataset <- lapply(1:max(rawhis$did),function(startday){
    keepdays <- 2
    obsdays <- c(2,5,10)
    close <- rawhis %>% acast(did~code,value.var='close')
    open <- rawhis %>% acast(did~code,value.var='open')
    val <- rawhis %>% acast(did~code,value.var='val')
    if(max(obsdays)+startday-1>nrow(close)){
      return(NULL)
    } else {
      vali <- apply(val[1:max(obsdays)+startday-1,],2,rev)
      closei <- apply(close[1:max(obsdays)+startday-1,],2,rev)
      openi <- apply(open[1:max(obsdays)+startday-1,],2,rev)
      pricech <- sapply(obsdays,function(i){(closei[1,]/openi[i,])})
      if(nrow(close)<max(obsdays)+startday+keepdays-1){
        roi <- NA
      } else {
        roi <- data.table(
          code=colnames(close),
          roi=close[max(obsdays)+startday+keepdays-1,]/open[max(obsdays)+startday,]  
        )
      }
      return(list(obs=max(obsdays)+startday-1,code=colnames(close),price=pricech,val=val[1,],roi=cbind(roi)))
    }
  })
)

dataset <- dataset[!sapply(dataset,is.null)]

#回测

system.time(
  test <- lapply(1:length(dataset),function(i){
    trainset <- dataset[0:29+i]
    if(length(trainset)+i>length(dataset)){
      return(NULL)
    }
    testset <- dataset[[length(trainset)+i]]
    X <- data.table(code=rownames(testset$price),testset$price,val=testset$val)
    Y <- testset$roi
    if(length(Y)==1){
      return(NULL)
    }
    Y <- Y[rowSums(is.na(X))==0,]
    X <- X[rowSums(is.na(X))==0,]
    X0 <- do.call(rbind,lapply(trainset,function(x){data.table(obs=x$obs,code=rownames(x$price),x$price)}))
    Y0 <- do.call(rbind,lapply(trainset,function(x){data.table(obs=x$obs,x$roi)}))
    X <- X %>%
      filter(val>=quantile(X$val,0.8,na.rm=T)) %>%
      filter(V1>1.01,V3>1)
    Y0 %>%
      merge(
        X0 %>% 
          filter(V1>1.01,V3>1,code%in%X$code) %>%
          group_by(obs) %>%
          summarise(n=n())    
      ) %>%
      group_by(code) %>%
      summarise(mean=sum(roi*n)/sum(n),sd=sd(rep(roi,n)),rate=mean(roi>1),mean2=mean(roi)) %>%
      merge(
        X0 %>%
          group_by(code) %>%
          summarise(benchrate=mean(V1>1),benchroi=prod((V1-1)/2+1))
      ) %>%
      merge(
        unique(hisdata %>% select(code,name))
      ) %>%
      merge(Y) %>%
      arrange(desc(mean)) %>%
      mutate(obsday=testset$obs,
             obsdate=sort(unique(rawhis$date))[obsday],
             buydate=sort(unique(rawhis$date))[obsday+1],
             selldate=sort(unique(rawhis$date))[obsday+2])
  })
)

test <- test[!sapply(test,is.null)]
# save(test,file='xunidog.rda')

rlt <- do.call(rbind,lapply(test,function(x){
  x %>%
    filter(mean>1,rate>0.5,rate>benchrate) %>%
    arrange(desc(mean)) %>%
    head(3) %>%
    mutate(name=paste0(name,'(',round(roi,2),')'))
})) %>%
  group_by(obsdate,buydate,selldate) %>%
  summarise(roi=(mean(roi,na.rm=T)-1)/2+1,buy=paste(name,collapse=',')) %>%
  filter(obsdate>=20230101) 

rlt %>%
  group_by(month=substr(buydate,1,6)) %>%
  summarise(roi=prod(roi))

prod(rlt$roi)
plot.ts(cumprod(rlt$roi))
rlt %>% filter(buydate>=20240801) %>% as.data.frame
