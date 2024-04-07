
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
# system('scp huzixin@10.88.1.14:/home/huzixin/documents/goldgou/qs.txt .')

########################################################################################################################

getpool <- function(...){
  do.call(rbind,lapply(list(...),function(x){
    do.call(rbind,lapply(readLines(x) %>% strsplit(','),function(xi){
      data.table(file=x,date=xi[1],code=xi[-1])
    }))
  }))
}

gethis <- function(codes,datei,h=10000){
  hisdata <- do.call(rbind,lapply(codes,function(codei){
    Sys.sleep(0.01)
    x <- ak$stock_zh_a_hist(symbol=codei,
                            period='daily',
                            start_date=as.numeric(datei)-h,
                            end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
      mutate(code=codei)
    x[[1]] <- sapply(x[[1]],paste)
    x
  }))
  raw <- hisdata %>% 
    select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
    mutate(date=gsub('-','',date)) %>%
    filter(date<datei)
  test <- melt(raw,id=1:2) %>%
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
        mutate(value=value/idx) %>%
        mutate(variable=paste0(variable,'p')) %>%
        select(date,code,variable,value)
    ) %>%
    mutate(value=ifelse((variable%in%c('volp','valp'))&(is.na(value)),0,value))
  raw
}

getmodelfile <- function(raw,ni=20,nj=10){
  rawi <- raw %>%
    filter(variable%in%c('open','close','volp','vol')) %>%
    acast(date~code~variable,value.var='value')
  datasets <- lapply((ni+nj):nrow(rawi),function(i){
      Xi <- rawi[ni:1+i-ni-nj,,]
      xi <- cbind(
        Xi[1,,]/(Xi[2,,]+1),
        t(Xi[1:5,,'open']/(Xi[1:5,,'close']+1)),
        Xi[1,,]/(apply(Xi[2:6,,],2:3,mean,na.rm=T)+1),
        Xi[1,,]/(apply(Xi[2:20,,],2:3,mean,na.rm=T)+1)  
      )
      Yi <- rawi[i-nj+1:nj,,]
      yi <- colMeans(Yi[,,'close'],na.rm=T)/Yi[1,,'open']
      list(xi,yi)
  })
  Xi <- rawi[nrow(rawi)-(1:ni-1),,]
  Xtes <- cbind(
      Xi[1,,]/(Xi[2,,]+1),
      t(Xi[1:5,,'open']/(Xi[1:5,,'close']+1)),
      Xi[1,,]/(apply(Xi[2:6,,],2:3,mean,na.rm=T)+1),
      Xi[1,,]/(apply(Xi[2:20,,],2:3,mean,na.rm=T)+1)  
  )
  Xs <- do.call(rbind,lapply(datasets,function(x){x[[1]]}))
  Ys <- do.call(c,lapply(datasets,function(x){x[[2]]}))
  codes <- unique(rownames(Xs))
  datasets <- lapply(codes,function(codei){
    Xtr <- Xs[rownames(Xs)==codei,]
    Ytr <- Ys[names(Ys)==codei]
    Ytr <- Ytr[rowMeans(is.na(Xtr))==0]
    Xtr <- Xtr[rowMeans(is.na(Xtr))==0,,drop=F]
    Xte <- Xtes[rownames(Xtes)==codei,,drop=F]
    list(code=codei,Xtr=Xtr,Ytr=Ytr,Xte=Xte)
  })
}

modeli_caret <- function(dataseti){
  # print(paste(dataseti$code,Sys.time()))
  Xtr <- dataseti$Xtr
  Xte <- dataseti$Xte
  Ytr <- dataseti$Ytr
  if(nrow(Xtr)<50){
    return(data.table(code=dataseti$code,sample=1,roi=NA))
  }
  weights <- log(1:nrow(Xtr))
  # weights <- rep(1,nrow(Xtr))
  dtrain <- xgb.DMatrix(data=Xtr,label=Ytr,weight=weights)
  dtest <- xgb.DMatrix(data=Xte)
  fitControl <- trainControl(method = "repeatedcv", 
                             number = 5,
                             verboseIter = FALSE,
                             search = "random" 
  )
  df <- data.frame(y=Ytr,Xtr)
  system.time(
    pred <- sapply(1:30,function(i){
      # print(paste(i,Sys.time()))
      set.seed(i+100)
      caret_xgb <- train(y~.,
                         data=df,
                         method="xgbLinear",
                         trControl=fitControl
      )
      caret_tune <- caret_xgb$bestTune
      caret_best <- caret_xgb$finalModel
      caret_fit <- predict(caret_best,as.matrix(df[,-1]))
      caret_pred <- predict(caret_best,as.matrix(data.frame(Xte)))
      caret_pred
    })
  )
  data.table(
    code=dataseti$code,
    sample=1:length(pred),
    roi=pred
  )
}

modeli_naive <- function(dataseti){
  # print(paste(dataseti$code,Sys.time()))
  # dataseti <- datasets[[which(sapply(datasets,function(x){x$code})==300411)]]
  Xtr <- dataseti$Xtr
  Xte <- dataseti$Xte
  Ytr <- dataseti$Ytr
  dimnames(Xte) <- dimnames(Xtr) <- NULL
  if(nrow(Xtr)<50){
    return(data.table(code=dataseti$code,sample=1,roi=NA))
  }
  # weights <- log(1:nrow(Xtr))
  # weights <- rep(1,nrow(Xtr))
  # dtrain <- xgb.DMatrix(data=Xtr,label=Ytr,weight=weights)
  dtest <- xgb.DMatrix(data=Xte)
  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 3,
    eta = 0.1
  )
  # params <- list(
  #   booster = "gbtree",
  #   objective = "reg:squarederror",
  #   eta = 0.1,
  #   max_depth = 3,
  #   min_child_weight = 1,
  #   subsample = 0.5,
  #   colsample_bytree = 0.5
  # )
  pred <- lapply(1:100,function(i){
    nrounds <- 50
    set.seed(i); sel <- sample(1:length(Ytr),replace=T)
    dtraini <- xgb.DMatrix(data=Xtr[sel,],label=Ytr[sel])
    dtesti <- xgb.DMatrix(data=Xtr[-sel,],label=Ytr[-sel])
    model <- xgb.train(params = params, data = dtraini, nrounds = nrounds)
    r <- cor(predict(model,dtesti),Ytr[-sel],use='pairwise')
    p <- predict(model,dtest)
    list(model=model,pred=p,r=r)
  })
  # summary(sapply(pred,function(x){x$r}))
  pred <- pred[sapply(pred,function(x){x$r})>=0.1]
  if(length(pred)==0){
    return(data.table(code=dataseti$code,sample=1,roi=NA))
  }
  data.table(
    code=dataseti$code,
    sample=1:length(pred),
    roi=sapply(pred,function(x){x$pred})
  )
}

modeli_naive2 <- function(dataseti){
  # print(paste(dataseti$code,Sys.time()))
  Xtr <- dataseti$Xtr
  Xte <- dataseti$Xte
  Ytr <- dataseti$Ytr
  dimnames(Xte) <- dimnames(Xtr) <- NULL
  dnew <- xgb.DMatrix(data=Xte)
  if(nrow(Xtr)<50){
    return(data.table(code=dataseti$code,sample=1,roi=NA))
  }
  system.time(
    outs <- lapply(1:10,function(i){
      set.seed(i);sel <- sample(1:nrow(Xtr),nrow(Xtr)*0.7)
      dtrain <- xgb.DMatrix(data=Xtr[sel,,drop=F],label=Ytr[sel])
      dtest <- xgb.DMatrix(data=Xtr[-sel,,drop=F],label=Ytr[-sel])
      watchlist <- list(train=dtrain,test=dtest)
      eval_log <- list()
      params <- list(
        booster = "gbtree",
        objective = "reg:squarederror",
        eta = 0.1,
        max_depth = 3,
        min_child_weight = 1,
        subsample = 0.5,
        colsample_bytree = 0.5
      )
      xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, watchlist=watchlist, verbose=F)
      nroundi <- which.min(residuals(lm(xgb_model$evaluation_log$test_rmse-xgb_model$evaluation_log$train_rmse~I(1:100))))
      models <- lapply(1:10,function(j){
        xgb_model <- xgb.train(params = params, data = dtrain, nrounds = nroundi, verbose=F)
      })
      testi <- sapply(models,predict,newdata=dtest)
      list(models=models,valid=cor(rowMeans(testi),Ytr[-sel]))
    })
  )
  models <- do.call(c,lapply(outs,function(x){x$models}))
  valids <- do.call(c,lapply(outs,function(x){x$valid}))
  pred <- sapply(models,predict,newdata=Xte)
  data.table(
    code=dataseti$code,
    sample=1:length(pred),
    roi=pred
  )
}

modeli_ard <- function(dataseti){
  # print(paste(dataseti$code,Sys.time()))
  # dataseti <- datasets[[which(sapply(datasets,function(x){x$code})==300411)]]
  Xtr <- dataseti$Xtr
  Xte <- dataseti$Xte
  Ytr <- dataseti$Ytr
  dimnames(Xte) <- dimnames(Xtr) <- NULL
  if(nrow(Xtr)<50){
    return(data.table(code=dataseti$code,sample=1,roi=NA))
  }
  pred <- lapply(1:100,function(i){
    nrounds <- 50
    set.seed(i); sel <- sample(1:length(Ytr),replace=T)
    model <- skl$ARDRegression()
    model$fit(Xtr[sel,],Ytr[sel])
    r <- cor(model$predict(Xtr[-sel,]),Ytr[-sel])
    p <- model$predict(Xtr[-sel,])
    list(model=model,pred=p,r=r)
  })
  r <- sapply(pred,function(x){x$r})
  pred <- sapply(pred,function(x){x$model$predict(Xte)})
  data.table(
    code=dataseti$code,
    sample=1:length(pred),
    roi=pred
  )
}

######################################################################

pool <- getpool('jgp.txt','qs.txt')
dates <- unique(pool$date)
datei <- dates[length(dates)]

print(paste(datei,Sys.time()))
datesi <- dates[which(dates==datei)-0:4]
pooli <- filter(pool,date%in%datesi) %>%
  group_by(code) %>%
  summarise(ndate=n_distinct(date),dates=paste(unique(date),collapse=','),
            nfile=n_distinct(file),files=paste(unique(file),collapse=','))
raw <- gethis(pooli$code,datei,h=20000)
datasets <- getmodelfile(raw=raw,ni=20,nj=5)

system.time(outs <- lapply(datasets,modeli_naive))
(resulti <- do.call(rbind,outs) %>%
  group_by(code) %>%
  summarise(mean=mean(roi),sd=sd(roi))) %>%
  merge(pooli) %>%
  arrange(desc(mean)) %>%
  head

system.time(outs <- lapply(datasets,modeli_naive2))
(resulti <- do.call(rbind,outs) %>%
    group_by(code) %>%
    summarise(mean=mean(roi),sd=sd(roi))) %>%
  merge(pooli) %>%
  arrange(desc(mean)) %>%
  head

system.time(outs <- lapply(datasets,modeli_ard))
(resulti <- do.call(rbind,outs) %>%
    group_by(code) %>%
    summarise(mean=mean(roi),sd=sd(roi))) %>%
  merge(pooli) %>%
  arrange(desc(mean)) %>%
  head

######################################################################
# Trace
######################################################################

#Get Pool
pool <- getpool('jgp.txt','qs.txt')
dates <- unique(pool$date)
dates2go <- dates[dates>=20230101]
codesinpool <- unique((pool %>% filter(date>=20240301))$code)

#Get His Data
datei <- dates[length(dates)]
# system.time(
#   hisdata <- do.call(rbind,lapply(codesinpool,function(codei){
#     print(codei)
#     Sys.sleep(0.01)
#     x <- ak$stock_zh_a_hist(symbol=codei,
#                             period='daily',
#                             start_date=as.numeric(datei)-30000,
#                             end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
#       mutate(code=codei)
#     x[[1]] <- sapply(x[[1]],paste)
#     x
#   }))
# )
# write.csv(hisdata,'hisdata230101.csv')
hisdata <- read.csv('hisdata210101.csv')[,-1]

#Modeling

results <- lapply(dates2go,function(datei){
  print(paste(datei,Sys.time()))
  datesi <- dates[which(dates==datei)-0:4]
  pooli <- filter(pool,date%in%datesi) %>%
    group_by(code) %>%
    summarise(ndate=n_distinct(date),dates=paste(unique(date),collapse=','),
              nfile=n_distinct(file),files=paste(unique(file),collapse=','))
  #get rawdata for modeli
  raw <- hisdata %>% 
    select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
    mutate(date=gsub('-','',date)) %>%
    filter(date<as.numeric(datei),date>=as.numeric(datei)-10000,code%in%pooli$code)
  test <- melt(raw,id=1:2) %>%
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
        mutate(value=value/idx) %>%
        mutate(variable=paste0(variable,'p')) %>%
        select(date,code,variable,value)
    ) %>%
    mutate(value=ifelse((variable%in%c('volp','valp'))&(is.na(value)),0,value))
  #modeling
  datasets <- getmodelfile(raw=raw,ni=20,nj=5)
  system.time(outs <- lapply(datasets,modeli_ard))
  do.call(rbind,outs) %>%
    group_by(date=datei,code) %>%
    summarise(mean=mean(roi),sd=sd(roi)) %>%
    arrange(desc(mean))
})

do.call(rbind,results) %>% 
  merge(
    pool %>%
    group_by(date,code) %>%
    summarise(pool=paste(gsub('.txt','',unique(file)),collapse=',')),all.x=T) %>%
  write.csv('traceard_since_20230101.csv')

result <- do.call(rbind,results) %>% 
  merge(
    pool %>%
    group_by(date,code) %>%
    summarise(pool=paste(gsub('.txt','',unique(file)),collapse=',')),all.x=T) 
result <- read.csv('tracenaive2_since_20230101.csv')[,-1]

ref <- hisdata %>%
  select(date=日期,code,open=开盘,close=收盘) %>%
  mutate(date=gsub('-','',date)) 
ref$did <- match(ref$date,sort(unique(ref$date)))

ref <- ref %>%
  select(date,did,code,open) %>%
  merge(
    ref %>%
      select(did,code,close) %>%
      mutate(did=did-1)
  ) %>%
  mutate(roi=close/open)

result <- result %>%
  merge(
    ref %>%
      select(code,date,roi)
  )

result %>%
  filter(!is.na(pool)) %>%
  mutate(sel=(mean>1)) %>%
  group_by(sel) %>%
  summarise(mean(roi),n=n())

######################################################################
# Strategy
######################################################################

setwd('/Users/huzixin/Documents/jingou')
results <- read.csv('tracenaive_since_20230101.csv')[,-1]
hisdata <- do.call(rbind,lapply(unique(results$code),function(codei){
  codei <- substr(paste(codei+1000000),2,7)
  print(codei)
  Sys.sleep(0.01)
  x <- ak$stock_zh_a_hist(symbol=codei,
                          period='daily',
                          start_date=20220101,
                          end_date=20240402, adjust='qfq') %>%
    mutate(code=codei)
  x[[1]] <- sapply(x[[1]],paste)
  x
}))
hisdata <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=gsub('-','',date)) 

hisdata$did <- match(hisdata$date,sort(unique(hisdata$date))) 
results$did <- match(results$date,sort(unique(hisdata$date))) 

test <- results %>%
  merge(
    hisdata %>% select(did,code,open)    
  ) %>%
  merge(
    hisdata %>% select(did,code,close) %>% mutate(did=did-2)    
  ) %>%
  mutate(roi=close/open) 

test %>%
  filter(!is.na(pool)) %>%
  mutate(sel=(mean>=1.1)&(mean-sd>=1.05)) %>%
  group_by(sel,pool) %>%
  summarise(mean(roi),n())
