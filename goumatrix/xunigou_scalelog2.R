
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)

use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
setwd('/home/huzixin/documents/goldgou')

########################################################################################################################
#Get Historical Data

# ak <- import("akshare")
# skl <- import('sklearn.linear_model')
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
#                             start_date=20180101,
#                             end_date=gsub('-','',Sys.Date()), adjust='hfq') %>%
#       mutate(code=codei)
#     x[[1]] <- sapply(x[[1]],paste)
#     x
#   }))
# )
# 
# #Get Hourly Data
# 
# system.time(
#   mindata <- do.call(rbind,lapply(allcodes$code,function(codei){
#     print(codei)
#     Sys.sleep(0.01)
#     x <- try(ak$stock_zh_a_hist_min_em(symbol=codei,
#                                        period='60',
#                                        adjust='hfq') %>%
#                mutate(code=codei)
#     )
#     if(class(x)=='try-error'){
#       return(NULL)
#     } else {
#       x[[1]] <- sapply(x[[1]],paste)
#       return(x)
#     }
#   }))
# )
# 
## Update Data
# 
# write.csv(allcodes,'data/allcodes.csv')
# hisdata <- hisdata
# write.csv(hisdata,paste0('data/day',gsub('-','',substr(Sys.time(),1,10)),'.csv'))
# mindata <- mindata
# write.csv(mindata,paste0('data/hour',gsub('-','',substr(Sys.time(),1,10)),'.csv'))

#Load Data

allcode <- fread('data/allcodes.csv')[,-1]

jgp <- read.csv('data/raw_grade.csv')[,-1] %>%
  filter(grade%in%c('增持','买入'),ex.price!='-') %>%
  mutate(ex.price=as.numeric(ex.price)) %>%
  group_by(code,date) %>%
  summarise(eprice=min(ex.price)) %>%
  filter(code%in%allcode$code)

allcode <- allcode %>%
  filter(code%in%jgp$code)

maxdate <- max(gsub('day|\\.csv','',dir('data',pattern='day')))

ddata <- fread(paste0('data/day',maxdate,'.csv'))[,-1] %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  filter(code%in%allcode$code)

hdata <- fread(paste0('data/hour',maxdate,'.csv'))[,-1] %>%
  select(time=时间,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(time=as.numeric(gsub("[^0-9]","",time))/100) %>%
  mutate(date=floor(time/10000)) %>%
  filter(code%in%allcode$code)

datemap <- sort(unique(ddata$date))
datemap <- data.table(date=datemap,did=1:length(datemap))

########################################################################################################################
#Daily Data Model

#Big Table

close <- ddata %>% acast(date~code,value.var='close')
open <- ddata %>% acast(date~code,value.var='open')
val <- ddata %>% acast(date~code,value.var='val')
val[is.na(val)] <- 0

for(i in 2:nrow(close)){
  close[i,] <- ifelse(is.na(close[i,]),close[i-1,],close[i,])
  open[i,] <- ifelse(is.na(open[i,]),open[i-1,],open[i,])
}

#Subdata

getbench <- function(iminus,b=29,k=2,codek=20){
  # iminus <- 2
  # b <- 29
  # k <- 2
  # codek <- 20
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
  pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
  if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}
  i <- nrow(roi)-iminus
  print(paste(rownames(roi)[i],Sys.time()))
  startdid <- datemap$date[which(datemap$date==rownames(roi)[i])-20]
  enddid <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi <- jgp %>% filter(date>startdid,date<=enddid)
  codei <- unique(jgpi$code) %>% paste
  data.table(obs=rownames(roi)[i],bench=mean(roi[i+2,],na.rm=T),jgp=mean(roi[i+2,codei],na.rm=T))
}

getdaycode <- function(iminus,b=29,k=2,codek=20){
  # b <- 29
  # k <- 2
  # codek <- 20
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
  pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
  if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}
  i <- nrow(roi)-iminus
  print(paste(rownames(roi)[i],Sys.time()))
  startdid <- datemap$date[which(datemap$date==rownames(roi)[i])-20]
  enddid <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi <- jgp %>% filter(date>startdid,date<=enddid)
  codei <- unique(jgpi$code) %>% paste
  roi0 <- roi[i-(b:0+k),codei]
  pop0 <- pop[i-(b:0+k),codei]
  roi2 <- roi[i-(b:0),codei]
  pop2 <- pop[i-(b:0),codei]
  x.ref <- (rbind(roi=roi0,pop=pop0))
  x.obs <- (rbind(roi=roi2,pop=pop2))
  y.ref <- data.table(
    code=colnames(roi0),
    obs=as.numeric(rownames(roi)[i-2]),
    buy=as.numeric(rownames(roi)[i-1]),
    sell=as.numeric(rownames(roi)[i]),
    roi=roi2[nrow(roi2),]
  )
  rlt <- data.table(
    obs=rownames(roi)[i],
    do.call(rbind,lapply(1:ncol(x.obs),function(j){
      xj.dist <- (x.obs[,j]-x.ref)^2
      n <- nrow(xj.dist) / 2
      xk <- data.table(
        code = colnames(x.obs)[j],
        roidist = colSums(xj.dist[1:n, ]),
        popdist = colSums(xj.dist[(n + 1):(2 * n), ]),
        y.ref %>% select(refcode=code,refroi=roi)
      )
      xk$roidist <- pnorm(scale(log(xk$roidist)))
      xk$popdist <- pnorm(scale(log(xk$popdist)))
      xk
    })) %>%
      filter(roidist<=0.05,popdist<=0.05) %>%
      group_by(code) %>%
      summarise(mean=mean(refroi),win=mean(refroi>1)) %>%
      filter(mean>1,win>0.5) %>%
      arrange(desc(mean)) 
  )
  rlt
}

print(head(dayrlt <- getdaycode(0)))
system.time(tests <- lapply(1400:0,getdaycode))
system.time(benchs <- lapply(1400:2,getbench))
save(tests,benchs,file='result/test0930.rda')
load('result/test0930.rda')

do.call(rbind,tests) %>%
  mutate(obs=as.numeric(obs),code=as.numeric(code)) %>%
  merge(datemap %>% select(obs=date,did=did)) %>%
  mutate(buy=did+1,sell=did+2) %>%
  merge(datemap %>% select(buy=did,buydate=date), by='buy') %>%
  merge(datemap %>% select(sell=did,selldate=date), by='sell') %>%
  merge(ddata %>% select(buydate=date,code,open), by=c('buydate','code')) %>%
  merge(ddata %>% select(selldate=date,code,close), by=c('selldate','code')) %>%
  mutate(roi=close/open) %>%
  select(code,name,obs,mean,win,roi) %>%
  arrange(obs,desc(mean)) %>%
  group_by(obs) %>%
  summarise(roi=mean(head(roi,10))) %>%
  merge(do.call(rbind,benchs) %>% mutate(obs=as.numeric(obs))) %>%
  mutate(roi=(roi-1)/2+1,bench=(bench-1)/2+1,jgp=(jgp-1)/2+1) %>%
  group_by(year=substr(obs,1,4)) %>%
  summarise(n=n(),roi=prod(roi),jgp=prod(jgp),bench=prod(bench)) 
 
########################################################################################################################
#Daily Data Model

modelfiles <- lapply(2:10,function(iminus){
  i <- length(datemap$date)-iminus
  obs <- datemap$date[i]
  jgpi <- jgp %>% filter(date>datemap$date[i-20],date<=datemap$date[i])
  codei <- unique(jgpi$code) %>% paste
  x.ref <- hdata %>%
    filter(
      date>=datemap$date[i-3],
      date<=datemap$date[i-2],
      code%in%codei
    )
  x.obs <- hdata %>%
    filter(
      date>=datemap$date[i-1],
      date<=datemap$date[i],
      code%in%codei
    )
  x.roi <- hdata %>%
    filter(
      date>=datemap$date[i+1],
      date<=datemap$date[i+2],
      code%in%codei
    )
  close.obs <- x.obs %>% acast(time~code,value.var='close')
  close.obs <- t(t(close.obs)/(x.obs %>% acast(time~code,value.var='open'))[1,])
  val.obs <- x.obs %>% acast(time~code,value.var='val')
  val.obs <- t(t(val.obs[-1,])/val.obs[1,])
  close.ref <- x.ref %>% acast(time~code,value.var='close')
  close.ref <- t(t(close.ref)/(x.ref %>% acast(time~code,value.var='open'))[1,])
  val.ref <- x.ref %>% acast(time~code,value.var='val')
  val.ref <- t(t(val.ref[-1,])/val.ref[1,])
  close.roi <- x.roi %>% acast(time~code,value.var='close')
  close.roi <- t(t(close.roi)/(x.roi %>% acast(time~code,value.var='open'))[1,])
  val.roi <- x.roi %>% acast(time~code,value.var='val')
  val.roi <- t(t(val.roi[-1,])/val.roi[1,])
  modelfile <- list(close.obs=close.obs,val.obs=val.obs,close.ref=close.ref,val.roi=val.roi,close.roi=close.roi)
  modelfile
})

X.close <- do.call(cbind,lapply(modelfiles,function(x){x$close.obs}))
X.val <- do.call(cbind,lapply(modelfiles,function(x){x$val.obs}))
Y <- do.call(cbind,lapply(modelfiles,function(x){x$close.roi}))
Y <- t(apply(Y,2,function(x){
  c(roi=as.numeric(x[length(x)]),mean=mean(x),min=min(x),max=max(x))
}))

X.close <- t(impute::impute.knn(t(X.close))[[1]])
X.val <- t(impute::impute.knn(t(X.val))[[1]])
cor(X.close)

dist.close <- apply(close.obs[,dayrlt$code],2,function(x){pnorm(scale(colSums((x-close.ref)^2)))}) %>%
  melt() %>%
  select(code=2,refcode=1,closedist=3)
dist.val <- apply(val.obs[,dayrlt$code],2,function(x){pnorm(scale(colSums((x-val.ref)^2)))}) %>%
  melt() %>%
  select(code=2,refcode=1,valdist=3)
rlt <- dist.close %>%
  cbind(valdist=dist.val$valdist) %>%
  merge(
    data.table(
      refcode=colnames(close.ref),
      t(apply(close.obs,2,function(x){
        c(max=max(x),min=min(x),mean=mean(x),roi=as.numeric(x[length(x)]))
      }))
    )
  )

dayrlt %>%
  mutate(code=as.numeric(code)) %>%
  merge(
    rlt %>%
      filter(closedist<0.3,valdist<0.3) %>%
      group_by(code) %>%
      summarise(mean2=mean(mean),max=mean(max),min=mean(min),roi=mean(roi))
  ) %>%
  arrange(desc(mean)) %>%
  arrange(desc(roi))
