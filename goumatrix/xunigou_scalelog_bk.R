# 
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(zoo)
library(Rcpp)
library(RcppArmadillo)
library(randomForest)

use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
setwd('/home/huzixin/documents/goldgou')

########################################################################################################################

# #Get Historical Data
# 
# ak <- import("akshare")
# skl <- import('sklearn.linear_model')
# allcodes <- ak$stock_sh_a_spot_em() %>%
#   rbind(ak$stock_sz_a_spot_em()) %>%
#   select(code=2,name=3)
# 
# #Get His Data
# system.time(
#   hisdata <- do.call(rbind,lapply(allcodes$code,function(codei){
#     # print(codei)
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
# #Update data
# hisdata <- hisdata %>% merge(allcodes)
# write.csv(hisdata,'hfq180101.csv')

#Load Data
hisdata <- fread('hfq180101.csv')[,-1]
dates <- sort(unique(as.numeric(gsub('-','',unique(hisdata$日期)))))

jgp <- read.csv('raw_grade.csv')[,-1] %>%
  filter(grade%in%c('增持','买入'),ex.price!='-') %>%
  mutate(ex.price=as.numeric(ex.price)) %>%
  group_by(code,date) %>%
  summarise(eprice=min(ex.price))

########################################################################################################################
#最新数据
########################################################################################################################

rawhis <- hisdata %>% 
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量,name) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  filter(code%in%jgp$code) %>%
  filter(open>0,close>0,low>0,high>0)
datesi <- sort(unique(rawhis$date))  
rawhis$did <- match(rawhis$date,datesi)
codes <- unique(rawhis$code)
jgp$did <- sapply(jgp$date,function(datei){max(which(datesi<=datei))})
jgp <- jgp[jgp$code%in%rawhis$code,]

#Big Table
close <- rawhis %>% acast(date~code,value.var='close')
open <- rawhis %>% acast(date~code,value.var='open')
val <- rawhis %>% acast(date~code,value.var='val')
val[is.na(val)] <- 0

for(i in 2:nrow(close)){
  close[i,] <- ifelse(is.na(close[i,]),close[i-1,],close[i,])
  open[i,] <- ifelse(is.na(open[i,]),open[i-1,],open[i,])
}

########################################################################################################################
#量价猩基金 scale_log
########################################################################################################################

k <- 5
roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
pop <- pop[-(1:(nrow(pop)-nrow(roi))),]

#回测用数据

b <- 29
codek <- 20

system.time(
  datapool <- lapply((b+3):nrow(roi),function(i){
    codei <- unique((jgp %>% filter(did>=i-codek,did<i))$code) %>% paste
    roi0 <- roi[i-(b:0+2),codei]
    pop0 <- pop[i-(b:0+2),codei]
    roi2 <- roi[i,codei]
    roi0[is.na(roi0)] <- 1
    roi2[is.na(roi2)] <- 1
    X <- (rbind(roi=roi0,pop=pop0))
    Y <- data.table(
      code=colnames(roi0),
      obs=as.numeric(rownames(roi)[i-2]),
      buy=as.numeric(rownames(roi)[i-1]),
      sell=as.numeric(rownames(roi)[i]),
      roi=roi2
    )
    list(X=X,Y=Y)
  })
)

datapool <- datapool[(which(sapply(datapool,function(x){nrow(x$Y)})>1)[1]+20):length(datapool)]

#回测

# testperiod <- length(datapool)-200:0
testperiod <- (k+1):length(datapool)

bench <- rbindlist(
  lapply(datapool[testperiod],function(x){
    x$Y %>%
      group_by(obs,buy,sell) %>%
      summarise(bench=mean(roi))
  })
) %>%
  group_by(obs,buy,sell) %>%
  summarise(bench=prod(bench))

bench %>%
  group_by('total') %>%
  summarise(min(obs),max(sell),prod(bench,na.rm=T))

# system.time(
#   tests <- lapply(testperiod,function(i){
#     print(paste(i,length(testperiod)+b,Sys.time()))
#     x.obs <- datapool[[i]]$X
#     y.obs <- datapool[[i]]$Y
#     x.ref <- datapool[[i-k]]$X
#     y.ref <- datapool[[i-k]]$Y
#     if(ncol(x.obs)==0){return(NULL)}
#     do.call(rbind,lapply(1:ncol(x.obs),function(j){
#       xj.dist <- (x.obs[,j]-x.ref)^2
#       n <- nrow(xj.dist) / 2
#       xk <- data.table(
#         code = colnames(x.obs)[j],
#         roidist = colSums(xj.dist[1:n, ]),
#         popdist = colSums(xj.dist[(n + 1):(2 * n), ]),
#         y.ref %>% select(refcode=code,refroi=roi)
#       )
#       xk$roidist <- pnorm(scale(log(xk$roidist)))
#       xk$popdist <- pnorm(scale(log(xk$popdist)))
#       xk
#     })) %>%
#       merge(y.obs) %>%
#       filter(roidist<=0.05,popdist<=0.05) %>%
#       group_by(obs,buy,sell,code) %>%
#       summarise(mean=mean(refroi),win=mean(refroi>1),actual=mean(roi)) %>%
#       filter(mean>1,win>0.5) %>%
#       arrange(desc(mean))
#   })
# )
# save(tests,file='trace_scalelog_0921b29k2ck10.rda')
load('trace_scalelog_0921b29k2ck10.rda')

rlt <- rbindlist(lapply(tests,head,100))
# write.csv(rlt,'trace_scalelog_0921b29k2ck10.csv')

rbindlist(lapply(tests,head,5)) %>%
  group_by(obs) %>%
  summarise(actual=mean(actual)) %>%
  group_by(obs=substr(obs,1,4)) %>%
  summarise(actual=prod((actual-1)/2+1,na.rm=T),n()) %>%
  merge(
    bench %>%
      group_by(obs=substr(obs,1,4)) %>%
      summarise(bench=prod((bench-1)/2+1))
  )

#新数据

b <- 29
k <- 2
codek <- 20

roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}

i <- nrow(roi)
print(range((jgp %>% filter(did>=i-codek+2,did<i+2))$date))
codei <- unique((jgp %>% filter(did>=i-codek+2,did<i+2))$code) %>% paste
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
  merge(
    unique(rawhis %>% select(code,name))
  ) %>%
  arrange(desc(mean)) %>%
  head(20)

