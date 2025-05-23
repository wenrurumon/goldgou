
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)

use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
setwd('/home/huzixin/documents/goldgou')

# setwd('/Users/huzixin/Documents/jingou/')

########################################################################################################################
#Get Historical Data
# 
# ak <- import("akshare")
# skl <- import('sklearn.linear_model')
# allcodes <- ak$stock_sh_a_spot_em() %>%
#   rbind(ak$stock_sz_a_spot_em()) %>%
#   select(code=2,name=3)
# write.csv(allcodes,'data/allcodes.csv')
# 
# #Get His Data
# 
# system.time(
#   hisdata <- do.call(rbind,lapply(allcodes$code,function(codei){
#     print(codei)
#     # Sys.sleep(0.01)
#     x <- ak$stock_zh_a_hist(symbol=codei,
#                             period='daily',
#                             start_date=20170101,
#                             end_date=gsub('-','',Sys.Date()), adjust='hfq') %>%
#       mutate(code=codei)
#     x[[1]] <- sapply(x[[1]],paste)
#     x
#   }))
# )
# write.csv(hisdata,paste0('data/hisdata.csv'))

# system.time(
#   hisdata <- do.call(rbind,lapply(allcodes$code,function(codei){
#     print(codei)
#     # Sys.sleep(0.01)
#     x <- ak$stock_zh_a_hist(symbol=codei,
#                             period='daily',
#                             start_date=as.numeric(substr(gsub('[^0-9]','',Sys.time()),1,8))-10000,
#                             end_date=gsub('-','',Sys.Date()), adjust='hfq') %>%
#       mutate(code=codei)
#     x[[1]] <- sapply(x[[1]],paste)
#     x
#   }))
# )
# write.csv(hisdata,paste0('data/day',gsub('-','',substr(Sys.time(),1,10)),'.csv'))
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

# maxdate <- max(gsub('day|\\.csv','',dir('data',pattern='day')))

hisddata <- fread('data/hisdata.csv')[,-1] %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  filter(code%in%allcode$code)
ddata <- hisddata

# ddata <- fread(paste0('data/day',maxdate,'.csv'))[,-1] %>%
#   select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
#   mutate(date=as.numeric(gsub('-','',date))) %>%
#   filter(code%in%allcode$code)

# hdata <- fread(paste0('data/hour',maxdate,'.csv'))[,-1] %>%
#   select(time=时间,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
#   mutate(time=as.numeric(gsub("[^0-9]","",time))/100) %>%
#   mutate(date=floor(time/10000)) %>%
#   filter(code%in%allcode$code)

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

close <- apply(close,2,function(x){
  x[which(!is.na(x))[1]+0:5] <- NA
  x
})
open <- apply(open,2,function(x){
  x[which(!is.na(x))[1]+0:5] <- NA
  x
})

#Subdata

getbench <- function(iminus,b=29,k=2,codek=20){
  # iminus <- 3
  # b <- 29
  # k <- 2
  # codek <- 2
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
  pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
  if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}
  i <- nrow(roi)-iminus
  print(paste(rownames(roi)[i],Sys.time()))
  startdid <- datemap$date[which(datemap$date==rownames(roi)[i])-codek]
  enddid <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi <- jgp %>% filter(date>startdid,date<=enddid)
  codei <- unique(jgpi$code) %>% paste
  data.table(obs=rownames(roi)[i],bench=mean(roi[i+2,],na.rm=T),jgp=mean(roi[i+2,codei],na.rm=T))
}

getdaycode <- function(iminus,b=29,k=2,codek0=20,codek2=10,r=0){
  # iminus <- 3
  # b <- 29
  # k <- 2
  # codek0 <- 15
  # codek2 <- 10
  # r <- 0
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
  pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
  if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}
  i <- nrow(roi)-iminus
  print(paste(iminus,rownames(roi)[i],Sys.time()))
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
    sell=as.numeric(rownames(roi)[i]),
    roi=roi1[nrow(roi1),]
  )
  rlt <- data.table(
    obs=rownames(roi)[i],
    buy=rownames(roi)[i+1],
    sell=rownames(roi)[i+2],
    do.call(rbind,lapply(1:ncol(x.obs),function(j){
      xj.dist <- (x.obs[,j]-x.ref)^2 * rep((0:b+1)^r,2)
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

getdaycode2 <- function(iminus,b=29,k=2,codek0=20,codek2=10,r=0){
  # iminus <- 3
  # b <- 29
  # k <- 2
  # codek0 <- 15
  # codek2 <- 10
  # r <- 0
  roi <- close[k:nrow(close),]/open[1:(nrow(close)-(k-1)),]
  pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
  pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]
  if(nrow(pop)>nrow(roi)){pop <- pop[-(1:(k-2)),]}
  i <- nrow(roi)-iminus
  print(paste(iminus,rownames(roi)[i],Sys.time()))
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
    sell=as.numeric(rownames(roi)[i]),
    roi=roi1[nrow(roi1),]
  )
  rlt <- data.table(
    obs=rownames(roi)[i],
    buy=rownames(roi)[i+1],
    sell=rownames(roi)[i+2],
    do.call(rbind,lapply(1:ncol(x.obs),function(j){
      xj.dist <- (x.obs[,j]-x.ref)^2 * rep((0:b+1)^r,2)
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

validtest <- function(tests,scale=4,sel=5){
  do.call(rbind,lapply(tests,head,sel)) %>%
    merge(
      melt(close[-1,]/open[-nrow(open),]) %>%
        select(code=2,sell=1,actual=3) %>%
        mutate(code=paste(code),sell=paste(sell)),
      all.x=T
    ) %>%
    group_by(sell) %>%
    summarise(actual=mean(actual-1)/2+1) %>%
    # summarise(actual=sum(actual),n=n()) %>%
    # mutate(actual=ifelse(is.na(actual),1,actual)) %>%
    # mutate(actual=((actual+(5-n)*1)/5-1)/2+1) %>% 
    merge(
      melt(close[-1,]/open[-nrow(open),]) %>%
        select(code=2,sell=1,actual=3) %>%
        mutate(code=paste(code),sell=paste(sell)) %>%
        group_by(sell) %>%
        summarise(bench=mean(actual,na.rm=T)) %>%
        mutate(bench=(bench-1)/2+1) %>%
        filter(sell>=min(tests1$sell,na.rm=T)),all.y=T
    ) %>%
    group_by(substr(sell,1,scale)) %>%
    summarise(
      miss=mean(is.na(actual)),
      simuroi=prod(actual,na.rm=T),
      benchroi=prod(bench,na.rm=T),
      winbench=mean(ifelse(is.na(actual),1,actual)>bench,na.rm=T),
      win1=mean(ifelse(is.na(actual),1,actual)>=1,na.rm=T),
      bench1=mean(bench>=1,na.rm=T)
    )
}

#Main

getdaycode(0,codek0=15,codek2=10,r=0) %>%
  merge(allcode %>% mutate(code=paste(code))) %>%
  arrange(desc(mean))

lapply(4:0,getdaycode,codek0=15,codek2=10,r=0)

#Test

system.time(tests <- lapply(nrow(filter(datemap,date>=20180000)):0,getdaycode,codek0=15,codek2=10))
save(tests,file='result/test_1011_1510.rda')
(rlt <- rbindlist(tests)) %>% write.csv('result/test_1011_1510.csv')

validtest(tests,scale=1,sel=5)

###########################################################################


