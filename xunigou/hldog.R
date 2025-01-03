
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)

# use_condaenv(condaenv='/home/huzixin/anaconda3/envs/test/bin/python',required=TRUE)
# setwd('/home/huzixin/documents/goldgou')

setwd('/Users/huzixin/Documents/goldgou/')
ak <- import("akshare")

########################################################################################################################
# Get Historical Data

updatedata <- function(){
  
  allcodes <- ak$stock_sh_a_spot_em() %>%
    rbind(ak$stock_sz_a_spot_em()) %>%
    select(code=2,name=3)
  write.csv(allcodes,'data/allcodes.csv')
  
  #Get His Data
  
  getcodei <- function(codei){
    x <- ak$stock_zh_a_hist(symbol=codei,
                            period='daily',
                            start_date=20181001,
                            end_date=gsub('-','',Sys.Date()), adjust='hfq') %>%
      mutate(code=codei)
    x[[1]] <- sapply(x[[1]],paste)
    x
  }
  
  system.time(
    hisdata <- lapply(allcodes$code,function(codei){
      print(codei)
      try(getcodei(codei))
    })
  )
  
  hisdata <- do.call(rbind,hisdata[which(sapply(hisdata,is.data.frame))])
  
  write.csv(hisdata,paste0('data/hisdata.csv'))
  
}

max(fread('data/hisdata.csv')[[2]])
# system.time(updatedata())

################################################################################
#Module

hgou <- function(iminus,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05){
  # iminus <- 0
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
  codei0 <- unique(c(jgpi0$code,hongli$code)) %>% paste
  roi0 <- roi[i-(b:0+k),codei0,drop=F]
  pop0 <- pop[i-(b:0+k),codei0,drop=F]
  roi1 <- roi[i-(b:0),codei0]
  startdid2 <- datemap$date[which(datemap$date==rownames(roi)[i])-codek2]
  enddid2 <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi2 <- jgp %>% filter(date>startdid2,date<=enddid2,code%in%colnames(close))
  codei2 <- unique(c(jgpi2$code,hongli$code)) %>% paste
  roi2 <- roi[i-(b:0),codei2,drop=F]
  pop2 <- pop[i-(b:0),codei2,drop=F]
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
    summarise(mean=mean(roi),win=mean(roi>1),reference=paste(paste0(code0,'@',obs),collapse=',')) %>%
    filter(mean>1,win>0.5) %>%
    arrange(desc(mean)) 
}

jgou <- function(iminus,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05){
  # iminus <- 0
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
  roi0 <- roi[i-(b:0+k),codei0,drop=F]
  pop0 <- pop[i-(b:0+k),codei0,drop=F]
  roi1 <- roi[i-(b:0),codei0]
  startdid2 <- datemap$date[which(datemap$date==rownames(roi)[i])-codek2]
  enddid2 <- datemap$date[which(datemap$date==rownames(roi)[i])]
  jgpi2 <- jgp %>% filter(date>startdid2,date<=enddid2,code%in%colnames(close))
  codei2 <- unique(jgpi2$code) %>% paste
  roi2 <- roi[i-(b:0),codei2,drop=F]
  pop2 <- pop[i-(b:0),codei2,drop=F]
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
    summarise(mean=mean(roi),win=mean(roi>1),reference=paste(paste0(code0,'@',obs),collapse=',')) %>%
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
    lower = open[obs,],
    upper = close[obs,]*1.
  )
  
  data.table(
    code=as.numeric(colnames(close)),
    obs=as.numeric(rownames(close)[obs]),
    yg0=rowMeans(yg0)==1,
    yg1,
    val0=val[nrow(val)-iminus,]
  ) %>%
    filter(yg0) %>%
    select(code,obs,lower,upper,val0) %>%
    arrange(val0)
  
}

valid <- function(tests,k=2,h=5,gaokai_rate=1.001){
  
  # scale <- 8
  # k <- 2
  # h <- 5
  # gaokai_rate <- 1.0
  
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

darisk <- function(k=2,ncode=50){
  mat <- close[k:nrow(close),] / open[1:(nrow(open)-k+1),]
  valt <- sapply(k:nrow(close),function(i){
    x <- colSums(val[1:k+i-k,],na.rm=T)
    x>=quantile(x,1-ncode/length(x),na.rm=T)
  }) %>% t
  risk <- data.table(
    obs=as.numeric(rownames(mat)),
    t(
      sapply(1:nrow(valt),function(i){
        c(
          roi50=mean(mat[i,valt[i,]],na.rm=T),
          win50=mean(mat[i,valt[i,]]>1,na.rm=T)
        )
      })
    )
  )
  risk$sroi50 <- c(NA,sapply(2:nrow(risk),function(i){prod(risk$roi50[-1:0+i])}))
  risk
}

tradelist <- function(dgaccount=300000,ignore.update=F,h=5){
  
  #   dgaccount <- 300000*0.95
  #   ignore.update <- T
  #   h <- 5
  
  qfq <- lapply(substr(paste(c(strategy$JinGou$code,strategy$YinGou$code)+1000000),2,7),function(codei){
    x <- ak$stock_zh_a_hist(symbol=codei,
                            period='daily',
                            start_date=as.numeric(gsub('-','',Sys.Date()-10)),
                            end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
      mutate(code=codei)
    x[[1]] <- sapply(x[[1]],paste)
    tail(x,2)
  }) %>%
    rbindlist() %>%
    select(code,date=1,open=2,close=3,high=4,low=5) %>%
    mutate(code=as.numeric(code)) %>%
    mutate(date=as.numeric(gsub('-','',date))) %>%
    arrange(code,date) %>%
    group_by(code) %>%
    summarise(obs.=date[1],buy.=date[2],open0=open[1],close0=close[1],open1=open[2],close1=close[2])
  
  jg <- strategy$JinGou %>%
    select(code,name,obs) %>%
    merge(qfq) %>%
    mutate(price_in=ifelse(open1>close0,close0,open1)) %>%
    head(h)
  
  yg <- strategy$YinGou %>%
    select(code,name,obs,val0) %>%
    merge(qfq) %>%
    filter(open1>=open0,open1<=close0) %>%
    mutate(price_in=open1) %>%
    arrange(desc(close1/open1)) %>%
    arrange(val0) %>%
    head(h) %>%
    select(-val0)
  
  jg <- data.table(dog='JG',jg)
  yg <- data.table(dog='YG',yg)
  
  if(!ignore.update){
    
    jg <- jg %>% filter(obs==obs.)
    yg <- yg %>% filter(obs==obs.)
    
  }
  
  ptradelist <- list(
    strategy=ifelse(tail(risk$sel,1),'YinGou','JinGou'),
    buydate=max(qfq$buy.),
    jg=jg,yg=yg)
  
  print(ptradelist)
  
  #PTrade
  
  transaction.day <- gsub('-','',Sys.Date())
  
  ptradelist$jg$code <- substr(paste(ptradelist$jg$code+1000000),2,7)
  ptradelist$jg <- ptradelist$jg %>%
    mutate(quantity=dgaccount/nrow(ptradelist$jg)/price_in) %>%
    mutate(quantity=floor(quantity/100)*100)
  
  ptradelist$yg$code <- substr(paste(ptradelist$yg$code+1000000),2,7)
  ptradelist$yg <- ptradelist$yg %>%
    mutate(quantity=dgaccount/nrow(ptradelist$yg)/price_in) %>%
    mutate(quantity=floor(quantity/100)*100)
  
  if(ptradelist[[1]]=='JinGou') {
    
    dgtrans <- data.frame(
      外部编号=rep('32f2259b-7b4',nrow(ptradelist$jg))
    ) %>%
      mutate(证券市场=if_else(substr(ptradelist$jg$code,1,1)==6,1,2),
             交易方向=1,
             策略编号=if_else(substr(ptradelist$jg$code,1,1)==6,'0','ATP'),
             算法编号='dg_2',
             委托数量=ptradelist$jg$quantity,
             委托属性=if_else(substr(ptradelist$jg$code,1,1)==6,'R','Q'),
             委托编号='',
             证券代码=ptradelist$jg$code,
             生成时间='131420888',
             资金账号='2044019641',
             委托价格=ptradelist$jg$price_in)
    
  } else {
    
    dgtrans <- data.frame(
      外部编号=rep('32f2259b-7b4',nrow(ptradelist$yg))
    ) %>%
      mutate(证券市场=if_else(substr(ptradelist$yg$code,1,1)==6,1,2),
             交易方向=1,
             策略编号=if_else(substr(ptradelist$yg$code,1,1)==6,'0','ATP'),
             算法编号='dg_2',
             委托数量=ptradelist$yg$quantity,
             委托属性=if_else(substr(ptradelist$yg$code,1,1)==6,'R','Q'),
             委托编号='',
             证券代码=ptradelist$yg$code,
             生成时间='131420888',
             资金账号='2044019641',
             委托价格=ptradelist$yg$price_in)
    
  }
  
  if(length(dir(path='sheet',pattern='csv'))>0){
    system('mv sheet/*.csv sheet/old/.')  
  }
  
  print(dgtrans)
  
  write.csv(dgtrans,paste0('sheet/dgtrans',transaction.day,'.csv'),row.names=F,fileEncoding='GBK')
  print(paste0('sheet/dgtrans',transaction.day,'.csv generated'))
  
  return(ptradelist)
  
}

track_order <- function(iminuss=2){
  
  lapply(iminuss:0,function(iminus){
    
    # print(iminus)
    
    Sys.sleep(.5)
    
    h <- 5
    
    jgi <- hgou(iminus,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05) %>% 
      arrange(desc(mean)) %>%
      head(h) %>%
      as.data.frame %>%
      mutate(dog='JinGou') %>%
      mutate(code=substr(paste(code+1000000),2,7)) %>%
      head(h)
    
    ygi <- ygou(iminus,k=2) %>% merge(allcode) %>% 
      filter(!grepl('ST|退',name)) %>% 
      arrange(val0) %>%
      mutate(dog='YinGou') %>%
      mutate(code=substr(paste(code+1000000),2,7))
    
    orders <- jgi
    if(nrow(jgi)==0){
      jgi <- NULL 
    } else {
      jgi <- merge(
        jgi %>% select(-buy,-sell),
        rbindlist(
          lapply(1:nrow(orders),function(i){
            x <- ak$stock_zh_a_hist(symbol=orders$code[i],
                                    period='daily',
                                    start_date=orders$obs[i],
                                    end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
              mutate(code=orders$code[i])
            x[[1]] <- sapply(x[[1]],paste)
            x %>% 
              mutate(code=orders$code[i]) %>%
              select(code=code,date=1,open=2,close=3,high=4,low=5) %>%
              mutate(date=as.numeric(gsub('-','',date))) %>%
              arrange(code,date) %>%
              group_by(code) %>%
              summarise(
                obs=date[1],buy=date[2],sell=date[min(length(date),3)],
                open0=open[1],close0=close[1],
                open1=open[2],close1=close[2],low1=low[2],
                open.=open[min(length(open),3)],close.=close[min(length(close),3)]
              )
          })
        ) %>%
          mutate(price_in=ifelse(close0>open1,open1,ifelse(low1>close0,NA,close0))),
        by=c('code','obs')
      ) %>% 
        select(dog,code,obs,buy,sell,price_in,close.) %>%
        mutate(roi=close./price_in) %>%
        mutate(roi=ifelse(is.na(roi),1,roi))
    }
    
    orders <- ygi
    if(nrow(ygi)==0){
      ygi <- NULL
    } else {
      ygi <- merge(
        ygi %>% select(dog,code,obs,val0),
        rbindlist(
          lapply(1:nrow(orders),function(i){
            x <- ak$stock_zh_a_hist(symbol=orders$code[i],
                                    period='daily',
                                    start_date=orders$obs[i],
                                    end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
              mutate(code=orders$code[i])
            x[[1]] <- sapply(x[[1]],paste)
            x %>% 
              mutate(code=orders$code[i]) %>%
              select(code=code,date=1,open=2,close=3,high=4,low=5) %>%
              mutate(date=as.numeric(gsub('-','',date))) %>%
              arrange(code,date) %>%
              group_by(code) %>%
              summarise(
                obs=date[1],buy=date[2],sell=date[min(length(date),3)],
                open0=open[1],close0=close[1],
                open1=open[2],close1=close[2],low1=low[2],
                open.=open[min(length(open),3)],close.=close[min(length(close),3)]
              )
          })
        ),
        by=c('code','obs')
      ) %>%
        filter(open1>=open0,open1<=close0) %>%
        mutate(price_in=open1) %>%
        arrange(val0) %>%
        select(dog,code,obs,buy,sell,price_in,close.) %>%
        mutate(roi=close./price_in) %>%
        head(h)
    }
    
    rbind(jgi,ygi)
    
  }) %>% rbindlist() %>% mutate(code=as.numeric(code)) %>% merge(allcode)
  
}

track <- function(){
  print(orders <- track_order(1) %>% arrange(-obs,dog))
  orders %>%
    group_by(dog,obs) %>%
    summarise(roi=mean(roi-1)/2) %>%
    arrange(desc(obs)) %>%
    as.data.frame() %>%
    print()
  orders %>%
    group_by(obs,code,name) %>%
    summarise(roi=mean(roi-1)/2) %>%
    arrange(desc(obs)) %>%
    as.data.frame() %>%
    print()
  orders %>%
    mutate(roi=ifelse(is.na(roi),1,roi)) %>%
    group_by(dog,obs) %>%
    summarise(roi=mean(roi-1)/2,code=paste(name,collapse=',')) %>%
    group_by(dog) %>%
    summarise(min(obs),max(obs),roi=prod(roi+1)-1) %>%
    as.data.frame() %>%
    print()
}


########################################################################################################################
# Data Setting

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
) %>%
  filter(obs>=20190000)

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

risk <- darisk(2,50) %>%
  mutate(sel=((win50>=0.6)|((sroi50>1.02)&(roi50<1))))

#Hongli

hongli <- read.csv('data/hongli1219.csv') %>%
  select(date=1,code=5,name=6) %>%
  filter(code%in%colnames(close))

################################################################################
#Onsite

iminus <- 0
print(
  strategy <- list(
    strategy=ifelse(tail(risk$sel,iminus+1)[1],'YinGou','JinGou'),
    JinGou=hgou(iminus,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05) %>% merge(allcode) %>% arrange(desc(mean)),
    YinGou=ygou(iminus,k=2) %>% merge(allcode) %>% filter(!grepl('ST|退',name)) %>% arrange(val0)
  )
)

#Trade

dgaccount <- 255000
rbindlist((tradelist(dgaccount*0.9,F,5))[c('jg','yg')])    
rbindlist((tradelist(dgaccount*0.9,F,10))[c('jg','yg')])    

while(T){
  if(substr(Sys.time(),12,16)=='09:26'){
    print(Sys.time())
    dgtrans <- rbindlist((tradelist(dgaccount*0.9,F,5))[c('jg','yg')])    
    stop()
  } else {
    print(Sys.time())
    Sys.sleep(10)
    
  }
}

#Tracking

track()

################################################################################
#Trace Back

#Tracing

start_date <- 20190101

load('result/test.rda')

# system.time(tests_jg <- lapply(sum(as.numeric(rownames(close))>=start_date):0,function(i){
#   print(paste(i,Sys.time()))
#   jgou(i,b=29,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05)
# }))
# 
# system.time(tests_hg <- lapply(sum(as.numeric(rownames(close))>=start_date):0,function(i){
#   print(paste(i,Sys.time()))
#   hgou(i,b=19,k=2,codek0=15,codek2=10,thres_roi=0.05,thres_pop=0.05)
# }))
# 
# system.time(
#   tests_yg <- lapply(sum(as.numeric(rownames(close))>=start_date):2,function(iminus){
#     testi <- ygou(iminus) %>% merge(allcode) %>% merge(obsmap,by='obs')
#     testi$open1 <- open[unique(paste(testi$buy)),paste(testi$code)]
#     testi$val0 <- open[unique(paste(testi$obs)),paste(testi$code)]
#     testi <- testi %>%
#       filter(!grepl('ST|退',name)) %>%
#       filter(open1>=lower,open1<=upper)
#     testi %>%
#       select(obs,buy,sell,code,val0) %>%
#       arrange(val0)
#   })
# )
# 
# system.time(
#   tests_yg2 <- lapply(sum(as.numeric(rownames(close))>=start_date):2,function(iminus){
#     testi <- ygou(iminus) %>% merge(allcode) %>% merge(obsmap,by='obs')
#     testi$open1 <- open[unique(paste(testi$buy)),paste(testi$code)]
#     testi$val0 <- open[unique(paste(testi$obs)),paste(testi$code)]
#     testi <- testi %>%
#       filter(!grepl('ST|退',name)) %>%
#       filter(open1>=lower,open1<=upper)
#     testi %>%
#       select(obs,buy,sell,code,val0) %>%
#       arrange(-val0)
#   })
# )
# 
# save(tests_yg,tests_yg2,tests_hg,tests_jg,file='result/test.rda')

#Validation

bench <- obsmap %>%
  filter(obs>=start_date) %>%
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

#Select Dog

h. <- 5

test <- obsmap %>%
  filter(obs>=start_date) %>%
  # filter(obs>20240101,obs<=20241301) %>%
  merge(
    valid(tests_jg,gaokai_rate=1.001,h=h.) %>% as.data.frame %>% select(obs,jg=roi),all.x=T,
    by='obs'
  ) %>%
  merge(
    valid(tests_yg,gaokai_rate=1.001,h=h.) %>% as.data.frame %>% select(obs,yg=roi),all.x=T,
    by='obs'
  ) %>%
  merge(
    valid(tests_yg2,gaokai_rate=1.001,h=h.) %>% as.data.frame %>% select(obs,yg2=roi),all.x=T,
    by='obs'
  ) %>%
  merge(
    valid(tests_hg,gaokai_rate=1.001,h=h.) %>% as.data.frame %>% select(obs,hg=roi),all.x=T,
    by='obs'
  ) %>%
  merge(bench) %>%
  mutate(
    jg=ifelse(is.na(jg),1,jg),
    yg=ifelse(is.na(yg),1,yg),
    yg2=ifelse(is.na(yg2),1,yg2),
    hg=ifelse(is.na(hg),1,hg)
  ) %>%
  merge(risk %>% select(obs,sel),all.x=T,by='obs') %>%
  mutate(roi=ifelse(sel,yg,ifelse(obs>=20230000,hg,jg))) %>%
  select(obs,buy,sell,sel,jg,yg,yg2,hg,roi,bench) 

#回撤

# sapply(5:nrow(test),function(i){
#   c(apply(test[1:5+i-5,1:3],2,max),apply(test[1:5+i-5,-1:-3],2,prod))
# }) %>%
#   t() %>%
#   as.data.table %>%
#   melt(id=1:3) %>%
#   ggplot() + 
#   geom_boxplot(aes(x=variable,y=value))

#Validation

test %>%
  mutate(roi=ifelse(sel,yg,roi)) %>%
  group_by(obs=floor(obs/10000)) %>%
  summarise(
    miss=mean(is.na(roi)),Niu=sum(yg==roi,na.rm=T)/n(),
    JinGou=prod(jg,na.rm=T),YinGou=prod(yg,na.rm=T),YinGou2=prod(yg2,na.rm=T),HongGou=prod(hg,na.rm=T),DualGou=prod(roi,na.rm=T),
    Bench=prod(bench,na.rm=T)
  ) %>%
  as.data.frame() %>%
  tail(12)

test %>%
  mutate(roi=ifelse(roi==yg,yg,1)) %>%
  melt(id=c('obs','buy','sell')) %>%
  group_by(obs=floor(obs/100)*100+1,variable) %>%
  summarise(value=prod(value)) %>%
  filter(variable%in%c('jg','yg','hg','roi')) %>%
  mutate(obs=as.Date(strptime(obs,format='%Y%m%d'))) %>%
  ggplot() +
  geom_line(aes(x=obs,y=value,colour=variable))

data.table(test[,1:3],apply(test[,-1:-3],2,function(x){
  # ifelse(is.na(x),1,x)
  cumprod(ifelse(is.na(x),1,x)) 
})) %>%
  select(obs,buy,sell,JinGou=jg,YinGou=yg,HongGou=hg,DualGou=roi,Benchmark=bench) %>%
  melt(id=1:3) %>%
  # mutate(value=log(value)) %>%
  mutate(obs=as.Date(strptime(obs,format='%Y%m%d'))) %>%
  ggplot() +
  geom_line(aes(x=obs,y=value,colour=variable)) +
  labs(x='obs',y='Value',colour='Strategy') +
  theme_bw()

################################################################################
#Parametering
#Risk

darisk2 <- function(){
  
  k=2
  ncode=50
  
  mat <- close[k:nrow(close),] / open[1:(nrow(open)-k+1),]
  # mat <- close[k:nrow(close),] / rbind(NA,close[1:(nrow(open)-k),])
  
  valt <- sapply(k:nrow(close),function(i){
    x <- colSums(val[1:k+i-k,],na.rm=T)
    x>=quantile(x,1-ncode/length(x),na.rm=T)
  }) %>% t
  
  risk <- data.table(
    obs=as.numeric(rownames(mat)),
    t(
      sapply(1:nrow(valt),function(i){
        c(
          roi50=mean(mat[i,valt[i,]],na.rm=T),
          win50=mean(mat[i,valt[i,]]>1,na.rm=T)
        )
      })
    )
  )
  
  risk$sroi50 <- c(NA,sapply(2:nrow(risk),function(i){prod(risk$roi50[-1:0+i])}))
  risk$swin50 <- c(NA,sapply(2:nrow(risk),function(i){mean(risk$win50[-1:0+i])}))
  
  risk
}

risk <- darisk2()

#Select Dog

h. <- 5

test <- obsmap %>%
  filter(obs>=start_date) %>%
  merge(
    valid(tests_jg,gaokai_rate=1.001,h=h.) %>% as.data.frame %>% select(obs,jg=roi),all.x=T,
    by='obs'
  ) %>%
  merge(
    valid(tests_yg,gaokai_rate=1.001,h=h.) %>% as.data.frame %>% select(obs,yg=roi),all.x=T,
    by='obs'
  ) %>%
  merge(bench) %>%
  merge(risk,all.x=T,by='obs') %>%
  mutate(
    jg=ifelse(is.na(jg),1,jg),
    yg=ifelse(is.na(yg),1,yg)
  ) %>%
  mutate(
    sel1=win50>=0.6,
    sel2=sroi50>1.02,
    sel3=roi50<1
  ) %>%
  mutate(sel=(sel1)|(sel2&sel3)) %>%
  mutate(roi=ifelse(sel,yg,jg))

test %>%
  group_by(sel1,sel2&sel3) %>%
  summarise(n(),mean(yg)-1,mean(jg)-1)

test %>%
  summarise(prod(jg),prod(yg),prod(roi))

#Validation

test %>%
  select(obs,buy,sell,jg,yg,bench,roi) %>%
  # filter(obs>=20240101) %>%
  group_by(obs=floor(obs/1000000)) %>%
  summarise(
    miss=mean(is.na(roi)),Niu=sum(yg==roi,na.rm=T)/n(),
    JinGou=prod(jg,na.rm=T),YinGou=prod(yg,na.rm=T),DualGou=prod(roi,na.rm=T),
    Bench=prod(bench,na.rm=T)
  ) %>%
  as.data.frame() 

test <- test %>%
  select(obs,buy,sell,jg,yg,bench,roi)

data.table(test[,1:4],apply(test[,-1:-4],2,function(x){
  cumprod(ifelse(is.na(x),1,x))
})) %>%
  select(obs,buy,sell,JinGou=jg,YinGou=yg,DualGou=roi,Benchmark=bench) %>%
  melt(id=1:3) %>%
  # mutate(value=log(value)) %>%
  mutate(obs=as.Date(strptime(obs,format='%Y%m%d'))) %>%
  ggplot() +
  geom_line(aes(x=obs,y=value,colour=variable)) +
  labs(x='obs',y='Value',colour='Strategy') +
  theme_bw()
