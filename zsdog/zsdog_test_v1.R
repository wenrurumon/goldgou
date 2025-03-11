
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(stringr)
library(purrr)

setwd('/Users/huzixin/Documents/goldgou/')

################################################################################
#Historical Data
################################################################################

rawjglist <- jglist <- fread('data/jglist.csv')[,-1] %>%
  mutate(
    buy=as.numeric(gsub('-','',buy)),
    date=as.numeric(gsub('-','',date)),
    time=floor(as.numeric(gsub(':','',time))/100)
  ) %>%
  filter(time<=915|buy>date)

jglist <- jglist %>%
  merge(
    jglist %>%
      group_by(jgid) %>%
      summarise(idx=n())    
  ) %>%
  mutate(idx=1/idx) %>%
  group_by(buy,code,name) %>%
  summarise(idx=sum(idx))

ddata <- fread('data/exedata.csv')[,-1] %>%
  select(date=日期,code,open=开盘,close=收盘,high=最高,low=最低,val=成交额,vol=成交量) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  unique()

datemap <- sort(unique(ddata$date))
datemap <- data.table(date=datemap,did=1:length(datemap))

obsmap <- data.table(
  obs=datemap$date,
  buy=c(datemap$date[-1],NA),
  sell=c(datemap$date[-1:-2],rep(NA,2))
) 

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

jglist <- jglist %>% filter(paste(code)%in%colnames(close))

################################################################################
#Xuanze Gou
################################################################################

system.time(
  
  test <- lapply(unique(jglist$buy),function(buyi){
    
    # buyi <- 20240925
    print(buyi)
    i <- which(rownames(close)==buyi)
    
    jgpi <- jglist %>% filter(buy==buyi) %>% mutate(code=paste(code))
    
    #纪要狗
    
    jgpi1 <- jglist %>% filter(buy==as.numeric(rownames(close)[i-1])) %>% mutate(code=paste(code))
    
    jgps <- jgpi %>%
      as.data.frame %>%
      select(code,name,idx0=idx) %>%
      merge(
        jgpi1 %>%
          as.data.frame %>%
          select(code,name,idx1=idx),
        all=T
      ) %>%
      as.data.table %>%
      mutate(
        idx0=ifelse(is.na(idx0),0,idx0),
        idx1=ifelse(is.na(idx1),0,idx1)
      ) %>%
      mutate(didx=idx0-idx1) %>%
      arrange(desc(didx),desc(idx0)) %>%
      select(code,idx0,idx1,didx)
    
    #标签狗
    
    hold.label <- 1
    code0 <- names(which(open[i,]/close[i-1,]>1.05))
    roi.label <- close[i-1:50,jgpi$code,drop=F]/open[i-1:50-hold.label,jgpi$code,drop=F]
    roi.label[is.na(roi.label)] <- min(mean(roi.label,na.rm=T),1)
    onite.label <- open[i-1:50-hold.label,code0,drop=F]/close[i-1:50-1-hold.label,code0,drop=F]
    rownames(roi.label) <- rownames(onite.label) <- NULL
    
    matlabel <- melt(onite.label) %>%
      select(did=1,code0=2,onite=3) %>%
      filter(onite>=1.05) %>%
      merge(
        melt(roi.label) %>%
          select(did=1,code1=2,roi=3)
      ) %>%
      group_by(code0,code1) %>%
      summarise(n=n(),win=mean(roi>1),roi=mean(roi)) %>%
      filter(n>3) %>%
      merge(
        melt(roi.label) %>%
          select(did=1,code1=2,roi=3) %>%
          group_by(code1) %>%
          summarise(refroi=mean(roi))
      ) %>%
      mutate(uplift=roi/refroi) %>%
      arrange(desc(uplift)) 
    
    matlabel <- matlabel %>%
      group_by(code=paste(code1)) %>%
      summarise(lwin=mean(uplift>=1),luplift=mean(uplift),lroi=mean(roi)) %>%
      arrange(desc(luplift))
    
    #银狗
    
    yg <- data.table(
      code=jgpi$code,
      open0=open[i,jgpi$code],
      open1=open[i-1,jgpi$code],
      close1=close[i-1,jgpi$code],
      close2=close[i-2,jgpi$code],
      val1=val[i-1,jgpi$code],
      val2=val[i-2,jgpi$code],
      close5=colMeans(close[i-1:5,jgpi$code,drop=F],na.rm=T),
      close10=colMeans(close[i-1:10,jgpi$code,drop=F],na.rm=T),
      val5=colMeans(val[i-1:5,jgpi$code,drop=F],na.rm=T),
      val10=colMeans(val[i-1:10,jgpi$code,drop=F],na.rm=T),
      close0=close[i,jgpi$code],
      high1=high[i-1,jgpi$code],
      low1=low[i-1,jgpi$code]
    ) %>%
      mutate(
        zt=ifelse(floor(as.numeric(code)/10000)%in%c(30,68),1.198,1.098)
      )
    
    #算回报率
    
    Y <- data.table(
      code=jgpi$code,
      name=jgpi$name,
      buy=jgpi$buy,
      close00=close[i-1,jgpi$code],
      open0=open[i,jgpi$code],
      close0=close[i,jgpi$code],
      close1=rbind(close,NA)[i+1,jgpi$code],
      open2=rbind(open,NA,NA)[i+2,jgpi$code],
      high0=high[i,jgpi$code],
      low0=low[i,jgpi$code],
      high1=rbind(high,NA)[i+1,jgpi$code],
      low1=rbind(low,NA)[i+1,jgpi$code]
    ) %>%
      mutate(zt=ifelse(floor(as.numeric(code)/10000)%in%c(30,68),1.198,1.098)) %>%
      mutate(
        roi=ifelse(
          (high0==open0)&(open0==low0)&(open0>close00),
          1,#如果买日横盘涨停买不进
          ifelse(
            (high1==close1)&(high1==low1)&(close1<close0),
            open2/open0,#如果卖日横盘跌停卖不出
            ifelse(
              high1/close0>=zt,
              close0*zt/open0,#如果卖日触发涨停直接卖
              close1/open0
            )
          )
        )
      ) %>%
      select(code,name,buy,roi)
    
    #结果
    
    X <- yg %>%
      merge(jgps,all.x=T) %>%
      merge(matlabel,all.x=T)
    
    list(X=X,Y=Y)
    
  })
  
)

valid <- lapply(test,function(x){
  
  # x <- test[[floor(length(test)/1)]]
  h <- 3
  
  X <- x$X
  Y <- x$Y
  
  rlt <- list()
  
  #yingou
  sel <- X %>%
    filter(high1/close2>=zt,close1>open1) %>%
    arrange(val1) %>%
    head(h)
  rlt[[length(rlt)+1]] <- data.table(
    s = length(rlt)+1,
    Y %>% filter(code%in%sel$code)
  )
  
  #yingou2
  sel <- X %>%
    filter(close1/close2>=zt,close1>open1) %>%
    arrange(val1) %>%
    head(h)
  rlt[[length(rlt)+1]] <- data.table(
    s = length(rlt)+1,
    Y %>% filter(code%in%sel$code)
  )
  
  #yingou3
  sel <- X %>%
    filter(close1/close2>=zt,close1>open1,open0<=close1) %>%
    arrange(val1) %>%
    head(h)
  rlt[[length(rlt)+1]] <- data.table(
    s = length(rlt)+1,
    Y %>% filter(code%in%sel$code)
  )
  
  #jingou
  sel <- X %>%
    filter(luplift>1,didx>0) %>%
    arrange(desc(didx),desc(luplift)) %>%
    head(h)
  rlt[[length(rlt)+1]] <- data.table(
    s = length(rlt)+1,
    Y %>% filter(code%in%sel$code)
  )
    
  rbindlist(rlt)
  
})

rbindlist(valid) %>%
  group_by(buy,s) %>%
  summarise(roi=mean(roi-1)/2+1) %>%
  group_by(s) %>%
  summarise(prod(roi,na.rm=T),mean(roi,na.rm=T),sd(roi,na.rm=T),n())

rbindlist(valid) %>%
  group_by(buy,s) %>%
  summarise(roi=mean(roi-1)/2+1) %>%
  filter(s%in%c(2,4)) %>%
  group_by(buy) %>%
  summarise(roi=max(roi)) %>%
  summarise(prod(roi,na.rm=T))

################################################################################
#Zhisun Gou
################################################################################

system.time(
  
  test <- lapply(unique(jglist$buy),function(buyi){
    
    h <- 20
    zhisun <- 0.92
    
    print(buyi)
    jgpi <- jglist %>% filter(buy==buyi) %>% mutate(code=paste(code))
    i <- which(rownames(close)==buyi)
    
    #标签
    
    hold.label <- 2
    code0 <- names(which(open[i,]/close[i-1,]>1.05))
    roi.label <- close[i-1:50,jgpi$code,drop=F]/open[i-1:50-hold.label,jgpi$code,drop=F]
    roi.label[is.na(roi.label)] <- min(mean(roi.label,na.rm=T),1)
    onite.label <- open[i-1:50-hold.label,code0,drop=F]/close[i-1:50-1-hold.label,code0,drop=F]
    rownames(roi.label) <- rownames(onite.label) <- NULL
    
    matlabel <- melt(onite.label) %>%
      select(did=1,code0=2,onite=3) %>%
      filter(onite>=1.05) %>%
      merge(
        melt(roi.label) %>%
          select(did=1,code1=2,roi=3)
      ) %>%
      group_by(code0,code1) %>%
      summarise(n=n(),win=mean(roi>1),roi=mean(roi)) %>%
      filter(n>3) %>%
      merge(
        melt(roi.label) %>%
          select(did=1,code1=2,roi=3) %>%
          group_by(code1) %>%
          summarise(refroi=mean(roi))
      ) %>%
      mutate(uplift=roi/refroi) %>%
      arrange(desc(uplift)) 
    
    matlabel <- matlabel %>%
      group_by(code=paste(code1)) %>%
      summarise(lwin=mean(uplift>=1),luplift=mean(uplift),lroi=mean(roi)) %>%
      arrange(desc(luplift))
    
    #纪要数量变化
    
    jgpi1 <- jglist %>% filter(buy==as.numeric(rownames(close)[i-1])) %>% mutate(code=paste(code))
    
    jgps <- jgpi %>%
      as.data.frame %>%
      select(code,name,idx0=idx) %>%
      merge(
        jgpi1 %>%
          as.data.frame %>%
          select(code,name,idx1=idx),
        all=T
      ) %>%
      as.data.table %>%
      mutate(
        idx0=ifelse(is.na(idx0),0,idx0),
        idx1=ifelse(is.na(idx1),0,idx1)
      ) %>%
      mutate(didx=idx0-idx1) %>%
      arrange(desc(didx))
    
    #银狗
    
    X <- data.table(
      jgpi,
      open0=open[i,jgpi$code],
      open1=open[i-1,jgpi$code],
      close1=close[i-1,jgpi$code],
      close2=close[i-2,jgpi$code],
      val1=val[i-1,jgpi$code],
      val2=val[i-2,jgpi$code],
      close5=colMeans(close[i-1:5,jgpi$code,drop=F],na.rm=T),
      close10=colMeans(close[i-1:10,jgpi$code,drop=F],na.rm=T),
      val5=colMeans(val[i-1:5,jgpi$code,drop=F],na.rm=T),
      val10=colMeans(val[i-1:10,jgpi$code,drop=F],na.rm=T),
      close0=close[i,jgpi$code],
      high0=high[i,jgpi$code],
      low0=low[i,jgpi$code]
    ) %>%
      merge(matlabel) %>%
      merge(jgps %>% select(code,didx)) %>%
      arrange(as.numeric(code))
    
    #算回报率
    
    yidx <- i+1:h 
    yidx <- yidx[yidx<=nrow(close)] # yidx是一个卖掉日期的索引
    highi <- high[yidx-1,jgpi$code,drop=F] #每日关注前一日的最高价
    highi[,] <- apply(highi,2,cummax) #每日关注从买入日至昨天的最高价
    lowi <- low[yidx,jgpi$code,drop=F] #卖掉日期的最低价
    holdi <- apply(lowi/highi <= zhisun,2,function(x){min(which(x))}) #如果观测日最低价小于累计最高价止损值则卖，不然持有
    closei <- close[max(yidx),jgpi$code,drop=F] #整个观测索引的最后一个收盘价
    selli <- highi * zhisun #假设是以前一日最高价的止损点卖出的
    # highi0 <- high[yidx-1,jgpi$code,drop=F]
    openi0 <- open[yidx,jgpi$code,drop=F]
    selli <- ifelse(openi0>selli,selli,openi0)#如果当天开盘价不足止损价，则以当天开盘价卖掉
    selli <- sapply(1:length(holdi),function(j){
      if(holdi[j]==Inf){
        closei[j]
      } else {
        selli[holdi[j],j]
      }
    })
    holdi[holdi==Inf] <- length(yidx)
    
    Y <- data.table(
      jgpi,
      hold=holdi,
      roi=selli/open[i,jgpi$code],
      xjj=close[min(nrow(close),i+1),jgpi$code]/open[i,jgpi$code]
    ) %>%
      mutate(rate=roi^(1/hold)) %>%
      arrange(as.numeric(code))
    
    Y$xjj[(X$open0==X$low0)&(X$open0==X$high0)&(X$open0>X$close1)] <- 1
    
    #结果
    
    list(X=X,Y=Y)
    
  })
  
)

#给一个策略测每天

valid <- lapply(test,function(x){
  
  X <- x$X
  Y <- x$Y
  
  rbind(
    data.table(
      s=0,
      data.table(X,Y[,-1:-4]) %>%
        select(buy,code,hold,rate,xjj)
    ),
    #1. 选luplift最高的三个票
    data.table(
      s=1,
      data.table(X,Y[,-1:-4]) %>%
        arrange(desc(luplift)) %>%
        head(3) %>%
        select(buy,code,hold,rate,xjj)
    ),
    #2. 选idx最高的三个票
    data.table(
      s=2,
      data.table(X,Y[,-1:-4]) %>%
        arrange(desc(idx)) %>%
        head(3) %>%
        select(buy,code,hold,rate,xjj)
    ),
    #3. 选didx最高的三个票
    data.table(
      s=3,
      data.table(X,Y[,-1:-4]) %>%
        arrange(desc(didx)) %>%
        head(3) %>%
        select(buy,code,hold,rate,xjj)
    ),
    #4. 热度提升
    data.table(
      s=4,
      data.table(X,Y[,-1:-4]) %>%
        filter(val1>val5,val5>val10) %>%
        arrange(desc(luplift)) %>%
        head(3) %>%
        select(buy,code,hold,rate,xjj)
    ),
    #5. 热度小提升
    data.table(
      s=5,
      data.table(X,Y[,-1:-4]) %>%
        filter(val1>val5,val5>val10,didx>0) %>%
        arrange(desc(luplift)) %>%
        head(3) %>%
        select(buy,code,hold,rate,xjj)
    )
  )
})

rbindlist(valid[-length(valid)]) %>%
  group_by(buy,s) %>%
  summarise(zsg=mean(rate,na.rm=T),xjj=mean(xjj,na.rm=T)) %>%
  group_by(s) %>%
  summarise(prod(zsg),prod(xjj))

apply(
  rbindlist(valid) %>%
    group_by(buy,s) %>%
    summarise(zsg=mean(rate,na.rm=T),xjj=mean(xjj,na.rm=T)) %>%
    melt(id=1:2) %>%
    filter(!is.na(buy)) %>%
    acast(buy~s+variable,value.var='value'),
  2,cumprod
) %>%
  melt() %>%
  mutate(date=as.Date(paste0(substr(Var1,1,4),'-',substr(Var1,5,6),'-',substr(Var1,7,8)))) %>%
  mutate(Var2=paste(Var2)) %>%
  ggplot() + 
  geom_line(
    aes(
      x=date,
      y=value,
      colour=substr(Var2,1,2),
      linetype=substr(Var2,3,nchar(Var2))
      )
  )

rbindlist(valid) %>%
  group_by(buy,s) %>%
  summarise(zsg=mean(rate,na.rm=T),zsg2=mean(rate2,na.rm=T),xjj=mean(xjj,na.rm=T)) %>%
  melt(id=1:2) %>%
  group_by(buy=floor(buy/10000),s,variable) %>%
  summarise(value=prod(value)) %>%
  dcast(buy~variable+s,value.var='value')
