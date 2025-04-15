
rm(list=ls())
library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(stringr)
library(purrr)
library(rpart)

ak <- import('akshare')

setwd('/Users/huzixin/Documents/goldgou/')

########################################################################################################################
# 下载东财数据
########################################################################################################################

updatedata <- function(start_date){
  
  start_date <- 20180101
  
  allcodes <- rbind(
    ak$stock_sh_a_spot_em(),
    ak$stock_sz_a_spot_em()
  ) %>%
    select(code=2,name=3) %>%
    unique()
  write.csv(allcodes,'data/allcodes.csv')
  
  #Get His Data
  
  getcodei <- function(codei){
    x <- ak$stock_zh_a_hist(symbol=codei,
                            period='daily',
                            start_date=start_date,
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

#Full Data

# system.time(updatedata(20180101))
# updateusdata()

################################################################################
#导入原始数据
################################################################################

#纪要数据

jglist <- fread('data/jglist_his.csv')[,-1] %>%
  rbind(fread('data/jglist_new.csv')[,-1]) %>%
  mutate(date=as.numeric(gsub('-','',date)),time=as.numeric(gsub(':','',time))) %>%
  filter(!code%in%c(30024,688981,858,300152),!grepl('证券|ST',name)) %>%
  unique()

#东财数据

allcodes <- fread('data/allcodes.csv')[,-1]

ddata <- fread('data/hisdata.csv')[,-1] %>%
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

#整合数据

ddata <- ddata %>% filter(code%in%jglist$code)
jglist <- jglist %>% filter(code%in%ddata$code)

jglist <- jglist %>%
  merge(
    jglist %>%
      group_by(jgid) %>%
      summarise(vote=n())    
  ) %>%
  mutate(vote=1/vote)

#构建宽表

close <- ddata %>% acast(date~code,value.var='close')
open <- ddata %>% acast(date~code,value.var='open')
val <- ddata %>% acast(date~code,value.var='val')
high <- ddata %>% acast(date~code,value.var='high')
low <- ddata %>% acast(date~code,value.var='low')
val[is.na(val)] <- 0
close[close<0] <- NA
open[open<0] <- NA
roi <- close[-1,]/open[-nrow(open),]

#处理机构数据

jglist <- jglist %>%
  mutate(date2=ifelse(time<=915,date-1,date))

jglist$obs <- sapply(jglist$date2,function(i){
  max(obsmap$obs[obsmap$obs<=i])
})

jglist <- jglist %>% select(-date2)

jglist2 <- jglist %>%
  group_by(obs,code) %>%
  summarise(vote=sum(vote)) %>%
  acast(obs~code,value.var='vote',fill=0)

jg<- t(t(jglist2)/rowSums(jglist2))

################################################################################
#每日选票
################################################################################

#导入模型

dir('result',pattern='models')
system.time(load('result/models_250415.rda'))

#昨天买什么

list1 <- function(){
  
  onite1 <- data.table(
    code=as.numeric(colnames(close)),
    onite0=open[nrow(open),]/close[nrow(close)-1,]
  )
  
  xidx <- rev(rownames(close))[1:20+1]
  xopen <- open[xidx[1:5],,drop=F] %>% t
  colnames(xopen) <- paste0('open',1:5)
  xclose <- close[xidx[1:5],,drop=F] %>% t
  colnames(xclose) <- paste0('close',1:5)
  xlow <- low[xidx[1:5],,drop=F] %>% t
  colnames(xlow) <- paste0('low',1:5)
  xhigh <- high[xidx[1:5],,drop=F] %>% t
  colnames(xhigh) <- paste0('high',1:5)
  xval <- val[xidx[1:5],,drop=F] %>% t
  colnames(xval) <- paste0('val',1:5)
  xjg <- jg[xidx[1:5],,drop=F] %>% t
  colnames(xjg) <- paste0('jg',1:5)
  
  X <- data.table(
    code=as.numeric(rownames(xopen)),
    obs=max(xidx),
    xopen,xclose,xlow,xhigh,xval,xjg,
    jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T)
  ) %>%
    merge(
      onite1 %>% mutate(code=as.numeric(code))
    ) %>%
    mutate(open0=close1*onite0) %>%
    select(-onite0)
  
  X <- X %>%
    merge(
      X %>%
        group_by(obs) %>%
        summarise(
          btoday1=mean(close1/open1,na.rm=T), 
          bonite0=mean(open0/close1,na.rm=T),
          broi1=mean(close1/open2,na.rm=T),
          broi5=mean(close1/open5,na.rm=T)
        ) %>%
        mutate(
          class=(btoday1>1)*1000+(bonite0>1)*100+(broi1>1)*10+(broi5>1)*1
        ),
      by='obs'
    ) %>%
    mutate(
      x1=open1/open0,
      x2=close1/open0,
      x3=high1/open0,
      x4=low1/open0,
      x5=jg1,
      x6=jg2,
      x7=jg3+jg4+jg5,
      x8=jg10,
      x9=close1/(close2+close3+close4+close5)*4
    ) %>%
    mutate(x10=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098)) %>%
    select(obs,code,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,btoday1,bonite0,broi1,broi5,class)
  
  k <- match(unique(X$class),classlist)
  
  system.time(
    rlti <- lapply(1:length(models_xjj[[k]]),function(i){
      modeli1 <- models_xjj[[k]][[i]]
      modeli2 <- models_idx[[k]][[i]]
      out1 <- predict(modeli1,newdata=X)
      out2 <- predict(modeli2,newdata=X)
      data.table(i=i,id=1:length(out1),pxjj=out1,pidx=out2)
    }) %>%
      rbindlist %>%
      group_by(id) %>%
      summarise(mxjj=mean(pxjj),midx=mean(pidx),sxjj=sd(pxjj),sidx=sd(pidx),wxjj=mean(pxjj>1),widx=mean(pidx>1))
  )
  
  out <- data.table(X[,1:2],rlti) %>%
    merge(
      data.table(
        code=as.numeric(colnames(close)),
        close1=close[nrow(close)-1,],
        open0=open[nrow(open),],
        close0=close[nrow(close),]
      ),
      by='code'
    ) %>%
    merge(
      allcodes,by='code'
    ) %>%
    filter(mxjj>1,open0<close1,widx>0.6,wxjj>0.6) %>%
    arrange(desc(midx)) %>%
    mutate(roi=close0/open0) %>%
    select(obs,code,name,pred_xjj=mxjj,pred_idx=midx,win_xjj=wxjj,win_idx=widx,price_in_qfq=open0,roi) %>%
    head(20)
  
  out$cumroi <- cummean(out$roi)

  out
  
}

list0 <- function(){
  
  onite0 <- rbind(
    ak$stock_sh_a_spot_em(),
    ak$stock_sz_a_spot_em()
  ) %>%
    select(
      code=`代码`,
      name=`名称`,
      open0=`今开`,
      close1=`昨收`
    ) %>%
    mutate(
      onite0=open0/close1
    ) %>%
    select(code,name,onite0,open0_hfq=open0,close1_hfq=close1) %>%
    unique() %>%
    mutate(onite0=ifelse(is.na(onite0),1,onite0))
  
  #做模型输入
  
  xidx <- rev(rownames(close))[1:20]
  xopen <- open[xidx[1:5],,drop=F] %>% t
  colnames(xopen) <- paste0('open',1:5)
  xclose <- close[xidx[1:5],,drop=F] %>% t
  colnames(xclose) <- paste0('close',1:5)
  xlow <- low[xidx[1:5],,drop=F] %>% t
  colnames(xlow) <- paste0('low',1:5)
  xhigh <- high[xidx[1:5],,drop=F] %>% t
  colnames(xhigh) <- paste0('high',1:5)
  xval <- val[xidx[1:5],,drop=F] %>% t
  colnames(xval) <- paste0('val',1:5)
  xjg <- jg[xidx[1:5],,drop=F] %>% t
  colnames(xjg) <- paste0('jg',1:5)
  
  X <- data.table(
    code=as.numeric(rownames(xopen)),
    obs=max(xidx),
    xopen,xclose,xlow,xhigh,xval,xjg,
    jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T)
  ) %>%
    merge(
      onite0 %>% mutate(code=as.numeric(code))
    ) %>%
    mutate(open0=close1*onite0) %>%
    select(-onite0,-name)
  
  X <- X %>%
    merge(
      X %>%
        group_by(obs) %>%
        summarise(
          btoday1=mean(close1/open1,na.rm=T), 
          bonite0=mean(open0/close1,na.rm=T),
          broi1=mean(close1/open2,na.rm=T),
          broi5=mean(close1/open5,na.rm=T)
        ) %>%
        mutate(
          class=(btoday1>1)*1000+(bonite0>1)*100+(broi1>1)*10+(broi5>1)*1
        ),
      by='obs'
    ) %>%
    mutate(
      x1=open1/open0,
      x2=close1/open0,
      x3=high1/open0,
      x4=low1/open0,
      x5=jg1,
      x6=jg2,
      x7=jg3+jg4+jg5,
      x8=jg10,
      x9=close1/(close2+close3+close4+close5)*4
    ) %>%
    mutate(x10=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098)) %>%
    select(obs,code,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,btoday1,bonite0,broi1,broi5,class)
  
  #输出结果
  
  k <- match(unique(X$class),classlist)
  
  system.time(
    rlti <- lapply(1:length(models_xjj[[k]]),function(i){
      modeli1 <- models_xjj[[k]][[i]]
      modeli2 <- models_idx[[k]][[i]]
      out1 <- predict(modeli1,newdata=X)
      out2 <- predict(modeli2,newdata=X)
      data.table(i=i,id=1:length(out1),pxjj=out1,pidx=out2)
    }) %>%
      rbindlist %>%
      group_by(id) %>%
      summarise(mxjj=mean(pxjj),midx=mean(pidx),sxjj=sd(pxjj),sidx=sd(pidx),wxjj=mean(pxjj>1),widx=mean(pidx>1))
  )
  
  list(
    threshold.5 = data.table(X[,1:2],rlti) %>%
      merge(
        data.table(
          code=as.numeric(colnames(close)),
          close1=close[nrow(close)-1,]
        ),
        by='code') %>%
      merge(
        onite0 %>% mutate(code=as.numeric(code)),by='code'
      ) %>%
      mutate(open0=close1*onite0) %>%
      filter(mxjj>1,open0<=close1,widx>0.5,wxjj>0.5) %>%
      arrange(desc(midx)) %>%
      select(obs,code,name,pred_xjj=mxjj,pred_idx=midx,win_xjj=wxjj,win_idx=widx,price_in=open0_hfq) %>%
      head(20),
    threshold.6 = data.table(X[,1:2],rlti) %>%
      merge(
        data.table(
          code=as.numeric(colnames(close)),
          close1=close[nrow(close)-1,]
        ),
        by='code') %>%
      merge(
        onite0 %>% mutate(code=as.numeric(code)),by='code'
      ) %>%
      mutate(open0=close1*onite0) %>%
      filter(mxjj>1,open0<=close1,widx>0.6,wxjj>0.6) %>%
      arrange(desc(midx)) %>%
      head(20)
  )
  
}

list1()

list0()



################################################################################
#训练模型
################################################################################

#建模数据

system.time(
  
  rawdata <- lapply(sort(unique(jglist$date)[which(unique(jglist$date) %in% obsmap$obs)])[-1:-100],function(buyi){
    
    # buyi <- 20241112
    buyi <- paste(buyi)
    # print(buyi)
    
    #确定观测市场
    
    i <- which(rownames(close)==buyi)
    yidx <- i+1:20
    xidx <- i-1:20
    yidx <- yidx[yidx<=nrow(close)]
    
    yidx <- rownames(close)[yidx]
    xidx <- rownames(close)[xidx]
    
    #X
    
    xopen <- open[xidx[1:5],,drop=F] %>% t
    colnames(xopen) <- paste0('open',1:5)
    xclose <- close[xidx[1:5],,drop=F] %>% t
    colnames(xclose) <- paste0('close',1:5)
    xlow <- low[xidx[1:5],,drop=F] %>% t
    colnames(xlow) <- paste0('low',1:5)
    xhigh <- high[xidx[1:5],,drop=F] %>% t
    colnames(xhigh) <- paste0('high',1:5)
    xval <- val[xidx[1:5],,drop=F] %>% t
    colnames(xval) <- paste0('val',1:5)
    xjg <- jg[xidx[1:5],,drop=F] %>% t
    colnames(xjg) <- paste0('jg',1:5)
    
    #Y
    
    if(length(yidx)==0){
      data.table(
        code=as.numeric(rownames(xopen)),
        buy=as.numeric(buyi),
        xopen,xclose,xlow,xhigh,xval,xjg,
        jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T),
        open0=open[buyi,],
        price_out=NA
      ) %>%
        mutate(
          ztrate=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098),
          xjj=ifelse(open0>close1*ztrate,1,price_out/open0)
        ) 
    } else {
      data.table(
        code=as.numeric(rownames(xopen)),
        buy=as.numeric(buyi),
        xopen,xclose,xlow,xhigh,xval,xjg,
        jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T),
        open0=open[buyi,],
        price_out=close[yidx[1],]
      ) %>%
        mutate(
          ztrate=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098),
          xjj=ifelse(open0>close1*ztrate,1,price_out/open0)
        ) 
    }
    
  }) %>%
    rbindlist()
  
) 

rawdata <- rawdata %>%
  merge(
    rawdata %>%
      group_by(buy) %>%
      summarise(
        btoday1=mean(close1/open1,na.rm=T), 
        bonite0=mean(open0/close1,na.rm=T),
        broi1=mean(close1/open2,na.rm=T),
        broi5=mean(close1/open5,na.rm=T),
        bxjj=mean(xjj,na.rm=T)
      )
  ) %>%
  mutate(
    class=(btoday1>1)*1000+(bonite0>1)*100+(broi1>1)*10+(broi5>1)*1
  ) %>%
  mutate(idx=xjj/bxjj)

datasets <- rawdata[rowSums(is.na(rawdata))==0,]

#根据分类拆分数据

mfiles <- lapply(unique(datasets$class),function(i){
  datasets %>%
    filter(class==i) %>%
    mutate(
      x1=open1/open0,
      x2=close1/open0,
      x3=high1/open0,
      x4=low1/open0,
      x5=jg1,
      x6=jg2,
      x7=jg3+jg4+jg5,
      x8=jg10,
      x9=close1/(close2+close3+close4+close5)*4,
      x10=ztrate,
      y=idx
    ) %>%
    select(buy,code,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y,xjj,btoday1,bonite0,broi1,broi5,class)
})

#设定抽样随机树，训练模型

set.seed(1); seeds <- sample(1:10000,300) #设定数据随机数

system.time(
  models_idx <- lapply(mfiles,function(mfile){
    print(Sys.time())
    lapply(1:length(seeds),function(i){
      set.seed(seeds[i])
      sel <- sample(1:nrow(mfile),10000)
      modeli <- rpart(y ~ ., 
                      data = mfile[sel,] %>% select(-buy,-code,-xjj,-class), 
                      method = "anova", 
                      control = rpart.control(cp = 0.001))
      modeli
    })
  })
)

system.time(
  models_xjj <- lapply(mfiles,function(mfile){
    print(Sys.time())
    lapply(1:length(seeds),function(i){
      set.seed(seeds[i])
      sel <- sample(1:nrow(mfile),10000)
      modeli <- rpart(xjj ~ ., 
                      data = mfile[sel,] %>% select(-buy,-code,-y,-class), 
                      method = "anova", 
                      control = rpart.control(cp = 0.001))
      modeli
    })
  })
)

classlist <- unique(datasets$class)

system.time(save(models_idx,models_xjj,classlist,file='result/models_test.rda'))

################################################################
#回测
################################################################

system.time(
  tests <- lapply(1:length(models_xjj),function(k){
    print(paste(k,Sys.time()))
    mfile <- mfiles[[k]]
    rlti <- lapply(1:length(seeds),function(i){
      modeli1 <- models_xjj[[k]][[i]]
      modeli2 <- models_idx[[k]][[i]]
      set.seed(seeds[i])
      sel <- sample(1:nrow(mfile),10000)
      out1 <- predict(modeli1,newdata=mfile)
      out2 <- predict(modeli2,newdata=mfile)
      out2[sel] <- out1[sel] <- NA
      data.table(i=i,id=1:length(out1),pxjj=out1,pidx=out2)
    }) %>% rbindlist()
    rlti <- rlti %>%
      filter(!is.na(pidx)) %>%
      group_by(id) %>%
      summarise(mxjj=mean(pxjj),midx=mean(pidx),sxjj=sd(pxjj),sidx=sd(pidx),wxjj=mean(pxjj>1),widx=mean(pidx>1))
    data.table(mfile,rlti[,-1])
  })
)

# write.csv(rbindlist(tests),'result/test_20150415.csv',row.names=F)

rbindlist(tests) %>%
  group_by(class) %>%
  summarise(cor(y,midx),cor(xjj,mxjj))

rbindlist(tests) %>%
  mutate(xjj=(xjj-1)/2+1) %>%
  group_by(class,buy) %>%
  summarise(benchroi=mean(xjj)) %>%
  merge(
    rbindlist(tests) %>%
      mutate(xjj=(xjj-1)/2+1) %>%
      filter(mxjj>1,x2>=1,wxjj>0.6,widx>0.6) %>%
      arrange(desc(midx)) %>%
      group_by(buy) %>%
      summarise(roi=mean(head(xjj))),
    all.x=T
  ) %>%
  mutate(roi=ifelse(is.na(roi),1,roi)) %>%
  group_by(floor(buy/100)) %>% 
  summarise(win=mean(roi>benchroi),bench=prod(benchroi),roi=prod(roi),n=n()) %>%
  mutate(idx=roi/bench) %>%
  as.data.frame()

rbindlist(tests) %>%
  mutate(xjj=(xjj-1)/2+1) %>%
  filter(mxjj>1,wxjj>0.6,widx>0.5) %>%
  arrange(buy,desc(midx)) %>%
  group_by(buy,class) %>%
  summarise(
    roi=mean(head(xjj)),
    code=paste(head(paste0(code,'(',round(xjj,2),')')),collapse=',')
  ) %>%
  tail()

