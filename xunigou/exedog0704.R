
rm(list=ls())

library(reticulate)
library(dplyr)
library(ggplot2)
library(data.table)
library(reshape2)
library(stringr)
library(purrr)
library(rpart)
library(parallel)

ak <- import('akshare')
setwd('/Users/huzixin/Documents/goldgou/')

################################################################################
# Process Sina Data
################################################################################

#导入csv数据

dir('data',pattern='sina')
paste0('data/',max(dir('data',pattern='sinadata')))
system('open data')

rawsina <- fread(paste0('data/',max(dir('data',pattern='sinadata')))) %>%
  filter(substr(code,1,2)!='bj') %>%
  select(code,date,open,close,high,low,val=amount) %>%
  mutate(code=as.numeric(substr(code,3,8))) %>%
  mutate(date=as.numeric(gsub('-','',date))) %>%
  filter(date>=20210000)

#处理交易日期

datemap <- as.numeric(gsub('-','',sapply(ak$tool_trade_date_hist_sina()$trade_date,paste)))
# datemap <- sort(unique(rawsina$date))
datemap <- data.table(date=datemap,did=1:length(datemap))

obsmap <- data.table(
  obs=datemap$date,
  buy=c(datemap$date[-1],NA),
  sell=c(datemap$date[-1:-2],rep(NA,2))
) 

#票清单

# stock_list_sina <- ak$stock_zh_a_spot()
stock_list_dc <- ak$stock_zh_a_spot_em()
stock_list <- stock_list_dc %>%
  select(code=`代码`,name=`名称`,
         new=`最新价`,close1=`昨收`,open0=`今开`,low0=`最低`,high0=`最高`,
         value=`成交额`,amount=`成交额`) #%>%
  # mutate(code=as.numeric(substr(code,3,8)),date=Sys.Date()) 
  
################################################################################
# Load Model
################################################################################

#执行模型

list1 <- function(){
  
  onite <- data.table(
    buy=as.numeric(rownames(close)[nrow(close)]),
    code=as.numeric(colnames(close)),
    roi=close[nrow(close),]/open[nrow(open),],
    open=open[nrow(open),]/close[nrow(close)-1,],
    high=high[nrow(open),]/close[nrow(close)-1,],
    low=low[nrow(open),]/close[nrow(close)-1,],
    val=val[nrow(val),]
  ) %>%
    merge(jglist %>% select(code,name) %>% unique())
  
  #开始计算新的一天的X
  
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
      onite %>% mutate(code=as.numeric(code)) %>% select(code,name,onite0=open)
    ) %>%
    mutate(open0=close1*onite0) %>%
    select(-onite0)
  
  X <- X %>%
    filter(jg10>0) %>%
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
    select(obs,code,name,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,btoday1,bonite0,broi1,broi5,class)
  
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
  
  out <- data.table(X[,1:3],rlti) %>%
    merge(
      onite %>% select(code,onite0=open,roi),by='code'
    ) %>%
    arrange(desc(midx)) %>%
    select(code,name,mxjj,midx,wxjj,widx,onite0,roi) 
  
  out <- out %>% filter(mxjj>1,onite0<=1,widx>0.5,wxjj>0.6) 
  
  out$cumroi <- cummean(out$roi)
  
  out
  
}

list0 <- function(){
  
  #更新数据
  
  if(as.numeric(gsub('-','',Sys.Date())) %in% obsmap$buy){ #今天是交易日
    
    run <- T
    
    onite <- ak$stock_zh_a_spot_em() %>%
      select(code=`代码`,name=`名称`,new=`最新价`,open=`今开`,close1=`昨收`,high=`最高`,low=`最低`,val=`成交额`) %>%
      mutate(code=as.numeric(code),roi=new/open,onite=open/close1) %>%
      select(code,name,open=onite,roi) %>%
      unique()
    
    if(as.numeric(substr(gsub('-| |:|CST','',Sys.time()),9,12))<0925){
      onite <- onite %>% mutate(open=1,roi=NA)
      run <- F
    }
    
  } else {
   
    onite <- ak$stock_zh_a_spot_em() %>%
      select(code=`代码`,name=`名称`,new=`最新价`,open=`今开`,close1=`昨收`,high=`最高`,low=`最低`,val=`成交额`) %>%
      mutate(code=as.numeric(code),roi=new/open,onite=open/close1) %>%
      select(code,name,open=onite,roi) %>%
      unique()
    
    onite <- onite %>% mutate(open=1,roi=NA)
    
    run <- F
    
  }
  
  #开始计算新的一天的X
  
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
  xjg2 <- jg2[xidx[1:5],,drop=F] %>% t
  colnames(xjg2) <- paste0('ojg',1:5)
  
  X <- data.table(
    code=as.numeric(rownames(xopen)),
    obs=max(xidx),
    xopen,xclose,xlow,xhigh,xval,xjg,xjg2,
    jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T)
  ) %>%
    merge(
      onite %>% mutate(code=as.numeric(code)) %>% select(code,name,onite0=open),all.x=T
    ) %>%
    mutate(open0=close1*onite0) %>%
    mutate(ztrate=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098)) %>%
    select(-onite0)
  
  X <- X %>%
    filter(jg10>0) %>%
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
      open1=open1/open0,
      close1=close1/open0,
      high1=high1/open0,
      low1=low1/open0,
      jg1=jg1,
      jg2=jg2,
      jg35=jg3+jg4+jg5,
      jg10=jg10,
      closegr=close1/(close2+close3+close4+close5)*4,
      ztrate=ztrate,
      onite0=1/close1
    ) %>%
    select(obs,code,open1,onite0,high1,low1,jg1,jg2,jg35,jg10,closegr,ztrate,btoday1,bonite0,broi1,broi5,class) 
  
  #输出结果
  
  if(run){
    Ks <- match(unique(X$class),classlist)
  } else {
    Ks <- match(unique(X$class)+c(0,100),classlist)  
  }
  
  if(!run){print("Dummy Today's Data")}
  
  lapply(Ks,function(k){
    
    system.time(
      rlti <- lapply(1:length(models_z[[k]]),function(i){
        modeli1 <- models_y[[k]][[i]]
        modeli2 <- models_z[[k]][[i]]
        out1 <- predict(modeli1,newdata=X)
        out2 <- predict(modeli2,newdata=X)
        data.table(
          i=i,
          X %>% 
            mutate(jg0=jg1/jg10) %>%
            select(class,obs,code,onite0,jg0),
          py=out1,pz=out2
        )
      }) %>%
        rbindlist %>%
        group_by(class,obs,code,jg0,onite0) %>%
        summarise(wy=mean(py>0),wz=mean(pz>0),py=mean(py),pz=mean(pz))
    )
    
    out <- rlti %>%
      merge(
        onite %>% select(code,name,onite0=open,roi)
      ) %>%
      mutate(obs=as.numeric(rownames(close)[nrow(close)])) %>%
      select(obs,code,name,py,pz,wy,wz,jg0,onite0,roi) %>%
      arrange(desc(jg0)) 
    
    out <- out %>% filter(py>0,pz>0,wy>0.6,wz>0.5)
    
    out$cumroi <- cummean(out$roi)
    
    out
    
  })
  
}

sell0 <- function(...){
  
  x <- c(300570,002144,002689,600595,300482,601963)
  
  onite <- rbind(
    data.table(`交易所`='sh',ak$stock_sh_a_spot_em()),
    data.table(`交易所`='sz',ak$stock_sz_a_spot_em())#,
    # data.table(`交易所`='bj',ak$stock_bj_a_spot_em())
  )  %>%
    select(code=`代码`,name=`名称`,new=`最新价`,open=`今开`,close1=`昨收`,high=`最高`,low=`最低`,val=`成交额`) %>%
    mutate(code=as.numeric(code),new=new/close1) %>%
    select(code,name,new) %>%
    unique()
  
  x.jg <- filter(jglist,code%in%x) %>%
    group_by(code,name) %>%
    summarise(lastdate=max(date)) 
    
  x.jg$llen <- sapply(x.jg$lastdate,function(x){
    sum(rownames(close)>=x)
  })
  
  x.close <- close[nrow(close)-2:0,paste(x.jg$code)]
  
  rbind(
    x.close[-1,]/x.close[-3,],
    new=onite$new[match(colnames(x.close),paste(onite$code))]
  )
  
}

################################################################################
# Process Jiyao Data
################################################################################

#导入数据

rawjg <- fread('data/jglist_his.csv')[,-1] %>%
  rbind(fread('data/jglist_new.csv')[,-1]) %>%
  mutate(date=as.numeric(gsub('-','',date)),time=as.numeric(gsub(':','',time))) %>%
  filter(!code%in%c(30024,688981,858,300152),!grepl('证券|ST|国泰海通|申万宏源|国联民生',name)) %>%
  unique()

#处理机构数据

jglist <- rawjg 

jglist <- jglist %>%
  merge(
    jglist %>%
      group_by(jgid) %>%
      summarise(vote=n())    
  ) %>%
  mutate(vote=1/vote) %>%
  mutate(date2=ifelse(time<=915,date-1,date)) 

jglist$obs <- sapply(jglist$date2,function(i){
  max(obsmap$obs[obsmap$obs<=i])
})

jglist <- jglist %>% 
  mutate(indayjy=ifelse((date2==obs)&(time>=920)&(time<=1500),1,0))

jg <- jglist %>%
  group_by(obs,code) %>%
  summarise(vote=sum(vote)) %>%
  merge(
    jglist %>%
      filter(indayjy==0) %>%
      group_by(obs,code) %>%
      summarise(onvote=sum(vote)),
    all.x=T
  ) %>%
  merge(obsmap,all.x=T) %>%
  mutate(onvote=ifelse(is.na(onvote),0,onvote)) %>%
  select(date=buy,code,vote,onvote)

jg <- jg %>%
  merge(
    jg %>%
      group_by(date) %>%
      summarise(vote2=sum(vote),onvote2=sum(onvote))
  ) %>%
  mutate(vote=vote/vote2,onvote=onvote/onvote2) %>%
  select(-vote2,-onvote2)

################################################################################
# Integrate Data
################################################################################

#构建宽表

ddata <- rawsina %>% 
  merge(jg,all.x=T) %>%
  mutate(
    vote=ifelse(is.na(vote),0,vote),
    onvote=ifelse(is.na(onvote),0,onvote)
  )

close <- ddata %>% acast(date~code,value.var='close')
open <- ddata %>% acast(date~code,value.var='open')
val <- ddata %>% acast(date~code,value.var='val')
high <- ddata %>% acast(date~code,value.var='high')
low <- ddata %>% acast(date~code,value.var='low')
jg <- ddata %>% acast(date~code,value.var='vote')
jg2 <- ddata %>% acast(date~code,value.var='onvote')

val[is.na(val)] <- 0
close[close<0] <- NA
open[open<0] <- NA

################################################################################
# Execute Trading
################################################################################

# #导入模型
# 
# dir('models',pattern='sina_2')
# # system('open models')
# system.time(load(paste0('models/',max(dir('models',pattern='sina_2')))))
# 
# #选票
# 
# # list1()
# list0()

################################################################################
#训练模型
################################################################################

#建模数据

system.time(
  
  rawdata <- lapply(sort(unique(jglist$date)[which(unique(jglist$date) %in% unique(ddata$date))])[-1:-100],function(buyi){
    
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
    xjg2 <- jg2[xidx[1:5],,drop=F] %>% t
    colnames(xjg2) <- paste0('ojg',1:5)
    
    #Y
    
    if(length(yidx)==0){
      data.table(
        code=as.numeric(rownames(xopen)),
        buy=as.numeric(buyi),
        xopen,xclose,xlow,xhigh,xval,xjg,xjg2,
        jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T),
        open0=open[buyi,],
        price_out=NA
      ) %>%
        mutate(
          ztrate=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098),
          roi=ifelse(open0>close1*ztrate,1,price_out/open0)
        ) 
    } else {
      data.table(
        code=as.numeric(rownames(xopen)),
        buy=as.numeric(buyi),
        xopen,xclose,xlow,xhigh,xval,xjg,xjg2,
        jg10=colMeans(jg[xidx[1:10],,drop=F],na.rm=T),
        open0=open[buyi,],
        price_out=close[yidx[1],]
      ) %>%
        mutate(
          ztrate=ifelse(floor(as.numeric(code) / 10000) %in% c(30, 68), 1.198, 1.098),
          roi=ifelse(open0>close1*ztrate,1,price_out/open0)
        ) 
    }
    
  }) %>%
    rbindlist()
  
) 

datasets <- rawdata[rowSums(is.na(rawdata))==0,]

datasets <- datasets %>%
  merge(
    datasets %>%
      group_by(buy) %>%
      summarise(
        btoday1=mean(close1/open1,na.rm=T), 
        bonite0=mean(open0/close1,na.rm=T),
        broi1=mean(close1/open2,na.rm=T),
        broi5=mean(close1/open5,na.rm=T),
        mean=mean(roi),
        sd=sd(roi)
      )
  ) %>%
  mutate(
    z=(roi-mean)/sd,
    class=(btoday1>1)*1000+(bonite0>1)*100+(broi1>1)*10+(broi5>1)*1
  )

#根据分类拆分数据

mfiles.train <- lapply(unique(datasets$class),function(i){
  datai <- datasets %>%
    filter(class==i) %>%
    filter(jg10>0) %>%
    # filter(buy<20250400) %>%
    mutate(
      open1=open1/open0,
      close1=close1/open0,
      high1=high1/open0,
      low1=low1/open0,
      jg1=jg1,
      jg2=jg2,
      jg35=jg3+jg4+jg5,
      jg10=jg10,
      closegr=close1/(close2+close3+close4+close5)*4,
      ztrate=ztrate,
      y=log(roi),
      onite0=1/close1
    ) %>%
    select(buy,code,open1,onite0,high1,low1,jg1,jg2,jg35,jg10,closegr,y,z,ztrate,btoday1,bonite0,broi1,broi5,class) 
  yi <- datai %>% select(buy,code,y,z,class) %>% as.matrix
  xi <- datai %>% select(-buy,-code,-y,-z,-class) %>% as.matrix
  data.table(xi,yi)
  # yi <- rbind(yi,NA) %>% as.data.table()
  # xi <- rbind(xi,NA) %>% as.data.table()
  # xxi <- lapply(1:(ncol(xi)-1),function(k){
  #   xxk <- sapply((k+1):ncol(xi),function(j){
  #     cbind(xi[[k]] * xi[[j]])
  #   }) 
  #   if(!is.matrix(xxk)){
  #     xxk <- matrix(xxk,nrow=1)
  #   }
  #   colnames(xxk) <- paste0('xx_',k,'_',(k+1):ncol(xi))
  #   xxk
  # })
  # xxi <- do.call(cbind,xxi)
  # out <- data.table(xi,xxi,yi)
  # out[-nrow(out),]
})

mfiles.test <- lapply(unique(datasets$class),function(i){
  datai <- datasets %>%
    filter(class==i) %>%
    filter(jg10>0) %>%
    filter(buy>20250400) %>%
    mutate(
      open1=open1/open0,
      close1=close1/open0,
      high1=high1/open0,
      low1=low1/open0,
      jg1=jg1,
      jg2=jg2,
      jg35=jg3+jg4+jg5,
      jg10=jg10,
      closegr=close1/(close2+close3+close4+close5)*4,
      ztrate=ztrate,
      y=log(roi),
      onite0=1/close1
    ) %>%
    select(buy,code,open1,onite0,high1,low1,jg1,jg2,jg35,jg10,closegr,y,z,ztrate,btoday1,bonite0,broi1,broi5,class) 
  yi <- datai %>% select(buy,code,y,z,class) %>% as.matrix
  xi <- datai %>% select(-buy,-code,-y,-z,-class) %>% as.matrix
  data.table(xi,yi)
  # yi <- rbind(yi,NA) %>% as.data.table()
  # xi <- rbind(xi,NA) %>% as.data.table()
  # xxi <- lapply(1:(ncol(xi)-1),function(k){
  #   xxk <- sapply((k+1):ncol(xi),function(j){
  #     cbind(xi[[k]] * xi[[j]])
  #   }) 
  #   if(!is.matrix(xxk)){
  #     xxk <- matrix(xxk,nrow=1)
  #   }
  #   colnames(xxk) <- paste0('xx_',k,'_',(k+1):ncol(xi))
  #   xxk
  # })
  # xxi <- do.call(cbind,xxi)
  # out <- data.table(xi,xxi,yi)
  # out[-nrow(out),]
})

#设定抽样随机树，训练模型

# filename <- 'test'
# set.seed(1); seeds <- sample(1:10000,100) #设定数据随机数
filename <- gsub('-','',Sys.Date())
set.seed(as.numeric(max(rownames(close)))); seeds <- sample(1:10000,500) #设定数据随机数

sample.rate <- 0.1

n_cores <- detectCores() 
n_cores <- 2^max(which(2^(1:5) < n_cores))
cl <- makeCluster(n_cores)
clusterExport(cl, varlist = c("mfiles.train", "seeds", "sample.rate"))
clusterEvalQ(cl, {
  library(rpart)
  library(dplyr)
})

system.time({
  models_both <- parLapply(cl, mfiles.train, function(mfile) {
    
    print(Sys.time())
    
    models_y <- lapply(1:length(seeds), function(i) {
      set.seed(seeds[i])
      sel <- sample(1:nrow(mfile), sample.rate * nrow(mfile))
      rpart(
        y ~ .,
        data = mfile[sel, ] %>%
          select(-buy, -code, -z, -class),
        method = "anova",
        control = rpart.control(cp = 0.001)
      )
    })
    
    models_z <- lapply(1:length(seeds), function(i) {
      set.seed(seeds[i])
      sel <- sample(1:nrow(mfile), sample.rate * nrow(mfile))
      rpart(
        z ~ .,
        data = mfile[sel, ] %>%
          select(-buy, -code, -y, -class),
        method = "anova",
        control = rpart.control(cp = 0.001)
      )
    })
    
    list(y = models_y, z = models_z)
  })
  
  # 拆开为两个对象
  models_y <- lapply(models_both, `[[`, "y")
  models_z <- lapply(models_both, `[[`, "z")
  
})

stopCluster(cl)

models_y <- lapply(models_both,function(x){x$y})
models_z <- lapply(models_both,function(x){x$z})

classlist <- unique(datasets$class)
system.time(save(models_y,models_z,classlist,sample.rate,file=paste0('models/sina_',filename,'.rda')))

################################################################
#回测
################################################################

# system.time(
#   valid.train <- lapply(1:length(models_y),function(k){
#     print(paste(k,Sys.time()))
#     mfile <- mfiles.train[[k]]
#     rlti <- lapply(1:length(seeds),function(i){
#       set.seed(seeds[i])
#       sel <- sample(1:nrow(mfile), sample.rate * nrow(mfile))
#       modeli1 <- models_y[[k]][[i]]
#       modeli2 <- models_z[[k]][[i]]
#       out1 <- predict(modeli1,newdata=mfile[-sel,])
#       out2 <- predict(modeli2,newdata=mfile[-sel,])
#       data.table(
#         i=i,
#         mfile[-sel,] %>% 
#           mutate(jg0=jg1/jg10) %>%
#           select(class,buy,code,y,z,onite0,jg0),
#         py=out1,pz=out2
#       )
#     }) %>% rbindlist()
#     rlti <- rlti %>%
#       group_by(class,buy,code,jg0) %>%
#       summarise(onite0=mean(onite0),wy=mean(py>0),wz=mean(pz>0),py=mean(py),pz=mean(pz),y=mean(y),z=mean(z))
#     rlti
#   })
# )

system.time(
  valid.test <- lapply(1:length(models_z),function(k){
    print(paste(k,Sys.time()))
    mfile <- mfiles.test[[k]]
    rlti <- lapply(1:length(seeds),function(i){
      modeli1 <- models_y[[k]][[i]]
      modeli2 <- models_z[[k]][[i]]
      out1 <- predict(modeli1,newdata=mfile)
      out2 <- predict(modeli2,newdata=mfile)
      data.table(
        i=i,
        mfile %>% 
          mutate(jg0=jg1/jg10) %>%
          select(class,buy,code,y,z,onite0,jg0),
        py=out1,pz=out2
      )
    }) %>% rbindlist()
    rlti <- rlti %>%
      group_by(class,buy,code,jg0) %>%
      summarise(onite0=mean(onite0),wy=mean(py>0),wz=mean(pz>0),py=mean(py),pz=mean(pz),y=mean(y),z=mean(z))
    rlti
  })
)

tests <- rbindlist(valid.test)
# tests <- rbindlist(valid.train)
# write.csv(tests,paste0('models/sina2_',filename,'.csv'),row.names=F)

tests %>%
  filter(onite0<1) %>%
  group_by(class) %>%
  summarise(n(),cor(y,py),cor(z,pz))

tests %>%
  mutate(roi=exp(y)) %>%
  mutate(roi=(roi-1)/2+1) %>%
  group_by(class,buy) %>%
  summarise(benchroi=mean(roi)) %>%
  merge(
    tests %>%
      filter(onite0<1) %>%
      mutate(roi=exp(y)) %>%
      mutate(roi=(roi-1)/2+1) %>%
      filter(py>0,pz>0,wy>0.6,wz>0.5) %>%
      arrange(desc(jg0)) %>%
      group_by(buy) %>%
      summarise(roi=mean(head(roi))),
    all.x=T
  ) %>%
  mutate(sel=ifelse(is.na(roi),0,1),roi=ifelse(is.na(roi),1,roi)) %>%
  group_by(floor(buy/1000000)) %>% 
  summarise(win=mean(roi>benchroi),bench=prod(benchroi),roi=prod(roi),n=n(),sel=mean(sel)) %>%
  mutate(idx=roi/bench) %>%
  as.data.frame()

tests %>%
  filter(onite0<=1) %>%
  mutate(roi=(exp(y)-1)/2+1) %>%
  filter(py>0,pz>0,wy>0.6,wz>0.5) %>%
  arrange(desc(jg0)) %>%
  group_by(buy,class) %>%
  summarise(
    roi=mean(head(roi)),
    code=paste(head(paste0(code,'(',round(roi,2),')')),collapse=',')
  ) %>%
  tail()

temp <- tests %>%
  filter(buy>=20250401) %>%
  mutate(roi=(exp(y)-1)/2+1) %>%
  group_by(class,buy) %>%
  summarise(benchroi=mean(roi)) %>%
  merge(
    tests %>%
      mutate(roi=(exp(y)-1)/2+1) %>%
      filter(onite0<=1) %>%
      filter(py>0,pz>0,wy>0.6,wz>0.5) %>%
      arrange(desc(jg0)) %>%
      group_by(buy) %>%
      summarise(roi=mean(head(roi,10)))
  ) %>%
  mutate(roi=ifelse(is.na(roi),1,roi)) %>%
  group_by(buy=as.Date(paste(substr(buy,1,4),substr(buy,5,6),substr(buy,7,8),sep='-'))) %>% 
  summarise(bench=prod(benchroi),roi=prod(roi)) 

temp$bench <- cumprod(temp$bench)
temp$roi <- cumprod(temp$roi)

temp %>%
  melt(id=1) %>%
  ggplot() + 
  geom_line(aes(x=buy,y=value,colour=variable))
