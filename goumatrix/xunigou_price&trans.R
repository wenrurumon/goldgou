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

cppFunction('arma::mat corpp(arma::mat X, arma::mat Y) {
    arma::mat Xt = X.t();
    arma::mat result = Xt*Y;
    result /= (Y.n_rows - 1);
    return result;
}', depends = "RcppArmadillo")

cppFunction('arma::mat cor_pvalue(arma::mat cor_mat, int n) {
    arma::mat t_stat = cor_mat % sqrt((n - 2) / (1 - cor_mat % cor_mat));
    arma::mat p_mat = 2 * arma::normcdf(-arma::abs(t_stat), 0, 1);
    return p_mat;
}', depends = "RcppArmadillo")

cppFunction('
NumericMatrix distCpp(NumericMatrix x1, NumericMatrix x2) {
  int n1 = x1.nrow();
  int n2 = x2.nrow();
  int p = x1.ncol();
  NumericMatrix dists(n1, n2);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double dist = 0.0;
      for (int k = 0; k < p; k++) {
        dist += pow(x1(i, k) - x2(j, k), 2);
      }
      dists(i, j) = sqrt(dist);
    }
  }
  
  return dists;
}')

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
#                             start_date=20210101,
#                             end_date=gsub('-','',Sys.Date()), adjust='qfq') %>%
#       mutate(code=codei)
#     x[[1]] <- sapply(x[[1]],paste)
#     x
#   }))
# )
# 
# #Update data
# hisdata <- hisdata %>% merge(allcodes)
# write.csv(hisdata,'hisdata210101.csv')

#Load Data
hisdata <- fread('hisdata210101.csv')[,-1]
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
  filter(code%in%jgp$code)
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
#量价猩基金
########################################################################################################################

roi <- close[2:nrow(close),]/open[1:(nrow(close)-1),]
pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]

#回测用数据

b <- 29

system.time(
  datapool <- lapply((b+3):nrow(roi),function(i){
    codei <- unique((jgp %>% filter(did>=i-20))$code) %>% paste
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

#回测

k <- 2

system.time(
  tests <- lapply((k+1):length(datapool),function(i){
    # print(paste(i,Sys.time()))
    test <- do.call(rbind,lapply(1:ncol(datapool[[i]]$X),function(j){
      x <- datapool[[i]]$X[,j]
      xk <- do.call(rbind,lapply(2:k,function(k){
        data.table(
          codei = colnames(datapool[[i]]$X)[j],
          roidist = colSums(((x-datapool[[i-k]]$X)^2)[1:(nrow(datapool[[1]]$X)/2),]),
          popdist = colSums(((x-datapool[[i-k]]$X)^2)[1:(nrow(datapool[[1]]$X)/2)+nrow(datapool[[1]]$X)/2,]),
          datapool[[i-k]]$Y %>% select(code,roi)
        )
      })) 
      xk
    })) %>%
      merge(
        datapool[[i]]$Y %>% select(codei=code,obs,buy,sell,actual=roi)
      )
    test <- data.table(i=i,test) 
    test %>%
      merge(
        test %>%
          group_by(codei) %>%
          summarise(roithres=quantile(roidist,0.05),popthres=quantile(popdist,0.05))
      ) %>%
      filter(roidist<roithres,popdist<popthres) %>%
      group_by(codei,obs,buy,sell) %>%
      summarise(mean=mean(roi),sd=sd(roi),win=mean(roi>1),actual=mean(actual),n=n()) %>%
      arrange(desc(actual))
  })
)

# save(tests,file='trace_2021_20240920.rda')
k2
test <- do.call(rbind,tests) %>%
  mutate(sel=(win>0.5)&(mean>1)) %>%
  arrange(obs,desc(mean)) %>%
  mutate(code=paste0(codei,'(',round(actual,2),')')) %>%
  group_by(obs,buy,sell) %>%
  summarise(bench=mean(actual),roi=mean(head(actual[sel])),code=paste(head(code[sel]),collapse=',')) %>%
  group_by(obs=substr(obs,1,6)) %>%
  summarise(bench=prod(bench),roi=prod(roi,na.rm=T))

do.call(rbind,tests) %>%
  mutate(sel=(win>0.5)&(mean>1)) %>%
  arrange(obs,desc(mean)) %>%
  mutate(code=paste0(codei,'(',round(actual,2),')')) %>%
  group_by(obs,buy,sell) %>%
  summarise(bench=mean(actual),roi5=mean(head(actual[sel],5)),roi10=mean(head(actual[sel],10)),roi20=mean(head(actual[sel],20))) %>%
  group_by(obs=substr(obs,1,4)) %>%
  summarise(
    mean(roi5>bench,na.rm=T),mean(roi10>bench,na.rm=T),mean(roi20>bench,na.rm=T),
    bench=prod(bench),roi5=prod(roi5,na.rm=T),roi10=prod(roi10,na.rm=T),roi20=prod(roi20,na.rm=T)
    ) 

rlt <- do.call(rbind,lapply(tests,function(x){
  x %>%
    filter(win>0.5,mean>1) %>%
    arrange(desc(mean)) %>%
    head(100)
}))
rlt %>% group_by(obs) %>% summarise(actual=mean(actual))
write.csv(rlt,'xunigou_k2b29.csv')

#明天买什么

i <- nrow(roi)
codei <- unique((jgp %>% filter(did>=i-20))$code) %>% paste

roi0 <- roi[i-(b:0),codei]
pop0 <- pop[i-(b:0),codei]
roi0[is.na(roi0)] <- 1
X.obs <- (rbind(roi=roi0,pop=pop0))
roi0 <- roi[i-(b:0+2),codei]
pop0 <- pop[i-(b:0+2),codei]
roi2 <- roi[i,codei]
roi0[is.na(roi0)] <- 1
roi2[is.na(roi2)] <- 1
X.ref <- rbind(roi=roi0,pop=pop0)
Y.ref <- roi2

test <- do.call(rbind,lapply(1:ncol(X.obs),function(j){
  x <- X.obs[,j]
  data.table(
    codei=colnames(X.obs)[j],
    roidist = colSums(((x-X.ref)^2)[1:(nrow(X.obs)/2),]),
    popdist = colSums(((x-X.ref)^2)[1:(nrow(X.obs)/2)+nrow(X.obs)/2,]),
    roi=Y.ref
  )
}))

test %>%
  merge(
    test %>%
      group_by(codei) %>%
      summarise(roithres=quantile(roidist,0.05),popthres=quantile(popdist,0.05))
  ) %>%
  filter(roidist<roithres,popdist<popthres) %>%
  group_by(codei) %>%
  summarise(mean=mean(roi),sd=sd(roi),win=mean(roi>1),n=n()) %>%
  merge(
    rawhis %>%
      select(codei=code,name=name) %>%
      unique()
  ) %>%
  filter(mean>1,win>0.5) %>%
  arrange(desc(mean)) %>%
  head
