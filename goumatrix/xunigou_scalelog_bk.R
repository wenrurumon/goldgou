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

roi <- close[2:nrow(close),]/open[1:(nrow(close)-1),]
pop <- apply(val,1,function(x){x/sum(x,na.rm=T)}) %>% t
pop <- pop[2:nrow(pop),]-pop[1:(nrow(pop)-1),]

#回测用数据

b <- 39

system.time(
  datapool <- lapply((b+3):nrow(roi),function(i){
    codei <- unique((jgp %>% filter(did>=i-20,did<=i))$code) %>% paste
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

k <- 2
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
#       xk <- xk %>%
#         mutate(dist=roidist*popdist)
#       xk
#     })) %>%
#       merge(y.obs) %>%
#       filter(roidist<=0.01,popdist<=0.01) %>%
#       group_by(obs,buy,sell,code) %>%
#       summarise(mean=mean(refroi),win=mean(refroi>1),actual=mean(roi)) %>%
#       filter(mean>1,win>0.5) %>%
#       arrange(desc(mean)) 
#   })
# )
# save(tests,file='trace_scalelog_0921b39k2.rda')
load('trace_scalelog_0921b29k2.rda')

valid <- rbindlist(lapply(tests,head,5)) %>%
  group_by(obs) %>%
  summarise(actual=mean(actual)) %>%
  group_by(obs=substr(obs,1,6)) %>%
  summarise(actual=prod((actual-1)/2+1,na.rm=T),n()) %>%
  merge(
    bench %>%
      group_by(obs=substr(obs,1,6)) %>%
      summarise(bench=prod((bench-1)/2+1))
  ) 
valid$cumactual <- cumprod(valid$actual)
valid$cumbench <- cumprod(valid$bench)

rlt <- rbindlist(lapply(tests,head,100))
# write.csv(rlt,'trace_0922slogb19k2.csv')

plot.ts(valid$cumactual,col=2); lines(valid$cumbench)

rbindlist(lapply(tests,head,5)) %>%
  group_by(obs) %>%
  summarise(actual=mean(actual)) %>%
  group_by(obs=substr(obs,1,1)) %>%
  summarise(actual=prod((actual-1)/2+1,na.rm=T),n()) %>%
  merge(
    bench %>%
      group_by(obs=substr(obs,1,1)) %>%
      summarise(bench=prod((bench-1)/2+1))
  ) 
