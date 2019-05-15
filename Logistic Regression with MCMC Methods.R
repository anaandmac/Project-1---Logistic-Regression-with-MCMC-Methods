#In this code we will know the true parameters values in a Logistic Regression and we will estimate they 
#with three different methods:maximum likelihood, Lasso, MCMC algorithm e SSVS.

#Setting the seed
set.seed(2000)

#Data Simulation

#True Value
beta=matrix(c(3,1.5,0.7,0,0,0,0,-4),ncol=1)

#Sample length
n=1000

#Covariates
x1=sample(seq(1:5),n,replace=TRUE)
x2=sample(seq(6:10),n,replace=TRUE)
x3=sample(seq(8:12),n,replace=TRUE)
x4=sample(seq(15:20),n,replace=TRUE)
x5=sample(seq(1:10),n,replace=TRUE)
x6=sample(seq(7:9),n,replace=TRUE)
x7=sample(seq(11:14),n,replace=TRUE)
X=matrix(c(rep(1,n),x1,x2,x3,x4,x5,x6,x7),byrow=FALSE,ncol=8)

#Probability vector
prob=NULL
pred=X%*%beta
for (i in 1:n){
  prob[i]=1/(1+exp(-(pred[i])))
}

library(ggplot2)

ggplot()+ 
  geom_histogram(aes(prob),,colour="#1F3552",fill="#4271AE",binwidth = 1/20)+
  labs(y="Frequency", x="Probability")+
  ggtitle("Simulated Data Probability")

#Simulation the Bernoulli data
y_resp=NULL
for (i in 1:n){
    y_resp[i]=rbinom(1,1,prob=prob[i])
}

table(y_resp)#542 458 

######################################################
#Maximum likelihood
######################################################

#model
model.ml=glm(y_resp~X-1,family=binomial(link="logit"))#We put -1 because X0 is a vector with 1s

#Saving the values
ml.estim=matrix(ncol=length(beta),nrow=1)
colnames(ml.estim)=c("beta0","beta1","beta2","beta3","beta4",
                                  "beta5","beta6","beta7")
for (k in 1:length(beta)){
    ml.estim[1,k]=model.ml$coefficients[k]
}

ml.estim

######################################################
#MCMC - with MCMCpack
######################################################

#######################
#Priori - Normal
#######################

#install.packages("MCMCpack")
library(MCMCpack)

priori.normal = function(beta){ 
  sum(dnorm(beta),log=TRUE)
}

#"Seeing" the density shape 
normal=as.data.frame(rnorm(10000))
ggplot(data=normal,aes(x=`rnorm(10000)`))+ 
  geom_histogram(colour="#1F3552",fill="#4271AE",binwidth=density(normal$`rnorm(10000)`)$bw,aes(y=..density..))+
  labs(x="X", y="Probability")+
  geom_density(fill="#1F3552",alpha=0.3)+
  ggtitle("Normal- Density")

#Model
model.mcmc.normal=MCMClogit(y_resp~X-1,user.prior.density=priori.normal)

#Saving the values
mcmc.normal.estim=matrix(ncol=length(beta),nrow=1)
colnames(mcmc.normal.estim)=c("beta0","beta1","beta2","beta3","beta4",
                                         "beta5","beta6","beta7")
for (k in 1:length(beta)){
  mcmc.normal.estim[1,k]=((summary(model.mcmc.normal))$statistics[,1])[k]
}

mcmc.normal.estim

#######################
#Priori - T-Student
#######################

gl=3#degrees of freedom

priori.t=function(beta){ 
  sum(dt(beta,df=gl),log=TRUE)
}

#"Seeing" the density shape 
t.stu=as.data.frame(rt(10000,df=gl))
ggplot(data=t.stu,aes(x=`rt(10000, df = gl)`))+ 
  geom_histogram(colour="#1F3552",fill="#4271AE",binwidth=density(t.stu$`rt(10000, df = gl)`)$bw,aes(y=..density..))+
  labs(x="X", y="Probability")+
  geom_density(fill="#1F3552",alpha=0.3)+
  ggtitle("T-Student - Density")

#Model
model.mcmc.t=MCMClogit(y_resp~X-1,user.prior.density=priori.t)

#Saving the values
mcmc.t.estim=matrix(ncol=length(beta),nrow=1)
colnames(mcmc.t.estim)=c("beta0","beta1","beta2","beta3","beta4",
                              "beta5","beta6","beta7")
for (k in 1:length(beta)){
  mcmc.t.estim[1,k]=((summary(model.mcmc.t))$statistics[,1])[k]
}

mcmc.t.estim

#######################
#Priori - Skew Normal
#######################

#install.packages("sn")
library(sn)

alpha=5#coefficient of asymmetry

priori.sn = function(beta){ 
  sum(dsn(beta,alpha),log=T)
}

#Using the normal
#priori.sn=function(beta,alpha){
#  sum(log(2*dnorm(beta)*pnorm(alpha*beta)),log=T)
#}

#"Seeing" the density shape 
sn=as.data.frame(rsn(10000,0,1,alpha))
ggplot(data=sn,aes(x=`rsn(10000, 0, 1, alpha)`))+ 
  geom_histogram(colour="#1F3552",fill="#4271AE",binwidth=density(sn$`rsn(10000, 0, 1, alpha)`)$bw,aes(y=..density..))+
  labs(x="X", y="Probability")+
  geom_density(fill="#1F3552",alpha=0.3)+
  ggtitle("Skew Normal- Density")

#Model
model.mcmc.sn=MCMClogit(y_resp~X-1,user.prior.density=priori.sn)

#Saving the values
mcmc.sn.estim=matrix(ncol=length(beta),nrow=1)
colnames(mcmc.sn.estim)=c("beta0","beta1","beta2","beta3","beta4",
                         "beta5","beta6","beta7")
for (k in 1:length(beta)){
  mcmc.sn.estim[1,k]=((summary(model.mcmc.sn))$statistics[,1])[k]
}

mcmc.sn.estim

######################################################
#Bayesian Lasso
######################################################

#install.packages("EBglmnet")
library(EBglmnet)

#Lasso NEg as default
cv=cv.EBglmnet(X,y_resp,family="binomial",prior="lassoNEG")#finding the best hyperparameters
parm=cv$hyperparameters
lasso=EBglmnet(X,y_resp,hyperparameters=c(parm[1],parm[2]),family="binomial")

#Saving the values
lasso.estim=matrix(ncol=length(lasso$fit[,1])+1,nrow=1)
colnames(lasso.estim)=c("beta0","beta1","beta2","beta7")

#Saving the intercept
lasso.estim[1,1]=lasso$Intercept

#Others coefficients
for(k in 1:length(lasso$fit[,1])){
    lasso.estim[1,k+1]=lasso$fit[k,3]
  }

lasso.estim

#####################################################
#SSVS
#####################################################
#install.packages("BoomSpikeSlab")
library("BoomSpikeSlab")

logit.ssvs=logit.spike(y_resp~X,niter=10000)

#Saving the values
logit.ssvs.estim=matrix(ncol=length(beta),nrow=1)
colnames(logit.ssvs.estim)=c("beta0","beta1","beta2","beta3","beta4","beta5","beta6","beta7")

for(k in 1:length(beta)){
    logit.ssvs.estim[1,k]=mean(logit.ssvs$beta[,k])
}

logit.ssvs.estim

#####################################################
#Comparing the estimates
#####################################################
ml.estim
mcmc.normal.estim
mcmc.t.estim
mcmc.sn.estim
lasso.estim
logit.ssvs.estim
