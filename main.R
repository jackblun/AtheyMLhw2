############################################################
### ECON 293 Homework 2

# Luis Armona and Jack Blundell
# Stanford University
# Spring 2017

# Section numbers correspond to assignment page

############################################################

# set your working directory

setwd("C:/Users/Jack/Documents/Git/AtheyMLhw2") # Jack
#setwd('/home/luis/AtheyMLhw1') #Luis
# clear things in RStudio

rm(list = ls())


# Call packages

library(ggplot2)
#library(dplyr)
#library(reshape2)
#library(glmnet)
#library(plotmo)
#library(pogs)
#library(balanceHD)

# set seed
set.seed(12345)

############################################################

# Load data
fname <- 'Data/charitable_withdummyvariables.csv'
char <- read.csv(fname)
#attach(char) # attach so don't have to call each time


############################################################
###  Regression for average treatment effect

reg.ols <- lm(out_amountgive ~ treatment, data = char) 
summary(reg.ols) # show results, significant at 90% but not 95% level
# Consistent with Table 4 of paper
confint(reg.ols, level=0.95) # CI


##############################
#randomly censor individuals
#via a  complex, highly nonlinear fcn  of votes 4 bush in state,
#

ps.fcn <- function(v,c,pg,t){
  #v_t <- (v-.25)/.5
  v_t <- v
  #ihs_pg <- log(pg + sqrt(pg ^ 2 + 1))/5
  #p<- (c*(acos(v_t))*atan(v_t^2)  - .5*exp(v_t))/4 + (t*((ihs_pg)) + (1-t))/2
  ihs_pg <- log(pg + sqrt(pg ^ 2 + 1))
  p<- (1-t)*(c+1)*(acos(v_t)*atan(v_t) )/3 + 
      t*(.01+(-.01*ihs_pg^5 + 1*ihs_pg^3)/300)
  p<- pmin(pmax(0,p),1)
  return(p)
}
#story to accompany this fcn: ACLU wants to help those in trouble in "red states" but do not 
#feel they can make a difference in really, really red states so target donors less often
plot(seq(0,1,.001),ps.fcn(seq(0,1,.001),4,800,0)) #a plot of the function
lines(seq(0,1,.001),ps.fcn(seq(0,1,.001),3,800,0))
lines(seq(0,1,.001),ps.fcn(seq(0,1,.001),2,200,0))
lines(seq(0,1,.001),ps.fcn(seq(0,1,.001),1,200,0))
jpeg(file='select_c')

plot(char$hpa, ps.fcn(0,0,char$hpa,1))
jpeg(file='select_t')
#char$mibush=char$perbush==-999
#char$perbush[char$mibush]=.5

# Selection rule
char$ps.select <- ps.fcn(char$perbush,char$cases,char$hpa,char$treatment) # hpa is highest previous contribution. cases is court cases from state which organization was involved.
#deal with those missing covariates
char$ps.select[ which(perbush==-999
            | cases==-999
            | hpa==-999)] <- 0.5
# True propensity score via Bayes' theorem
prop.treat <- mean(char$treatment)
char$ps.select.t <- ps.fcn(char$perbush,char$cases,char$hpa,1)
char$ps.select.c <- ps.fcn(char$perbush,char$cases,char$hpa,0)
char$ps.true <- (prop.treat*char$ps.select.t)/(prop.treat*char$ps.select.t + (1 - prop.treat)*char$ps.select.c)
char$ps.true[ which(perbush==-999
                      | cases==-999
                      | hpa==-999)] <- prop.treat

# Plot CDF of this nonlinear function
ggplot(char,aes(x=ps.true))+ stat_ecdf()

# Set seed
set.seed(21) 

#replace -999s with 0s (since there are already missing dummies)
for (v in names(char)){
  mi_v <- paste(v,'_missing',sep='') 
  if (mi_v %in% names(char)){
    print(paste('fixing',v))
    char[(char[,mi_v]==1),v]<-0
  }
}

# Selection rule (=1 of uniform random [0,1] is lower, so those with higher ps.true more likely to be selected)
selection <- runif(nrow(char)) <= char$ps.select

char.censored <- char[selection,] #remove observations via propensity score rule

ggplot(char,aes(x=perbush)) + geom_histogram()+xlim(c(0,1))

ggplot(char.censored,aes(x=ps.true)) + geom_histogram() +xlim(c(0,1))

#overlap in true propensity score
ggplot(char.censored,aes(x=ps.true,colour=factor(treatment))) + stat_ecdf()

ggplot(char.censored,aes(x=ps.true,y=hpa,colour=factor(treatment))) + geom_point()
#there is clear overlap, but clearly assymetries going on with hpa as well

#################################
# Feature engineering for censored data

covars.all <- char.censored[,c(14:22,23:63)] #skip the state indicator used for summ stats
#formula to interact all covariates no interactions for missing dummies.
#for tractability, we interact individ. covars with each other, and state vars with each other
#create design matrix storing all features
covars.regular <-char.censored[,c(14:22,23:44)]
covars.missing <- char.censored[,c(45:63)]
int.level = 2 #the degree of interaction between covariates that are not missing dummies
covars.poly.str = paste('(', paste(names(covars.regular)[1:9],collapse='+'),')^',int.level,
                        ' + (', paste(names(covars.regular)[11:31],collapse='+'),')^',int.level,
                        ' + ',paste(names(covars.missing),collapse='+'),sep='') 
#covars.poly.str = paste('(', paste(names(covars.regular),collapse='+'),')^',int.level,
#                        ' + ',paste(names(covars.missing),collapse='+'),sep='') 
covars.poly <-model.matrix(as.formula(paste('~ ',covars.poly.str)),data=char.censored)

###################################
