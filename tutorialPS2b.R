# Tutorial focused on Problem Set 2

# First, install packages, being sure to follow the following syntax
# install.packages("devtools")
# install.packages("randomForest") 
# install.packages("rattle) 
# .... and so on, until you have installed all the packages you've loaded below

# Every time you open RStudio, you'll need to reload the packages
# You may not use all of these packages in the tutorial, but they
# might be useful later on
library(devtools)
library(randomForest) 
library(rpart) # decision tree
library(rpart.plot) # enhanced tree plots
library(rattle) # fancy tree plot
library(ROCR)
library(Hmisc)
library(corrplot)
library(texreg)
library(glmnet)
library(reshape2)
library(knitr)
library(xtable)
library(lars)
library(ggplot2)
library(matrixStats)
library(plyr)
library(doMC)
library(stargazer)
registerDoMC(cores=4) # for a simple parallel computation

################## INSTALL CUSTOM PACKAGES ***###############################
# Note you also need to install RTools to install packages from source
# you may need to recopy after installation: put them in c:/rtools 
# WITHOUT any subdirectories for versions

install_github("susanathey/causalTree", ref="master", force=TRUE)
install_github("swager/randomForestCI")

#Note this package cannot be installed using install_github.  Use the following instead.
install.packages("https://raw.github.com/swager/gradient-forest/master/releases/gradient-forest-alpha.tar.gz", repos = NULL, type = "source")
# WINDOWS (if trouble, make sure Rtools installed in c:\Rtools with no version number subdirectory; also there is a windows
# installer file on the github)


##########################################################



#install.packages("rpart.plot", dependencies=TRUE, repos='http://cran.us.r-project.org')
#install.packages("reshape2", dependencies=TRUE, repos='http://cran.us.r-project.org')
#install.packages("plyr", dependencies=TRUE, repos='http://cran.us.r-project.org')

library(causalTree)
library(randomForestCI)
library(reshape2)
library(plyr)


# set your working directory
# write the path where the folder containing the files is located
# setwd("/Users/munyikz/Documents/tutorial")
setwd("C:/Users/athey/Google Drive/MLClass/2017/R Tutorial")

# clear things in RStudio
rm(list = ls())

# Loading data
# We use data from a Social Voting  (paper is attached) experiment
# The data comes in a csv format
filename = 'socialneighbor.csv'
social <- read.csv(filename)

# some simple print statements 
print(paste("Loaded csv:", filename, " ..."))
colnames(social)

# We generate noise covariates and add them in the data
set.seed(123)
noise.covars <- matrix(data = runif(nrow(social) * 13), 
                       nrow = nrow(social), ncol = 13)
noise.covars <- data.frame(noise.covars)
names(noise.covars) <- c("noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                         "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")

# Add these noise covariates to the social data
working <- cbind(social, noise.covars)

# We want to run on a subsample of the data only
# This is the main dataset used in this tutorial
set.seed(333)
working <- working[sample(nrow(social), 20000), ]

# Pick a selection of covariates

covariate.names <- c("yob", "hh_size", "sex", "city", "g2000","g2002", "p2000", "p2002", "p2004"
                     ,"totalpopulation_estimate","percent_male","median_age", "percent_62yearsandover"
                     ,"percent_white", "percent_black", "median_income",
                     "employ_20to64", "highschool", "bach_orhigher","percent_hispanicorlatino",
                     "noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                     "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")

# The dependent (outcome) variable is whether the person voted, 
# so let's rename "outcome_voted" to Y
names(working)[names(working)=="outcome_voted"] <- "Y"

# Extract the dependent variable
Y <- working[["Y"]]

# The treatment is whether they received the "your neighbors are voting" letter
names(working)[names(working)=="treat_neighbors"] <- "W"

# Extract treatment variable & covariates
W <- working[["W"]]
covariates <- working[covariate.names]

# some algorithms require our covariates be scaled
# scale, with default settings, will calculate the mean and standard deviation of the entire vector, 
# then "scale" each element by those values by subtracting the mean and dividing by the sd
covariates.scaled <- scale(covariates)
processed.unscaled <- data.frame(Y, W, covariates)
processed.scaled <- data.frame(Y, W, covariates.scaled)


# some of the models in the tutorial will require training, validation, and test sets.
# set seed so your results are replicable 
# divide up your dataset into a training and test set. 
# Here we have a 90-10 split, but you can change this by changing the the fraction 
# in the sample command
set.seed(44)
smplmain <- sample(nrow(processed.scaled), round(9*nrow(processed.scaled)/10), replace=FALSE)

processed.scaled.train <- processed.scaled[smplmain,]
processed.scaled.test <- processed.scaled[-smplmain,]

y.train <- as.matrix(processed.scaled.train$Y, ncol=1)
y.test <- as.matrix(processed.scaled.test$Y, ncol=1)

# create 45-45-10 sample
smplcausal <- sample(nrow(processed.scaled.train), 
                     round(5*nrow(processed.scaled.train)/10), replace=FALSE)
processed.scaled.train.1 <- processed.scaled.train[smplcausal,]
processed.scaled.train.2 <- processed.scaled.train[-smplcausal,]


# Creating Formulas
# For many of the models, we will need a "formula"
# This will be in the format Y ~ X1 + X2 + X3 + ...
# For more info, see: http://faculty.chicagobooth.edu/richard.hahn/teaching/formulanotation.pdf
print(covariate.names)
sumx = paste(covariate.names, collapse = " + ")  # "X1 + X2 + X3 + ..." for substitution later
interx = paste(" (",sumx, ")^2", sep="")  # "(X1 + X2 + X3 + ...)^2" for substitution later

# Y ~ X1 + X2 + X3 + ... 
linearnotreat <- paste("Y",sumx, sep=" ~ ")
linearnotreat <- as.formula(linearnotreat)
linearnotreat

# Y ~ W + X1 + X2 + X3 + ...
linear <- paste("Y",paste("W",sumx, sep=" + "), sep=" ~ ")
linear <- as.formula(linear)
linear

# Y ~ W * (X1 + X2 + X3 + ...)   
# ---> X*Z means include these variables plus the interactions between them
linearhet <- paste("Y", paste("W * (", sumx, ") ", sep=""), sep=" ~ ")
linearhet <- as.formula(linearhet)
linearhet

processed.scaled.test$propens <- mean(processed.scaled.test$W) #note this is randomized experiment so will use constant propens
processed.scaled.test$Ystar <- processed.scaled.test$W * (processed.scaled.test$Y/processed.scaled.test$propens) -
  (1-processed.scaled.test$W) * (processed.scaled.test$Y/(1-processed.scaled.test$propens))
MSElabelvec <- c("")
MSEvec <- c("")

# causal tree/forest params

# Set parameters
split.Rule.temp = "CT"
cv.option.temp = "CT"
split.Honest.temp = T
cv.Honest.temp = T
split.alpha.temp = .5
cv.alpha.temp = .5
split.Bucket.temp = T
bucketMax.temp= 100
bucketNum.temp = 5
minsize.temp=50


# Some of the models need to get causal effects out by comparing mu(X,W=1)-mu(X,W=0).  Create datasets to do that easily

processed.scaled.testW0 <- processed.scaled.test
processed.scaled.testW0$W <- rep(0,nrow(processed.scaled.test))


processed.scaled.testW1 <- processed.scaled.test
processed.scaled.testW1$W <- rep(1,nrow(processed.scaled.test))

# make this bigger -- say 2000 -- if run time permits
numtreesCT <- 200
numtreesGF <- 200

#################################################
#######  FINISH SETUP####################################################
###############################################


# Propensity forest


# Now try a propensity forest--split based on W rather than on treatment effects

ncolx<-length(processed.scaled.train)-2 #total number of covariates
ncov_sample<-floor(ncolx/3) #number of covariates (randomly sampled) to use to build tree


pf <- propensityForest(as.formula(paste("Y~",sumx)), 
                       data=processed.scaled.train,
                       treatment=processed.scaled.train$W, 
                       split.Bucket=F, 
                       sample.size.total = floor(nrow(processed.scaled.train) / 2), 
                       nodesize = 25, num.trees=numtreesCT,
                       mtry=ncov_sample, ncolx=ncolx, ncov_sample=ncov_sample )

pfpredtest <- predict(pf, newdata=processed.scaled.test, type="vector")

pfpredtrainall <- predict(pf, newdata=processed.scaled.train, 
                          predict.all = TRUE, type="vector")
print(c("mean of ATE treatment effect from propensityForest on Training data", 
        round(mean(pfpredtrainall$aggregate),5)))

pfvar <- infJack(pfpredtrainall$individual, pf$inbag, calibrate = TRUE)
plot(pfvar)



# calculate MSE against Ystar
pfMSEstar <- mean((processed.scaled.test$Ystar-pfpredtest)^2)
print(c("MSE using ystar on test set of causalTree/propforest",pfMSEstar))

MSElabelvec <- append(MSElabelvec,"propensity forest")
MSEvec <- append(MSEvec,pfMSEstar)

####################################################################
# Now try gradient forest
#####################################################################


library(gradient.forest)
X = as.matrix(processed.scaled.train[,covariate.names])
X.test = as.matrix(processed.scaled.test[,covariate.names])
Y  = as.matrix(processed.scaled.train[,"Y"])
W  = as.matrix(processed.scaled.train[,"W"])
gf <- causal.forest(X, Y, W, num.trees = numtreesGF, ci.group.size = 4,
                    precompute.nuisance = FALSE)
preds.causal.oob = predict(gf, estimate.variance=TRUE)
preds.causal.test = predict(gf, X.test, estimate.variance=TRUE)
mean(preds.causal.oob$predictions)  
plot(preds.causal.oob$predictions, preds.causal.oob$variance.estimates)


mean(preds.causal.test$predictions)  
plot(preds.causal.test$predictions, preds.causal.test$variance.estimates)


# calculate MSE against Ystar
gfMSEstar <- mean((processed.scaled.test$Ystar-preds.causal.test$predictions)^2)
print(c("MSE using ystar on test set of gradient causal forest",gfMSEstar))
MSElabelvec <- append(MSElabelvec,"gradient causal forest")
MSEvec <- append(MSEvec,gfMSEstar)

# Now try with orthogonalization--orthog does not have to be random forest

Yrf <- regression.forest(X, Y, num.trees = numtreesGF, ci.group.size = 4)
Yresid <- Y - predict(Yrf)$prediction


# orthogonalize W -- if obs study
Wrf <- regression.forest(X, W, num.trees = numtreesGF, ci.group.size = 4)
Wresid <- W - predict(Wrf)$predictions # use if you are orthogonalizing W, e.g. for obs study


gfr <- causal.forest(X,Yresid,Wresid,num.trees=numtreesGF, ci.group.size=4, 
                     precompute.nuisance = FALSE)
preds.causalr.oob = predict(gfr, estimate.variance=TRUE)
mean(preds.causalr.oob$predictions)  
plot(preds.causalr.oob$predictions, preds.causal.oob$variance.estimates)

Xtest = as.matrix(processed.scaled.test[,covariate.names])
preds.causalr.test = predict(gfr, Xtest, estimate.variance=TRUE)
mean(preds.causalr.test$predictions)  
plot(preds.causalr.test$predictions, preds.causal.test$variance.estimates)

# calculate MSE against Ystar
gfrMSEstar <- mean((processed.scaled.test$Ystar-preds.causalr.test$predictions)^2)
print(c("MSE using ystar on test set of orth causal gradient forest",gfrMSEstar))
MSElabelvec <- append(MSElabelvec,"orth causal gradient forest")
MSEvec <- append(MSEvec,gfrMSEstar)



###### # ######### # ############## # ################### # ##################
# We can the created formulas to do linear regression and logit regression #
###### # ######### # ############## # ################### # ##################

#####################
# Linear Regression with Treatment Effect Heterogeneity #
#####################

lm.linearhet <- lm(linearhet, data=processed.scaled)
summary(lm.linearhet)

#######################
# Logistic Regression #
#######################
# See:http://www.ats.ucla.edu/stat/r/dae/logit.htm

# The code below estimates a logistic regression model using 
# the glm (generalized linear model) function. 
mylogit <- glm(linearhet, data = processed.scaled, family = "binomial")
summary(mylogit)

##################################
# LASSO variable selection + OLS #
##################################
# see https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
# and also help(glmnet)

# LASSO takes in a model.matrix
# First parameter is the model (here we use linear, which we created before)
# Second parameter is the dataframe we want to creaate the matrix from
linear.train <- model.matrix(linearhet, processed.scaled.train)[,-1]
linear.test <- model.matrix(linearhet, processed.scaled.test)[,-1]
linear.train.1 <- model.matrix(linearhet, processed.scaled.train.1)[,-1]
linear.train.2 <- model.matrix(linearhet, processed.scaled.train.2)[,-1]

# Use cross validation to select the optimal shrinkage parameter lambda
# and the non-zero coefficients
lasso.linear <- cv.glmnet(linear.train.1, y.train[smplcausal,],  alpha=1, parallel=TRUE)

# prints the model, somewhat information overload, 
# but you can see the mse, and the nonzero variables and the cross validation steps
lasso.linear

# plot & select the optimal shrinkage parameter lambda
plot(lasso.linear)
lasso.linear$lambda.min
lasso.linear$lambda.1se
# lambda.min gives min average cross-validated error 
# lambda.1se gives the most regularized model such that error is 
# within one standard error of the min; this value of lambda is used here.

# List non-zero coefficients found. There are two ways to do this.
# coef(lasso.linear, s = lasso.linear$lambda.1se) # Method 1
coef <- predict(lasso.linear, type = "nonzero") # Method 2

# index the column names of the matrix in order to index the selected variables
colnames <- colnames(linear.train.1)
selected.vars <- colnames[unlist(coef)]

# do OLS using these coefficients USING independent sample
linearwithlass <- paste("Y", paste(append(selected.vars, "W"),collapse=" + "), sep = " ~ ") 
linearwithlass <- as.formula(linearwithlass)
lm.linear.lasso <- lm(linearwithlass, data=processed.scaled.train.2)
yhat.linear.lasso <- predict(lm.linear.lasso, newdata=processed.scaled.test)
summary(lm.linear.lasso)

predictedW0 <- predict(lm.linear.lasso, newdata=processed.scaled.testW0)

predictedW1 <- predict(lm.linear.lasso, newdata=processed.scaled.testW1)

lassocauseff <- predictedW1-predictedW0

# calculate MSE against Ystar
lassoMSEstar <- mean((processed.scaled.test$Ystar-lassocauseff)^2)
print(c("MSE using ystar on test set of lasso",lassoMSEstar))
MSElabelvec <- append(MSElabelvec,"lasso")
MSEvec <- append(MSEvec,lassoMSEstar)



###################################
# Single Tree  #
###################################

# Classification Tree with rpart
# grow tree 
set.seed(444)
linear.singletree <- rpart(formula = linear, data=processed.scaled.train, 
                         method = "anova", y=TRUE,
                         control=rpart.control(cp=1e-04, minsplit=30))

linear.singletree$cptable
printcp(linear.singletree) # display the results 
plotcp(linear.singletree) # visualize cross-validation results 

# very detailed summary of splits, uncomment the code below and execute to see
# summary(linear.singletree) 

# prune the tree
op.index <- which.min(linear.singletree$cptable[, "xerror"])
cp.vals <- linear.singletree$cptable[, "CP"]
treepruned.linearsingle <- prune(linear.singletree, cp = cp.vals[op.index])

# apply model to the test set to get predictions
singletree.pred.class <- predict(treepruned.linearsingle, newdata=processed.scaled.test)

# plot tree 
plot(treepruned.linearsingle, uniform=TRUE, 
     main="Classification Tree Example")
text(treepruned.linearsingle, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postscript plot of tree  
# (still not super attractive, saves it in current directory)
post(treepruned.linearsingle, file = "tree.ps", 
     title = "Classification Tree Example")

# Visualize (the first few layers of) the tree 
# We would need to adjust the complexity parameter cp
visual.pruned.tree <- prune(linear.singletree, cp = 0.003)
plot(visual.pruned.tree, uniform=TRUE, 
     main="Visualize The First Few Layers of The Tree")
text(visual.pruned.tree, use.n=TRUE, all=TRUE, cex=.8)

post(visual.pruned.tree, file = "visual_tree.ps", 
     title = "Visualize The First Few Layers of The Tree")


predictedW0 <- predict(treepruned.linearsingle, newdata=processed.scaled.testW0)

predictedW1 <- predict(treepruned.linearsingle, newdata=processed.scaled.testW1)

singletreecauseff <- predictedW1-predictedW0

# calculate MSE against Ystar
stMSEstar <- mean((processed.scaled.test$Ystar-singletreecauseff)^2)
print(c("MSE using ystar on test set of single tree",stMSEstar))
MSElabelvec <- append(MSElabelvec,"single tree")
MSEvec <- append(MSEvec,stMSEstar)



###################################
# Causal Tree #
###################################



# Set parameters
split.Rule.temp = "CT"
cv.option.temp = "CT"
split.Honest.temp = T
cv.Honest.temp = T
split.alpha.temp = .5
cv.alpha.temp = .5
split.Bucket.temp = T
bucketMax.temp= 100
bucketNum.temp = 5
minsize.temp=50

# get the dishonest version--estimated leaf effects on training sample
CTtree <- causalTree(as.formula(paste("Y~",sumx)), 
                   data=processed.scaled.train.1, treatment=processed.scaled.train.1$W, 
                   split.Rule=split.Rule.temp, split.Honest=split.Honest.temp, 
                   split.Bucket=split.Bucket.temp, bucketNum = bucketNum.temp, 
                   bucketMax = bucketMax.temp, cv.option=cv.option.temp, cv.Honest=cv.Honest.temp, 
                   minsize = minsize.temp, 
                   split.alpha = split.alpha.temp, cv.alpha = cv.alpha.temp, 
                   HonestSampleSize=nrow(processed.scaled.train.2),
                   cp=0)
opcpid <- which.min(CTtree$cp[,4])
opcp <- CTtree$cp[opcpid,1]
tree_dishonest_CT_prune <- prune(CTtree, cp = opcp) 

# we can get the honest tree manually, by estimating the leaf effects on a new sample
tree_honest_CT_prune2 <- estimate.causalTree(object=tree_dishonest_CT_prune,
                                             data=processed.scaled.train.1, 
                                             treatment=processed.scaled.train.1$W)

print(tree_honest_CT_prune2)

processed.scaled.train.1$leaffact <- as.factor(round(predict(tree_dishonest_CT_prune, 
                                        newdata=processed.scaled.train.1,type="vector"),4))
processed.scaled.train.2$leaffact <- as.factor(round(predict(tree_dishonest_CT_prune, 
                                                             newdata=processed.scaled.train.2,type="vector"),4))
processed.scaled.test$leaffact <- as.factor(round(predict(tree_dishonest_CT_prune, 
                                                             newdata=processed.scaled.test,type="vector"),4))

# These show leaf treatment effects and standard errors; can test hypothesis that leaf 
# treatment effects are 0
summary(lm(Y~leaffact+W*leaffact-W-1, data=processed.scaled.train.1))
summary(lm(Y~leaffact+W*leaffact-W-1, data=processed.scaled.train.2))
summary(lm(Y~leaffact+W*leaffact-W-1, data=processed.scaled.test))

#This specification tests whether leaf treatment effects are different than average
summary(lm(Y~leaffact+W*leaffact-1, data=processed.scaled.train.2))


CTpredict = predict(tree_honest_CT_prune2, newdata=processed.scaled.test, type="vector")
# calculate MSE against Ystar
CTMSEstar <- mean((processed.scaled.test$Ystar-CTpredict)^2)
print(c("MSE using ystar on test set of single forest",CTMSEstar))
MSElabelvec <- append(MSElabelvec,"causal tree")
MSEvec <- append(MSEvec,CTMSEstar)




#################################
# Random Forest (Single Forest) # 
#################################
# use linear formula

# We first set the outcome variable as factor to indicate that
# we want to do classification with random forest
processed.scaled.train$Y <- as.factor(processed.scaled.train$Y)

# fitting model
set.seed(90)
fit.rf <- randomForest(linear, processed.scaled.train, ntree = numtreesCT, do.trace = 10)

summary(fit.rf)
print(fit.rf)
importance(fit.rf)

# plotting
plot(fit.rf)
plot(importance(fit.rf), lty = 2, pch = 16)
lines(importance(fit.rf))

# create partial dependence plots
imp = importance(fit.rf)
impvar = rownames(imp)[order(imp[, 1], decreasing=TRUE)]
op = par(mfrow=c(2, 2))
#for (i in seq_along(impvar)) {
#  partialPlot(fit.rf, processed.scaled.train, impvar[i], xlab=impvar[i],
#              main=paste("Partial Dependence on", impvar[i]),
#              ylim=c(0, 1))
#}

# Predict Output 
predicted = predict(fit.rf, processed.scaled.test)


par(mfrow=c(1,1))
varImpPlot(fit.rf)

# Note -- to get causal effects out of this, need to create a test set with W replaced by W=0, the same test set with W replaced by W=1, 
# and then take the difference


predictedW0 <- predict(fit.rf, processed.scaled.testW0, type="prob")

predictedW1 <- predict(fit.rf, processed.scaled.testW1, type="prob")

rfcauseff <- predictedW1[,1]-predictedW0[,1]


# calculate MSE against Ystar
rfMSEstar <- mean((processed.scaled.test$Ystar-rfcauseff)^2)
print(c("MSE using ystar on test set of single forest",rfMSEstar))
MSElabelvec <- append(MSElabelvec,"single forest")
MSEvec <- append(MSEvec,rfMSEstar)




#######################################
# Causal Forest

ncolx<-length(processed.scaled.train)-2 #total number of covariates
ncov_sample<-floor(2*ncolx/3) #number of covariates (randomly sampled) to use to build tree
# ncov_sample<-p #use this line if all covariates need to be used in all trees

# now estimate a causalForest
cf <- causalForest(as.formula(paste("Y~",sumx)), data=processed.scaled.train, 
                   treatment=processed.scaled.train$W, 
                   split.Rule="CT", double.Sample = T, split.Honest=T,  split.Bucket=T, 
                   bucketNum = 5,
                   bucketMax = 100, cv.option="CT", cv.Honest=T, minsize = 50, 
                   split.alpha = 0.5, cv.alpha = 0.5,
                   sample.size.total = floor(nrow(processed.scaled.train) / 2), 
                   sample.size.train.frac = .5,
                   mtry = ncov_sample, nodesize = 5, 
                   num.trees= numtreesCT,ncolx=ncolx,ncov_sample=ncov_sample
) 

cfpredtest <- predict(cf, newdata=processed.scaled.test, type="vector")

cfpredtrainall <- predict(cf, newdata=processed.scaled.train, 
                          predict.all = TRUE, type="vector")

# calculate MSE against Ystar
cfMSEstar <- mean((processed.scaled.test$Ystar-cfpredtest)^2)
print(c("MSE using ystar on test set of causalTree/causalForest",cfMSEstar))
mean(cfMSEstar)

print(c("mean of ATE treatment effect from causalForest on Training data", 
        round(mean(cfpredtrainall$aggregate),5)))

print(c("mean of ATE treatment effect from causalForest on Test data", 
        round(mean(cfpredtest),5)))


# use infJack routine from randomForestCI
# This gives variances for each of the estimated treatment effects; note tau is labelled y.hat
cfvar <- infJack(cfpredtrainall$individual, cf$inbag, calibrate = TRUE)
plot(cfvar)


# This code shows how to make a plot in two dimensions while holding others at their medians
namesD <- names(processed.scaled.train)
D = as.matrix(processed.scaled.train)
medians = apply(D, 2, median)

unique.yob = sort(unique(as.numeric(D[,"yob"])))
unique.totalpopulation_estimate = sort(unique(as.numeric(D[,"totalpopulation_estimate"])))
unique.vals = expand.grid(yob = unique.yob, totalpopulation_estimate = unique.totalpopulation_estimate)

D.focus = outer(rep(1, nrow(unique.vals)), medians)
D.focus[,"yob"] = unique.vals[,"yob"]
D.focus[,"totalpopulation_estimate"] = unique.vals[,"totalpopulation_estimate"]
D.focus = data.frame(D.focus)
numcol = ncol(D.focus)
names(D.focus) = namesD



direct.df = expand.grid(yob=factor(unique.yob), totalpopulation_estimate=factor(unique.totalpopulation_estimate))
direct.df$cate=  predict(cf, newdata=D.focus, type="vector", predict.all=FALSE)

heatmapdata <- direct.df
heatmapdata <- heatmapdata[,c("yob","totalpopulation_estimate","cate")]
heatmapdata <- heatmapdata[order(heatmapdata$yob),]
heatmapdata <- dcast(heatmapdata, yob~totalpopulation_estimate, mean)

heatmapdata <- heatmapdata[,!(names(heatmapdata) %in% c("yob"))]

#need to remove the labels from this heatmap--to do
heatmap(as.matrix(heatmapdata), Rowv=NA, Colv=NA, col = cm.colors(256), scale="column", margins=c(5,10),
        labCol<-rep("",ncol(heatmapdata)), labRow<-rep("",nrow(heatmapdata)))


# gg plot needs some massaging to make it look nice--to do
ggplot(direct.df, aes(yob,totalpopulation_estimate)) + geom_tile(aes(fill = cate)) 


# try another covariate
unique.yob = sort(unique(as.numeric(D[,"yob"])))
unique.percent_male = sort(unique(as.numeric(D[,"percent_male"])))
unique.vals = expand.grid(yob = unique.yob, percent_male = unique.percent_male)

D.focus = outer(rep(1, nrow(unique.vals)), medians)
D.focus[,"yob"] = unique.vals[,"yob"]
D.focus[,"percent_male"] = unique.vals[,"percent_male"]
D.focus = data.frame(D.focus)
numcol = ncol(D.focus)
names(D.focus) = namesD



direct.df = expand.grid(yob=factor(unique.yob), percent_male=factor(unique.percent_male))
direct.df$cate=  predict(cf, newdata=D.focus, type="vector", predict.all=FALSE)

heatmapdata <- direct.df
heatmapdata <- heatmapdata[,c("yob","percent_male","cate")]
heatmapdata <- heatmapdata[order(heatmapdata$yob),]
heatmapdata <- dcast(heatmapdata, yob~percent_male, mean)

heatmapdata <- heatmapdata[,!(names(heatmapdata) %in% c("yob"))]

#need to remove the labels from this heatmap--to do
heatmap(as.matrix(heatmapdata), Rowv=NA, Colv=NA, col = cm.colors(256), scale="column", margins=c(5,10),
        labCol<-rep("",ncol(heatmapdata)), labRow<-rep("",nrow(heatmapdata)))


# gg plot needs some massaging to make it look nice--to do
ggplot(direct.df, aes(yob,percent_male)) + geom_tile(aes(fill = cate)) 



################Summary output##########################################################
print(MSElabelvec)
print(MSEvec)



