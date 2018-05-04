setwd("J:/Spring2018/IST597")

firedata = read.csv("logreg_data.csv")
names(firedata)
#breast_cancer = read.csv("breast_cancer.csv")

library(randomForest)
library(e1071)
library(gbm)
library(ada)
library(rpart)
library(ROCR)
library(vegan)
library(dummies)
library(ada) 
library(pROC)
library(xgboost)


############################ Naive Bayes #######################################
################################################################################

firedata$Y = as.factor(firedata$Y)
# Divide into training and test
n <- dim(firedata)[1]
p <- dim(firedata)[2]

set.seed(2016)
test <- sample(n, round(n/4))
train <- (1:n)[-test]
fire.train <- firedata[train,]
fire.test <- firedata[test,]

firemodel = naiveBayes(fire.train$Y ~ ., data = fire.train)

class(model)
summary(model)
print(model)

### Prediction
preds1 <- predict(firemodel, newdata = fire.test)
preds <- predict(firemodel, newdata = fire.test, type = "raw")
score = preds[,2]
pred = prediction(score, fire.test$Y)
nb.prff = performance(pred, "tpr", "fpr")
plot(nb.prff)


## Prediction accuracy
conf_matrix = table(preds1, fire.test$Y)
conf_matrix
test.accuracy = (conf_matrix[1,1]+conf_matrix[2,2])/nrow(fire.test)
test.accuracy


############ Plots ############
###############################

par(mfrow = c(3,3))

rocplot <- function(pred, truth, ...){
  predob <- prediction(pred, truth)
  perf <- performance(predob, "tpr", "fpr")
  auc <- performance(predob,"auc")
  auc <- round(as.numeric(auc@y.values),5)
  plot(perf, colorize=TRUE,...)
  abline(a=0, b= 1)
  text(0.75, 0.25, auc, cex = .8)
  text(0.75, 0.35, "AUC", cex = .8)
}



############################ Boosting ##########################################
################################################################################

boost.fire = gbm(Y ~., data = fire.train, distribution="bernoulli", n.trees = 5000, interaction.depth = 1)
#summary(boost.fire)
yhat.boost.train = predict(boost.fire, newdata = fire.train, n.trees = 5000, type="response")

yhat.boost.train = round(yhat.boost.train)

rocplot(yhat.boost.test, fire.test$Y, main = "Test data")


############################ AdaBoost ##########################################
################################################################################


# Build best ada boost model 
model = ada(x = fire.train[,-1], 
            y = fire.train$Y, 
            iter=20, loss="logistic", verbose=TRUE) # 20 Iterations 

# Look at the model summary
model
summary(model)

# Predict on train data  
pred_Train  =  predict(model, fire.train[,-1])  

# Build confusion matrix and find accuracy   
cm_Train = table(fire.train$Y, pred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
#rm(pred_Train, cm_Train)

# Predict on test data
pred_Test = predict(model, fire.test[,-1])

# Build confusion matrix and find accuracy   
cm_Test = table(fire.test$Y, pred_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
#rm(pred_Test, cm_Test)

accu_Train
accu_Test

############################ XGBoost ###########################################
################################################################################

xgb_fire = xgboost(data = data.matrix(fire.train[,-1]), 
                   label = fire.train$Y,
                   eta = 0.1,
                   max_depth = 15,
                   nround = 25,
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "merror",
                   objective = "multi:softprob",
                   num_class = 12,
                   nthread = 3)

y_pred = predict(xgb_fire, data.matrix(fire.test[,-1]))

# Let's see what actaul tree looks like:
model = xgb.dump(xgb_fire, with_stats = T)
# Top 10 nodes of the model

model[1:10] 

# Get the feature real names
names = dimnames(data.matrix(fire.train,-1))[[2]]


# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb_fire)

# Graph
xgb.plot.importance(importance_matrix[1:10,])


# output_vector = fire.train[, 1] == "Responder"
# test <- chisq.test(fire.train$TSURF_anom_May, output_vector)
# print(test)

############################ Random Forest #####################################
################################################################################


bestmtry <- tuneRF(fire.train[-1], fire.train$Y, ntreeTry=100, 
                   stepFactor=1.5, improve=0.01, dobest=FALSE, plot = F)

## RF on Training Data

rf.fire = randomForest(Y~., data = fire.train, mtry=8, ntree= 1000, importance = TRUE)
yhat.rf.train = predict(rf.fire, type = "prob", newdata = fire.train)[,2]
yhat.rf.train1 = predict(rf.fire, newdata = fire.train)

plot(rf.fire, main = "Random Forest Model: Number of Trees Vs. Error")
#par(mfrow = c(1,2))

# ROC curve
roc.Train <- roc(fire.train$Y, yhat.rf.train, direction = "<")
plot.roc(roc.Train, col = "blue", auc.polygon = TRUE, main = "Full model, train", xlab = "False Positive Rate", ylab = "True Positive Rate", print.auc = TRUE)

# Misclassification rate (FP+FN)/total
confusion <- table(yhat.rf.train1, fire.train$Y)
train.accuracy = (confusion[1,1]+confusion[2,2])/nrow(fire.train)

train.error = round((confusion[2,1]+confusion[1,2])/(sum(confusion[,1])+sum(confusion[,2])),3)
train.error
train.accuracy

## RF on Test Data

yhat.rf.test = predict(rf.fire, type = "prob", newdata = fire.test)[,2]
roc.test <- roc(fire.test$Y, yhat.rf.test, direction = "<")

yhat.rf.test1 = predict(rf.fire, newdata = fire.test)

# ROC curve
plot.roc(roc.test, col = "blue", auc.polygon = TRUE, main = "Full model, test", xlab = "False Positive Rate", ylab = "True Positive Rate", print.auc = TRUE)

# Misclassification rate (FP+FN)/total
confusion <- table(yhat.rf.test1, fire.test$Y)
test.accuracy = (confusion[1,1]+confusion[2,2])/nrow(fire.test)
test.error = round((confusion[2,1]+confusion[1,2])/(sum(confusion[,1])+sum(confusion[,2])),3)

train.accuracy
test.accuracy

importance(rf.fire)
varImpPlot(rf.fire, type = 1)


############################ Support Vector Machine ############################
################################################################################

set.seed(2016)
test <- sample(n, round(n/4))
train <- (1:n)[-test]
fire.train <- firedata[train,]
fire.test <- firedata[test,]

#--------------------------------------------------
# Linear kernel
#--------------------------------------------------


tune.out <- tune(svm, Y ~., data = fire.train, kernel = "linear", ranges = list(cost = c(.001,.01,.1,1,5)))

# Look for a best model
summary(tune.out)
bestmod <- tune.out$best.model
summary(bestmod)

ypred <- predict(bestmod, fire.test)
table(predict = ypred, truth = fire.test$Y)

svmfit.best1 <- svm(Y~., data = fire.train, kernel = "linear", cost = bestmod$cost, decision.values = T)
fitted1 <- attributes(predict(svmfit.best1, fire.train, decision.values = T))$decision.values
fitted.test1 <- attributes(predict(svmfit.best1, fire.test, decision.values = T))$decision.values

rocplot(fitted1, fire.train$Y, main = "Training")
rocplot(fitted.test1, fire.test$Y, main = "Test")


decisionplot(svmfit.best1, fire.train, class = "Y", main = "SVD (linear)")


#--------------------------------------------------
# Radial kernel
#--------------------------------------------------

tune.out2 <- tune(svm, Y ~., data = fire.train, kernel = "radial", ranges = list(cost = c(.001,.01,.1), gamma = c(1,5,50)))

bestmod <- tune.out2$best.model
ypred <- predict(bestmod, fire.test)
table(predict = ypred, truth = fire.test$Y)

svmrad2 <- svm(Y~., data = fire.train, kernel = "radial", gamma = tune.out2$best.model$gamma, cost = tune.out2$best.model$cost, decision.values = T)
fitted2 <- attributes(predict(svmrad2, fire.train, decision.values = T))$decision.values
fitted.test2 <- attributes(predict(svmrad2, fire.test, decision.values = T))$decision.values

rocplot(fitted2, fire.train$Y, main = "Training")
rocplot(fitted.test2, fire.test$Y, main = "Test")

#--------------------------------------------------
# Polynomial kernel
#--------------------------------------------------

tune.out3 <- tune(svm, Y ~., data = fire.train, kernel = "polynomial", ranges = list(cost = c(.001,.01,.1,1), degree = c(2,3)))
bestmod <- tune.out3$best.model
ypred <- predict(bestmod, fire.test)
table(predict = ypred, truth = fire.test$Y)

svmrad3 <- svm(Y~., data = fire.train, kernel = "polynomial", cost = tune.out3$best.model$cost, degree = tune.out3$best.model$degree, decision.values = T)
fitted3 <- attributes(predict(svmrad3, fire.train, decision.values = T))$decision.values
fitted.test3 <- attributes(predict(svmrad3, fire.test, decision.values = T))$decision.values

rocplot(fitted3, fire.train$Y, main = "Training")
rocplot(fitted.test3, fire.test$Y, main = "Test")


##################################################################################
##---------------------------- Variable Selection ------------------------------## 
##------------------------------------------------------------------------------##
##################################################################################

grid=10^seq(10,-2,length=100)


X.train = fire.train[,-1]
X.train = as.matrix(X.train)
y.train = fire.train$Y

X.test = fire.test[,-1]
X.test = as.matrix(X.test)
y.test = fire.test$Y

lasso.mod = glmnet(X.train, y.train, alpha=1, lambda = grid, family = "binomial")

plot(lasso.mod, xvar="lambda", label=TRUE) 

set.seed(1)

cv.out = cv.glmnet(X.train, y = y.train, alpha=1)

plot(cv.out, xvar="lambda",label=TRUE) 

bestlam=cv.out$lambda.min

lasso.pred = predict(lasso.mod, s = bestlam, newx= X.test)
mean((lasso.pred- y.test)^2)
out=glmnet(X.train, y.train, alpha=1,lambda=grid)
lasso.coef=predict(out, type="coefficients",s=bestlam)[1:73, ]

lasso.coef[lasso.coef == 0]




##### Exploratory: 

par(mfrow = c(1,1))

hist(firedata$EVLAND_Jun, main = "Evap June")

hist(firedata$SFMC_Jun, main = "Moist June")

hist(firedata$TSURF_Jun, main = "Temp June")

hist(firedata$PRECTOT_Jun, main = "Precip June")

hist(firedata$TSURF_anom_Jun, main = "Temp Anomaly June")

hist(firedata$PRECTOT_anom_Jun, main = "Precip Anomaly June")


x=matrix(rnorm (200*2) , ncol=2)
x[1:100,]=x[1:100,]+2 
x[101:150 ,]=x[101:150,] -2 
y=c(rep (1,150) ,rep (2,50) ) 
dat=data.frame(x=x,y=as.factor(y))