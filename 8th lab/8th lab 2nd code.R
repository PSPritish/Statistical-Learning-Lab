library(randomForest)
library(MLmetrics)
library(MASS)

## Classification 

# Heart DAta
data(heart)
?heart
Heartdata <- as.data.frame(heart) # convert dataset into dataframe
Heartdata

# Splitting
set.seed(1)
sample <-sample(c(TRUE, FALSE), nrow(Heartdata), replace = TRUE , prob=c(0.7,0.3))
train <- Heartdata[sample,]
test <- Heartdata[-sample,]

# Training the model
rf.heart <- randomForest(factor(y)~.,data=train)
summary(rf.heart)
rf.heart

#Confusion Matrix
ypred<-predict(rf.heart,test,type='class')
table(predict=ypred,truth=test$y)

# Metrics
ConfusionMatrix(ypred,test$y)
Accuracy(ypred,test$y)
Precision(ypred,test$y)
Recall(ypred,test$y)
F1_Score(ypred,test$y)

rf.heart<- randomForest(factor(y)~.,mtry = 6 ,data=train,importance = TRUE)


### REGRESSION
data("Boston")
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(Boston), replace = TRUE , prob=c(0.7,0.3))
