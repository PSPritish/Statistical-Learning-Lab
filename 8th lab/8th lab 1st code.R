library(catdata)# provide heart dataset
library(rpart) # building decison tree
library(rpart.plot)# plotting decison tree
library(MLmetrics)

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

# Training the Decision TRee model
tree.heart<- rpart(y~sbp+tobacco+ldl+adiposity+factor(famhist)+typea+obesity+alcohol+age,data=train, method='class')
summary(tree.heart)
tree.heart
rpart.plot(tree.heart, cex = 0.6)

# Confusion Metrics
ypred<- predict(tree.heart, test, type='class')
table(predict=ypred, truth=test$y)

# Metrics
ConfusionMatrix(ypred,test$y)
Accuracy(ypred,test$y)
Precision(ypred,test$y)
Recall(ypred,test$y)
F1_Score(ypred,test$y)

# Pruning
plotcp(tree.heart) # Plot complexity parameter (CP) vs error
tree.heart$cptable # Get CP table
index <- which.min(tree.heart$cptable[,"xerror"]) #Find optimal cp index
index
cpopt <- tree.heart$cptable[index,"CP"] # Optimal CP value
cpopt
opttree.heart <- prune(tree.heart,cp=cpopt)
rpart.plot(opttree.heart)

# Confusion Metrics
ypred<- predict(opttree.heart, test, type='class')
table(predict=ypred, truth=test$y)

# Metrics
ConfusionMatrix(ypred,test$y)
Accuracy(ypred,test$y)
Precision(ypred,test$y)
Recall(ypred,test$y)
F1_Score(ypred,test$y)



# regression
library(MASS)
data("Boston")
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(Boston), replace = TRUE , prob=c(0.7,0.3))
train <- Boston[sample,]
test <- Boston[-sample,]

# Training the model
tree.boston <- rpart(medv~. ,data = train , method='anova')
summary(tree.boston)
tree.boston
rpart.plot(tree.boston,cex=0.5, box.palette ="Browns")


#Metrics
ypred<- predict(tree.boston,test)
MAE(ypred,test$medv)
MSE(ypred,test$medv)
RMSE(ypred,test$medv)
MAPE(ypred,test$medv)
ss_total <- sum((test$medv - mean(test$medv))^2)  # Total sum of squares
ss_residual <- sum((test$medv - ypred)^2)  # Residual sum of squares

r_squared <- 1 - (ss_residual / ss_total)
print(r_squared)

# Pruning 
