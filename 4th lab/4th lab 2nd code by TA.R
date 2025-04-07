library(MASS)
library(ggplot2)
attach(iris)
str(iris)
head(iris)

# Scaling the dataset
iris[1:4] <- scale(iris[1:4]) # Standardization , Lda ke liye must hai scaling but not for qda
head(iris)

# To returns a vector of the means and standard deviation of the variables 
set.seed(1)
sample <- sample(c(TRUE , FALSE) , nrow(iris),replace = TRUE,prob = c(0.7,0.3))
sample
# train and test sample
train <- iris[sample,]
test <- iris[!sample,]

model <- lda(Species~.,data = train)
model
#Quadratic discriminant analysis
qmodel <- qda(Species~., data = train)
qmodel
qpredicted <- predict(qmodel,test)
qp1 <- predict(qmodel, train)$class
qp1
qtab <- table(Predicted = qp1 , Actual = train$Species)
qtab
qp2 <- predict(qmodel, test)$class
qp2
qtab1 <- table(Predicted = qp2 , Actual = test$Species)
qtab
sum(diag(qtab1))/sum(qtab1)
