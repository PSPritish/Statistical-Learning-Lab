library(MASS)
library(ggplot2)
library(caTools)
library(dplyr)
library(class) # when we are working with classification problems
library(MASS)
Default
head(Default)
str(Default)
set.seed(10000)
# Unlike many of our previous methods, knn() required
Default$student = as.numeric(Default$student)  - 1 
Default$student
head(Default)
dim(Default)
default_index = sample(nrow(Default),5000)
default_train = Default[default_index,]
default_test = Default[-default_index,]
# training data 
X_default_train = default_train[,-1]
y_default_train = default_train$default
dim(y_default_train)

X_default_test = default_test[,-1]
y_default_test = default_test$default

predicted1 = knn(train = X_default_train,test = X_default_test, cl = y_default_train,k = 50)

predicted2 = knn(train = X_default_train, test = X_default_test , cl = y_default_train, k = 100)

library(caret)

accuracy = confusionMatrix(predicted2, y_default_test)$overall["Accuracy"]
accuracy

set.seed(42)
k_to_try = 1:100
acc_k = rep(x = 0 , times = length(k_to_try))

for(i in seq_along(k_to_try)){
  pred = knn(train = scale(X_default_train),test = scale(X_default_test), cl = y_default_train , k= k_to_try[i])
  
  # calculate accuracy using confusion matrix 
  cm = confusionMatrix(pred, y_default_test)
  acc_k[i] = cm$overall["Accuracy"]
}
acc_k

#plot accuracy
plot(acc_k, )