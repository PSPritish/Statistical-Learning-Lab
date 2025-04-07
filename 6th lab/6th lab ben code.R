---
  title: "Subset Selection and Dimension Reduction"
author: "Ben Abraham Biju"
date: "2025-02-11"
output: html_document
---
  
  ## Statistical Learning Lab
  *Best Subset selection method*
  
  {r}
library(ISLR)
library(leaps)
head(Hitters)
dim(Hitters)

Hitters = na.omit(Hitters)
regfit.3 <- regsubsets(Salary ~., data= Hitters, nvmax = 3)
summary(regfit.3)


Let us try a few more variables with best subset selection

{r}
regfit <- regsubsets(Salary ~., data= Hitters)
summary(regfit)
regfit.full <- regsubsets(Salary ~., data= Hitters, nvmax =19)
reg.sum <- summary(regfit.full)
names(reg.sum)
which.min(reg.sum$cp)
coef(regfit.full, 10)


{r}
plot(reg.sum$cp, xlab ="Number of variables included", ylab="Cp")
points(10, reg.sum$cp[10], pch=10, col="red")


Let us use the plot function of leaps library
{r}
plot(regfit.full, scale="Cp")


Forward Selection Method

{r}
regfit.fwd <- regsubsets(Salary ~., data= Hitters, nvmax =19, method ="forward")
summary(regfit.fwd)
plot(regfit.fwd)
regsum.fwd <- summary(regfit.fwd)
which.min(regsum.fwd$bic)
coef(regfit.fwd, 6)           # from this we can have all coeficients and can check the deviance between 10 variables and 6 variables model to choose which one is the best




{r}
library(ISLR)
library(glmnet)
library(dplyr)
library(tidyr)

View(Hitters)
head(Hitters)

#set up data

x = model.matrix(Salary~.,Hitters)[,-1] #trim off the salary column  and model.matrix is used to convert dataframe into matrix (design matrix)
#leaving only the predictors
head(x)

y = Hitters %>%
  select(Salary)%>%                # salary as the response variable
  unlist %>%
  as.numeric()                      # convert into numeric vector
head(y)  

grid = 10^seq(10, -2, length = 100)  # it takes 100 values from -2 to 10 and stored in grid variable which is use for lambda

ridge_mod = glmnet(x, y, alpha = 0, lambda = grid) # for alpha =0 it is ridge regression
dim(coef(ridge_mod))
plot(ridge_mod, xvar="lambda", label=TRUE)
plot(ridge_mod)                                  #draw plot of coefficients
cv.ridge = cv.glmnet(x,y,alpha=0)
plot(cv.ridge)
best_lamda = cv.ridge$lambda.min
best_lamda
ridge_mod$lamda[50] #display 0 th lambda value
coef(ridge_mod)[,50] #display coefficients associated with the 50 th lambda value.
sqrt(sum(coef(ridge_mod)[-1,50]^2)) #calculate l2 norm


#visualize the shrinking in coefficients with increase in lambda
l2_norm = apply(coef(ridge_mod)[-1, ], 2, function(coefs) sqrt(sum(coefs^2)))
plot(ridge_mod$lambda, l2_norm, type = "b",log="x",xlab="lambda(log scale)",ylab="L2 Norm of Coefficients")


set.seed(1)
train = Hitters%>%
  sample_frac(0.5)
test = Hitters %>%
  setdiff(train)

# Create model matrices for training and testing data
x_train <- model.matrix(Salary ~ ., train)[, -1]
x_test <- model.matrix(Salary ~ ., test)[, -1]

# Extract response variables for training and testing data
y_train <- train %>%
  select(Salary) %>%
  unlist() %>%
  as.numeric()

y_test <- test %>%
  select(Salary) %>%
  unlist() %>%
  as.numeric()


# Fit Ridge Regression model
ridge_mod = glmnet(x_train, y_train, alpha = 0, lambda = grid, thresh = 1e-12)
ridge_pred=predict(ridge_mod,s=4,newx=x_test)
mean((ridge_pred-y_test)^2)
plot(ridge_mod,xvar="lambda",label=TRUE)

# with large value of lambda
ridge_pred = predict(ridge_mod, s = 1e10, newx =x_test)
mean((ridge_pred -y_test)^2)

# check with least square
ridge_pred =predict(ridge_mod, s=0, newx = x_test, exact =T, x=x_train, y=y_train)
mean((ridge_pred - y_test)^2)

ridge_pred =predict(ridge_mod, s=0, newx = x_test, exact =T, x=x_train, y=y_train)[1]

ridge_pred

# selection of tuning parameter lambda
set.seed(1)
cv.out = cv.glmnet(x_train, y_train, alpha =0 )# Fit ridge regression model on training data

bestlam = cv.out$lambda.min # select lambda that minimizes training MSE
bestlam
plot(cv.out)



