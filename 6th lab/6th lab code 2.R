library(ISLR)
library(glmnet)
library(dplyr)
library(leaps)
head(Hitters)
hit<- Hitters
dim(hit)
hit = na.omit(hit)
# set up data i.e x and y variables
x = model.matrix(Salary~., hit)[,-1] # trim off the salary 
                                     # leaving only the predictors
head(x)
y = hit %>% select(Salary) %>% unlist() %>% as.numeric()
head(y)

grid = 10^seq(10, -2 , length = 100)

ridge_mod = glmnet(x, y , alpha = 0 , lambda = grid)
dim(coef(ridge_mod))
plot(ridge_mod, xvar= 'lambda', label = TRUE)
plot(ridge_mod)
