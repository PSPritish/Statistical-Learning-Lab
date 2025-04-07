library(ISLR)
library(ggplot2)
library(boot)
set.seed(2)
# Randomly sample 196 obs out of 392 observations for training set
head(Auto)
dim(Auto)
train  = sample(392,196)
Auto.tr <- Auto[train,]
dim(Auto.tr)
# Fir LInear Model on training data
lm.fit <- lm(mpg~horsepower , data = Auto , subset = train)
summary(lm.fit)
# Calculate MSE FOR validation dataset
Auto.test <- Auto[-train,]
pred <- predict(lm.fit,newdata= Auto.test)
mse <- mean((Auto.test$mpg - pred)^2)
mse
# Calculate MSE FOR validation set
attach(Auto)
mean((mpg-predict(lm.fit,Auto))[-train]^2)

# Cross validation
?cv.glm
glm.auto <- glm(mpg~horsepower , data = Auto )
summary(glm.auto)

cv.err <- cv.glm(Auto,glm.auto)
cv.err$delta # this delta gives cross validation error 

cv.auto <- cv.glm(Auto,glm.auto, K = 5 )
cv.auto$delta
 


# LEave-One-Out Cross-Validation

coef(glm.auto)
coef(lm.fit)
library(boot)
glm.fit= glm(mpg~horsepower, data = Auto)
cv.err = cv.glm(Auto,glm.fit)
cv.err$delta
cv.error = rep(0,5)
for(i in 1:5){
    glm.fit = glm(mpg ~ poly(horsepower,i),data = Auto)
    cv.error[i] = cv.glm(Auto, glm.fit)$delta[1]
}
cv.error
plot(seq(1,5), cv.error , xlab = "Model order" , ylab = "MSE" , type = "o",col = "blue", pch = 16)

# K-Fold Cross Validation
set.seed(17)
cv.error.10 = rep(0,10)
for(i in 1:10){
  glm.fit = glm(mpg ~ poly(horsepower,i),data = Auto)
  cv.error.10[i] = cv.glm(Auto, glm.fit,K=10)$delta[1]
}
cv.error.10

# Bootstrap
# Define the function to estimate the coefficients
boot.fn <- function(data,index){
   # Fit the linear regression midel with mpg as the response and horsepower as the pre
   fit <- lm(mpg ~ horsepower , data = data , subset = index)
   return(coef(fit))
}
# Perform bootstrap with 1000 resamples
results <- boot(Auto, boot.fn, R = 1000)
print(results)
par(mfrow=c(1,2)) # BAsically divides our graph into two parts mtlb row to 1 hi rahega but columns do kr dega
hist(results$t[,1], main = "Bootstrap Distribution of Intercept" , xlab = "Intercept")
hist(results$t[,2], main = "Bootstrap Distribution of Slope", xlab = "Slope")

# Cross-validation
# Create cost function for cross-validation
cost <- function(r, pi = 0) mean(abs(default-pi))
