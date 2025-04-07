library(MASS)
library(ISLR2)
is.na(Boston)
names(Boston)
?Boston
fit1 <- lm(medv ~ lstat , data = Boston)
summary(fit1)
anova(fit1)
predict(fit1, data.frame(lstat = c(5,10,15)))
predict(fit1, data.frame(lstat = c(5,10,15)),interval = "confidence")
dim(Boston)
train <- sample(nrow(Boston),404) #404 is 80% of 506 data
train_boston <- Boston[train,]
test_boston <- Boston[-train,]
fit2 <- lm(medv ~. , data = train_boston)
summary(fit2)
dummy <- subset(test_boston,select = -medv)
head(dummy)

pred <- predict(fit2 , newdata = dummy )
mean((test_boston$medv - pred)^2)
