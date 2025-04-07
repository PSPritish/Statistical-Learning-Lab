library(ISLR)
library(ggplot2)
library(MASS)
head(Smarket)
attach(Smarket)
# Preliminary analysis
unique(Direction)
unique(Year)
dim(Smarket)
?Smarket
boxplot(Direction,)
ggplot(Smarket ,aes(x=Volume,y= Direction , fill = Direction)) + geom_boxplot()
ggplot(Smarket ,aes(x=Volume  )) + geom_histogram()
lda_fit <- lda(Direction ~ Lag1 + Lag2, data = Smarket , subset = Year<2005)
lda_fit
par(mar = c(1,1,1,1)) # mar stands for margins
plot(lda_fit)
subset.2005 <- subset(Smarket, Year == 2005)
pred <- predict(lda_fit , newdata= subset.2005)
pred.df <- data.frame(pred)
head(pred.df)
?predict.lda
table(Actual = subset.2005$Direction,Predicted = pred.df$class)
