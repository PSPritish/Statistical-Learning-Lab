library(ISLR2)
head(Default)
m1 <- glm(default ~ . , family = binomial , data = Default)
summary(m1)
deviance(m1)
logLik(m1)

m2 <- glm(default ~ student + balance , family = binomial , data = Default) #when only two input
summary(m2)
deviance(m1) - deviance(m2)
qchisq(0.95, 1)

library(ggplot2)
library(boot)
library(MASS)

data("Default")
head(Default)
cor(Default)
str(Default)
num_data <- Default[,sapply(Default,is.numeric)] #select numeric columns
cor_matrix <- cor(num_data , use = "complete.obs")
print(cor_matrix)

install.packages("corrplot") # install the package
library(corrplot) #load the package
corrplot(cor_matrix , method = "circle" , addCoef.col = "black")

#Logistic Regression
glm.fits = glm(default~ student + balance , family = binomial , data = Default)
summary(glm.fits)
coef(glm.fits)
library(stats)
log_likelihood <- logLik(glm.fits)
log_likelihood
summary(glm.fits)$coef
summary(glm.fits)$coef [,4]
glm.probs <- predict(glm.fits , Default , type = "response")
glm.probs[1:10]
contrasts(Default$default)
glm.pred <- ifelse(glm.probs > 0.5 , "Yes" , "No") #convert into binary predictions
glm.pred
table(glm.pred , Default$default) #to find accuracy , sum of diagonal elements divided by total sum in confusion matrix
mean(glm.pred == Default$default)

#Splitting the dataset
trn <- sample(dim(Default)[1], 8000)
trn
Default_train <- Default[trn,]
Default_test <- Default[-trn,]
head(Default_test)
Default_test <- Default_test[,-1]
head(Default_test)

dim(Default_train)
dim(Default_test)

#fit llogistic regression model
glm.fit2 = glm(default~ student + balance , family = binomial , data = Default_train)

#predict using test dataset
pred <- predict(glm.fit2 , Default_test , type = "response")
pred[1:8]

#Classify the prediction in default = "Yes" or "No"
pred_class <- ifelse(pred>=0.5 , "Yes" , "No")
pred_class[1:8]

#Create confusion matrix
table(Default[-trn,]$default , pred_class)

#Test Accuracy
mean(pred_class == Default[-trn,]$default)
 
precision <- 20/(6+20) # true positive / by sum of true positive and false positive
#Precison and F1 score and recall ye sab dekh lena
contrasts(Default$default)

?predict.glm

head(iris)
unique(iris$Species)
library(VGAM)
library(nnet)

m_multi <- multinom(Species~ ., data = iris)
summary(m_multi)


?cv.glm

attach(Default) # mujhe baar baar dataset explicitly mention krne ki jarurat nhi padegi

#Exploratory Data Analysis using boxplot
boxplot(balance ~ default)
boxplot(income ~ default)
ggplot(Default , aes(x=default , y= balance , fill = default)) + geom_boxplot()





