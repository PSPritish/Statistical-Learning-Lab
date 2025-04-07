#LungCapData2
LungCapData2 <- read.csv(file="D:/Sem study materials/study materials 6th sem/Study materials by me/Statistical Learning Lab/7th lab/LungCapData2.csv")
head(LungCapData2)
summary(LungCapData2)
attach(LungCapData2)
# Make a plot of Lungcap vs. Age
plot(Age, LungCap, main="RElation", las =2)
#Smoke<- as.factor(Smoke)

# Fitting a linear regression model
model1<- lm(LungCap ~ Age)
summary(model1)
lines(smooth.spline(Age,predict(model1)), col="yellow",lwd =3)
#pred_values = predict(model1)

# Fitting a polyhomial linear regression model with degree 2
model2 <- lm(LungCap ~ Age + I(Age^2))
summary(model2)
lines(smooth.spline(Age,predict(model2)), col="blue",lwd =3)

# Fitting a polyhomial linear regression model with degree 3
model3<- lm(LungCap ~ Age + I(Age^2) + I(Age^3))
summary(model3)
lines(smooth.spline(Age,predict(model3)), col="red",lwd =3)

# Add a legend
legend("topleft", legend=c("Degree1", "Degree2", "Degree3"),col=c("yellow","blue","red"),lwd=3, lty =1)

model4 <- lm(LungCap ~ poly(Age,degree=4,raw=T))
summary(model4)

# Again add a legend for model 4

#For higher degree of polynomial 
model5 <- lm(LungCap ~ poly(Age,degree=100,raw=T))
summary(model5)
lines(smooth.spline(Age,predict(model5)),col="purple",lwd = 3)

# Again add a legend for 

# testing of models , are they improving or not
anova(model1, model2)
anova(model2,model5)

# Also can be done like this
AgeSquare <- Age^2
models <- lm(LungCap~Age + AgeSquare)

