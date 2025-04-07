head(Auto)

#forward selection
null_model <- lm(mpg ~1 ,data = mtcars)
summary(null_model)
fwd <- step(null_model , direction = "forward" ,data = mtcars)
summary(fwd)

full_model <- lm(mpg~.,data = mtcars)
summary(full_model)

bwd <- step(null_model,direction = "backward")
summary(bwd)
