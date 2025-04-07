library(ISLR)

# Load the Wage dataset
LungCapData2 <- read.csv(file="D:/Sem study materials/study materials 6th sem/Study materials by me/Statistical Learning Lab/7th lab/LungCapData2.csv")


# Create age categories (step function bins)
LungCapData2$Age <- cut(LungCapData2$Age, breaks = c( 3 , 6, 9 ,12,15, 19))

# Fit a linear regression model with age categories
model <- lm(LungCap ~ Age, data = LungCapData2)

# Summarize the model
summary(model)

#Alternative Step Function

table(cut(LungCapData2$Age,5))
fit_step = lm(LungCap~cut(Age,5), data = LungCapData2)
print(coef(summary(fit_step)))
# Predict the value of the generated ages, returning the standard error using se = TRUE
# Get min/max values of age using the range() function
library(dplyr)  

agelims <- LungCap %>%
  select(Age) %>%
  range()

age_grid = seq(from = min(agelims), to = max(agelims))
preds = predict(fit_step, newdata = list(age = age_grid), se = TRUE)

# Compute error bands (2*SE)
se_bands = cbind("upper" = preds$fit+2*preds$se.fit, 
                 "lower" = preds$fit-2*preds$se.fit)

# Plot
library(ggplot2)
ggplot() +
  geom_point(data = LungCapData2, aes(x = age, y = LungCap)) +
  geom_line(aes(x = age_grid, y = preds$fit), color = "red") +
  geom_ribbon(aes(x = age_grid, 
                  ymin = se_bands[,"lower"], 
                  ymax = se_bands[,"upper"]), 
              alpha = 0.3) +
  xlim(agelims) +
  labs(title = "Step Function")
