library(ISLR)

# Load the LungCapData2 dataset
LungCapData2 <- read.csv(file="D:/Sem study materials/study materials 6th sem/Study materials by me/Statistical Learning Lab/7th lab/LungCapData2.csv")



# Create Age categories (step function bins)
Age_cat <- cut(LungCapData2$Age, breaks = c(0,3, 6, 9, 12, 15))

# Fit a linear regression model with Age categories
model <- lm(LungCap ~ Age_cat, data = LungCapData2)

# Summarize the model
summary(model)

#Alternative Step Function
table(cut(LungCapData2$Age,4))
fit_step = lm(LungCap~cut(Age,4), data = LungCapData2)
print(coef(summary(fit_step)))
# Predict the value of the generated Ages, returning the standard error using se = TRUE
# Get min/max values of Age using the range() function
library(dplyr)  

Agelims <- LungCapData2 %>%
  select(Age) %>%
  range()

Age_grid = seq(from = min(Agelims), to = max(Agelims))
preds = predict(fit_step, newdata = list(Age = Age_grid), se = TRUE)

# Compute error bands (2*SE)
se_bands = cbind("upper" = preds$fit+2*preds$se.fit,
                 "lower" = preds$fit-2*preds$se.fit)

# Plot
library(ggplot2)
ggplot() +
  geom_point(data = LungCapData2, aes(x = Age, y = LungCap)) +
  geom_line(aes(x = Age_grid, y = preds$fit), color = "red") +
  geom_ribbon(aes(x = Age_grid,
                  ymin = se_bands[,"lower"],
                  ymax = se_bands[,"upper"]),
              alpha = 0.3) +
  xlim(Agelims) +
  labs(title = "Step Function")