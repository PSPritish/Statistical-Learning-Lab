View(manufacturing)
df<- manufacturing
head(df)
colnames(df)
library(ISLR)
library(ggplot2)
library(boot)
#LOOCV
cv.error = rep(0,5)
for(i in 1:5){
  glm.fit = glm(Quality.Rating ~ poly(Temperature...C.,i),data = df)
  cv.error[i] = cv.glm(df, glm.fit)$delta[1]
}
cv.error
# K fold cross validation
cv.error.5 = rep(0,5)
for(i in 1:5){
  glm.fit = glm(Quality.Rating ~ poly(Temperature...C.,i),data = df)
  cv.error.5[i] = cv.glm(df, glm.fit, K =5)$delta[1]
}
cv.error.5

cv.error.10 = rep(0,5)
for(i in 1:5){
  glm.fit = glm(Quality.Rating ~ poly(Temperature...C.,i),data = df)
  cv.error.10[i] = cv.glm(df, glm.fit, K =10)$delta[1]
}
cv.error.10


my_table <- data.frame(
  Degree = 1:5,
  MSE_LOOCV = cv.error,
  MSE_5FoldCV = cv.error.5,
  MSE_10FoldCV = cv.error.10
)
print(my_table)

library(ggplot2)
library(reshape2)

# Convert data for ggplot
cv_results_melt <- melt(my_table, id.vars = "Degree", variable.name = "CV_Type", value.name = "CV_Error")

# Plot
ggplot(cv_results_melt, aes(x = Degree, y = CV_Error, color = CV_Type, group = CV_Type)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  labs(title = "Cross-Validation Errors vs Polynomial Degree",
       x = "Degree of Polynomial",
       y = "CV Error") +
  theme_minimal()

library(boot)   # For cross-validation
library(ggplot2)

# Define different predictor sets
models <- list(
  "Temp" = "Quality.Rating ~ Temperature...C.",
  "Temp-Press" = "Quality.Rating ~ Temperature...C. + Pressure..kPa.",
  "Temp-MatFus" = "Quality.Rating ~ Temperature...C. + Material.Fusion.Metric",
  "Temp-MatFus-MatTrans" = "Quality.Rating ~ Temperature...C. + Material.Fusion.Metric + Material.Transformation.Metric",
  "Temp-Press-MatFus-MatTrans" = "Quality.Rating ~ Temperature...C. + Pressure..kPa. + Material.Fusion.Metric + Material.Transformation.Metric"
)

# Initialize error storage
cv_errors <- data.frame(Model = character(), CV_Error = numeric())

# Perform 5-Fold Cross-Validation
for (name in names(models)) {
  formula <- as.formula(models[[name]])  # Convert to formula
  glm.fit <- glm(formula, data = df)  # Fit model
  cv_result <- cv.glm(df, glm.fit, K = 5)  # Cross-validation
  cv_errors <- rbind(cv_errors, data.frame(Model = name, CV_Error = cv_result$delta[1]))
}

# Print results
print(cv_errors)





ggplot(cv_errors, aes(x = Model, y = CV_Error, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Cross-Validation Errors for Different Models",
       x = "Model (Predictor Combinations)",
       y = "CV Error") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12))

set.seed(3)
population_data <- rnorm(50, mean = 50, sd = sqrt(2))
# Bootstrap: 100 samples of size 20
boot_means <- boot_vars <- numeric(100)              
for (i in 1:100) {
  samp <- sample(data, 20, replace = TRUE)
  boot_means[i] <- mean(samp)
  boot_vars[i] <- var(samp)
  
  # Check if variance is NA and replace with 0 if true
  if (is.na(boot_vars[i])) {
    boot_vars[i] <- 0
  }
}
cat("Estimated Mean:", mean(boot_means), "\n")
cat("Estimated Variance:", mean(boot_vars), "\n")
