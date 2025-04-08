# Load necessary libraries
library(ISLR2)         # Access to datasets like 'Hitters', 'Auto', etc.
library(boot)          # Bootstrap & cross-validation
library(MASS)          # Stepwise regression, LDA/QDA
library(mice)          # Handling missing data
library(corrplot)      # Correlation matrix visualization
library(ggplot2)       # For data visualization
library(leaps)         # Subset selection in regression
library(glmnet)        # Lasso and ridge regression
library(dplyr)         # Data manipulation
library(tidyr)         # Data reshaping
library(splines)       # Regression splines
library(mgcv)          # Generalized Additive Models (GAMs)
library(rpart)         # Decision tree models
library(rpart.plot)    # Plotting decision trees
library(MLmetrics)     # ML model evaluation metrics
library(randomForest)  # Random forest models

head(Credit)
dfr <- Credit

# EDA
# Understanding the structure
str(dfr)
summary(dfr)
glimpse(dfr)
head(dfr)

# Check for Missing Values
colSums(is.na(dfr))
md.pattern(dfr)

# use vim if there is missing values

# Univariate Analysis
# Numerical variable histogram
ggplot(dfr, aes(x = Balance)) + 
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  theme_minimal()

# Boxplot to detect outliers
ggplot(dfr, aes(y = Income)) + 
  geom_boxplot(fill = "orange") +
  theme_minimal()

# Bivariate Analysis
data_without_catagorical <- dfr[,sapply(dfr, is.numeric)]
cor_matrix <- cor(data_without_catagorical)
print(cor_matrix)
corrplot(cor_matrix, method='circle')
pairs(data_without_catagorical)

# Boxplot for categorical vs numeric
ggplot(dfr, aes(x = Student, y = Balance)) +
  geom_boxplot(fill = 'salmon')+
  theme_classic()

# Scatterplot between numeric variables
ggplot(dfr, aes(x = Income, y = Balance)) +
  geom_point(color = "steelblue") +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal()

# Check for Duplicated and Outliers
sum(duplicated(dfr))

# Outlier using IQR method
Q1 <- quantile(dfr$Income, 0.25)
Q3 <- quantile(dfr$Income, 0.75)
IQR <- Q3 - Q1
outliers <- dfr$Income < (Q1 - 1.5 * IQR) | dfr$Income > (Q3 + 1.5 * IQR)
sum(outliers)

# Applying models
m1 <- lm(Balance ~ ., data = data_without_catagorical)
summary(m1)
aov(m1)
anova(m1)
confint(m1)

# 5. Residual Diagnostics
resid_values <- residuals(m1)
qqnorm(resid_values); qqline(resid_values)

# Diagnostic plots (Residuals, leverage, etc.)
par(mfrow = c(2, 2))
plot(m1)
par(mfrow = c(1, 1))  # Reset layout

# test train split
set.seed(123)
data <- data_without_catagorical
train_index <- sample(nrow(data), 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

m2 <- lm(Balance ~ ., data = train_data)
summary(m2)

#predict 
test_features <- test_data[,-7]
predictions <- predict(m2, newdata = test_features)

# mean squared error
mse <- mean((test_data$Balance - predictions)^2)
print(paste("Test MSE:", round(mse, 2)))

null_model <- lm(Balance ~ 1, data = data)              # Only intercept
full_model <- lm(Balance ~ ., data = data)              # All predictors

# Forward selection
forward <- step(null_model, direction = "forward", scope = list(lower = null_model, upper = full_model))
summary(forward)

# Backward elimination
backward <- step(full_model, direction = "backward")
summary(backward)

# Both directions
stepwise <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "both")
summary(stepwise)

#----------------------Logistics
data("OJ")  # OJ data from ISLR2

# Quick checks
head(OJ, 10)         # First 10 rows
str(OJ)              # Structure: note 'Purchase' is a factor with levels "CH" & "MM"
summary(OJ)          # Summary stats

# Extract numeric columns for correlation
num_cols <- OJ[, sapply(OJ, is.numeric)]
cor_matrix <- cor(num_cols)

# Visualize correlations
corrplot(cor_matrix)

# Full logistic model
m1 <- glm(Purchase ~ ., data = OJ, family = binomial)
summary(m1)

# Deviance and log-likelihood
dev_m1 <- deviance(m1)
logLik_m1 <- logLik(m1)
cat("Deviance (m1):", dev_m1, "\n")
cat("Log-Likelihood (m1):", logLik_m1, "\n")

# For illustration, let's only use PriceCH and PriceMM
m2 <- glm(Purchase ~ PriceCH + PriceMM, data = OJ, family = binomial)
summary(m2)

# Compare deviance of m1 and m2
dev_m2 <- deviance(m2)
cat("Deviance (m2):", dev_m2, "\n")

diff_dev <- dev_m1 - dev_m2
cat("Difference in Deviance:", diff_dev, "\n")

# Chi-square critical value for 1 degree of freedom at 95% confidence
cat("Chi-square cutoff (0.95, df=1):", qchisq(0.95, 1), "\n")
#If diff_dev > qchisq(0.95, 1), it suggests the dropped variables collectively have a significant contribution.

#5. Make Predictions on the Full Dataset
glm.probs <- predict(m2, OJ, type = "response")  # probabilities
head(glm.probs, 10)

# Convert to classes: CH if prob > 0.5, else MM
glm.pred <- ifelse(glm.probs > 0.5, "CH", "MM")
head(glm.pred, 10)

# Confusion matrix & accuracy
tab <- table(Predicted = glm.pred, Actual = OJ$Purchase)
tab
accuracy_full <- mean(glm.pred == OJ$Purchase)
cat("Full-data Accuracy:", accuracy_full, "\n")

set.seed(123)
train_idx <- sample(nrow(OJ), 800)  # e.g., 800 for training
OJ_train <- OJ[train_idx, ]
OJ_test  <- OJ[-train_idx, ]

dim(OJ_train)
dim(OJ_test)

# Let's reuse the reduced model for simplicity
glm.fit2 <- glm(Purchase ~ PriceCH + PriceMM, data = OJ_train, family = binomial)
summary(glm.fit2)

# Predict probabilities on test set
test_probs <- predict(glm.fit2, OJ_test, type = "response")

# Classify as CH / MM
test_pred_class <- ifelse(test_probs > 0.5, "CH", "MM")

# Confusion matrix
confusion <- table(Actual = OJ_test$Purchase, Predicted = test_pred_class)
confusion

# Test accuracy
test_accuracy <- mean(test_pred_class == OJ_test$Purchase)
cat("Test Accuracy:", test_accuracy, "\n")

# Extract confusion matrix elements
TP <- confusion["CH","CH"]
FP <- confusion["MM","CH"]
FN <- confusion["CH","MM"]
TN <- confusion["MM","MM"]

precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)

cat("Precision:", precision, "\n")
cat("Recall:   ", recall,    "\n")
cat("F1 Score: ", f1,        "\n")

#######################################
### 1. Setup & Initial Exploration  ###
#######################################
# Load required packages
library(ISLR)       # For Smarket dataset
library(ggplot2)    # For visualizations
library(MASS)       # For LDA (and QDA if needed)
library(caret)      # For confusion matrix and model evaluation
library(class)      # For KNN classification
library(dplyr)      # For data manipulation

# Load the Smarket dataset and inspect it
data(Smarket)
head(Smarket, 10)        # View first 10 rows
str(Smarket)             # Check structure: note Direction and Year
summary(Smarket)         # Summary statistics
unique(Smarket$Direction)# Unique values in Direction (e.g., "Up", "Down")
unique(Smarket$Year)     # Unique trading years
dim(Smarket)             # Dimensions (rows, columns)

#######################################
### 2. Exploratory Data Analysis (EDA)###
#######################################

# Boxplot: Compare Volume across market directions
ggplot(Smarket, aes(x = Direction, y = Volume, fill = Direction)) +
  geom_boxplot() +
  ggtitle("Boxplot of Volume by Market Direction") +
  theme_minimal()

# Histogram: Distribution of Volume
ggplot(Smarket, aes(x = Volume)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  ggtitle("Histogram of Volume") +
  theme_minimal()

#######################################
### 3. Logistic Regression Analysis   ###
#######################################
# Fit a logistic regression model predicting Direction using Lag1 and Lag2
logit_model <- glm(Direction ~ Lag1 + Lag2, data = Smarket, family = binomial)
summary(logit_model)       # Coefficient estimates and significance

# Check goodness-of-fit using deviance and log-likelihood
model_deviance <- deviance(logit_model)
model_logLik   <- logLik(logit_model)
cat("Deviance:", model_deviance, "\n")
cat("Log-Likelihood:", model_logLik, "\n")

#######################################
### 4. Linear Discriminant Analysis  ###
#######################################
# Fit LDA on data from years before 2005
lda_fit <- lda(Direction ~ Lag1 + Lag2, 
               data = Smarket, 
               subset = Year < 2005)
print(lda_fit)
# Optional: visualize the LDA results (e.g., plot densities)
par(mar = c(1,1,1,1))
plot(lda_fit)

# Test LDA: Predict on observations from the year 2005
subset2005 <- subset(Smarket, Year == 2005)
lda_pred   <- predict(lda_fit, newdata = subset2005)
pred_df    <- data.frame(lda_pred)  # Contains 'class', 'posterior', etc.
head(pred_df)

# Confusion matrix for LDA predictions
conf_matrix_lda <- table(Actual = subset2005$Direction, Predicted = lda_pred$class)
print(conf_matrix_lda)

#######################################
### 5. K-Nearest Neighbors (KNN)      ###
#######################################
# Prepare training and testing datasets:
# We'll use data before 2005 for training and 2005 data for testing.
train_idx <- subset(Smarket, Year < 2005)
test_idx  <- subset(Smarket, Year == 2005)

# Use Lag1 and Lag2 as predictors.
train_data  <- train_idx[, c("Lag1", "Lag2")]
test_data   <- test_idx[, c("Lag1", "Lag2")]
train_labels <- train_idx$Direction

# KNN classification for a specific k (e.g., k = 3)
set.seed(123)
knn_pred <- knn(train = train_data, test = test_data, cl = train_labels, k = 3)

# Evaluate KNN performance using a confusion matrix
conf_matrix_knn <- confusionMatrix(knn_pred, test_idx$Direction)
print(conf_matrix_knn)

# Optional: Search for optimal k by varying k and plotting accuracy
set.seed(42)
ks <- 1:20
accuracy_k <- sapply(ks, function(k) {
  pred_k <- knn(train = train_data, test = test_data, cl = train_labels, k = k)
  mean(pred_k == test_idx$Direction)
})
# Plot accuracy vs. number of neighbors
plot(ks, accuracy_k, type = "b", pch = 19, col = "blue",
     xlab = "Number of Neighbors (k)", ylab = "Accuracy", 
     main = "KNN Accuracy vs. k on Smarket Dataset")

# --------------------------
# 1. Setup & Data Preparation
# --------------------------
library(ISLR)        # Contains the Auto dataset
library(ggplot2)     # For data visualization
library(boot)        # For cross-validation and bootstrap
set.seed(2)          # For reproducibility

# Load and inspect the Auto dataset
data(Auto)
head(Auto)
dim(Auto)  # Total observations: 392

# ---------------------------------
# 2. Splitting Data into Training/Test Sets
# ---------------------------------
# Randomly sample 196 observations (50% of 392) for training
train_indices <- sample(392, 196)
Auto.tr <- Auto[train_indices, ]
dim(Auto.tr)

# ---------------------------------
# 3. Fit a Simple Linear Model using the training set
# ---------------------------------
lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train_indices)
summary(lm.fit)  # Check coefficient estimates, p-values, R-squared

# ---------------------------------
# 4. Validate Model Performance on the Test Set
# ---------------------------------
# Use the remaining data as the test set
Auto.test <- Auto[-train_indices, ]
# Predict mpg for test data using the fitted model
pred <- predict(lm.fit, newdata = Auto.test)
# Calculate Mean Squared Error (MSE)
mse <- mean((Auto.test$mpg - pred)^2)
cat("Test Set MSE:", mse, "\n")

# Alternatively, calculate MSE without detaching data:
attach(Auto)
mse_alternative <- mean((mpg - predict(lm.fit, Auto))[-train_indices]^2)
cat("Test Set MSE (alternative calculation):", mse_alternative, "\n")
detach(Auto)

# ---------------------------------
# 5. Cross-Validation using cv.glm
# ---------------------------------
# Fit a generalized linear model for cross-validation
glm.auto <- glm(mpg ~ horsepower, data = Auto)
summary(glm.auto)

# Leave-One-Out Cross-Validation (LOOCV)
cv_err_loocv <- cv.glm(Auto, glm.auto)$delta  # delta[1] is raw cv estimate
cat("LOOCV Error:", cv_err_loocv, "\n")

# 5-Fold Cross-Validation
cv.auto_5fold <- cv.glm(Auto, glm.auto, K = 5)$delta
cat("5-Fold CV Error:", cv.auto_5fold, "\n")

# ---------------------------------
# 6. Evaluating Model Complexity: Polynomial Regression
# ---------------------------------
# Compare models with increasing polynomial degree using CV
cv_error_poly <- rep(0, 5)
for (i in 1:5) {
  glm_poly <- glm(mpg ~ poly(horsepower, i), data = Auto)
  cv_error_poly[i] <- cv.glm(Auto, glm_poly)$delta[1]
}
cv_error_poly
# Plot CV error vs. polynomial degree
plot(1:5, cv_error_poly, xlab = "Polynomial Degree", 
     ylab = "CV Error (MSE)", type = "o", col = "blue", pch = 16,
     main = "LOOCV Error vs. Model Complexity")

# ---------------------------------
# 7. K-Fold Cross Validation: 10-Fold Example
# ---------------------------------
set.seed(17)
cv_error_10fold <- rep(0, 10)
for (i in 1:10) {
  glm_poly <- glm(mpg ~ poly(horsepower, i), data = Auto)
  cv_error_10fold[i] <- cv.glm(Auto, glm_poly, K = 10)$delta[1]
}
cv_error_10fold
# You can visualize these errors if needed:
plot(1:10, cv_error_10fold, type = "o", pch = 19, col = "red",
     xlab = "Polynomial Degree", ylab = "10-Fold CV Error (MSE)",
     main = "10-Fold CV Error vs. Polynomial Degree")

# ---------------------------------
# 8. Bootstrap to Estimate Coefficient Variability
# ---------------------------------
# Define a function to compute regression coefficients
boot_fn <- function(data, index) {
  fit <- lm(mpg ~ horsepower, data = data, subset = index)
  return(coef(fit))
}

# Run bootstrap with 1000 resamples
results_boot <- boot(Auto, boot_fn, R = 1000)
print(results_boot)

# Plot histograms for bootstrap distributions of the Intercept and Slope
par(mfrow = c(1, 2))
hist(results_boot$t[, 1], main = "Bootstrap Distribution of Intercept",
     xlab = "Intercept", col = "lightgreen")
hist(results_boot$t[, 2], main = "Bootstrap Distribution of Slope",
     xlab = "Slope", col = "lightblue")
par(mfrow = c(1, 1))  # Reset plotting layout

# ---------------------------------
# 9. (Optional) Define a Custom Cost Function for CV
# ---------------------------------
cost <- function(r, pi = 0) {
  # Example: Mean Absolute Error (customizable)
  mean(abs(r - pi))
}
# Note: In this context, 'r' can be the observed values and 'pi' the predictions.

# -----------------------------------------------------------
# End of Template
# -----------------------------------------------------------
# Load and view dataset
data <- read.csv("path_to_csv_file.csv")
head(data)
summary(data)
attach(data)

# Plot response vs predictor
plot(Predictor, Response, main="Relation", las=2)

# Fit Linear Regression Model
model1 <- lm(Response ~ Predictor)
summary(model1)
lines(smooth.spline(Predictor, predict(model1)), col="yellow", lwd=3)

# Fit Polynomial Regression Model (Degree 2)
model2 <- lm(Response ~ Predictor + I(Predictor^2))
summary(model2)
lines(smooth.spline(Predictor, predict(model2)), col="blue", lwd=3)

# Fit Polynomial Regression Model (Degree 3)
model3 <- lm(Response ~ Predictor + I(Predictor^2) + I(Predictor^3))
summary(model3)
lines(smooth.spline(Predictor, predict(model3)), col="red", lwd=3)

# Legend for comparison
legend("topleft", legend=c("Degree 1", "Degree 2", "Degree 3"),
       col=c("yellow", "blue", "red"), lwd=3, lty=1)

# Higher Degree using poly() function
model4 <- lm(Response ~ poly(Predictor, degree=4, raw=TRUE))
summary(model4)
model5 <- lm(Response ~ poly(Predictor, degree=100, raw=TRUE))
summary(model5)
lines(smooth.spline(Predictor, predict(model5)), col="purple", lwd=3)

# Model comparison using ANOVA
anova(model1, model2)
anova(model2, model5)
# Load required libraries
library(dplyr)
library(splines)
library(ggplot2)
library(mgcv)

# Read data
data <- read.csv("path_to_csv_file.csv")

# Base plot
plot(data$X, data$Y, col="gray", xlab="X", ylab="Y", main="Regression with Splines")

# Define knot
knot <- K_VALUE

# Subset the data
data_before <- subset(data, X < knot)
data_after <- subset(data, X >= knot)

# Fit cubic polynomial before and after knot
model_before <- lm(Y ~ poly(X, 3, raw=TRUE), data = data_before)
model_after <- lm(Y ~ poly(X, 3, raw=TRUE), data = data_after)

# Prediction and plot
x_before <- seq(min(data$X), knot, length.out=100)
x_after <- seq(knot, max(data$X), length.out=100)

pred_before <- predict(model_before, newdata = list(X = x_before))
pred_after <- predict(model_after, newdata = list(X = x_after))

lines(x_before, pred_before, col="red", lwd=2)
lines(x_after, pred_after, col="blue", lwd=2)
abline(v=knot, col="black", lty=2)

# Fit cubic spline with 1 knot
model_spline <- lm(Y ~ bs(X, knots=knot, degree=3), data=data)
x_grid <- seq(min(data$X), max(data$X), length.out=100)
pred_spline <- predict(model_spline, newdata=list(X=x_grid))
lines(x_grid, pred_spline, col="green", lwd=2)

# Fit regression spline with multiple knots
model_multi <- lm(Y ~ bs(X, knots=c(K1, K2, K3)), data=data)
pred <- predict(model_multi, newdata=data.frame(X=x_grid), se=TRUE)
se_bands <- with(pred, cbind(upper = fit + 2 * se.fit, lower = fit - 2 * se.fit))

# Plot with ggplot2
plot_data <- data.frame(
  X = x_grid,
  Fit = pred$fit,
  Lower = se_bands[, "lower"],
  Upper = se_bands[, "upper"]
)

ggplot() +
  geom_point(data = data, aes(x=X, y=Y)) +
  geom_line(data = plot_data, aes(x=X, y=Fit), color="blue") +
  geom_ribbon(data = plot_data, aes(x=X, ymin=Lower, ymax=Upper), alpha=0.3) +
  xlim(range(data$X))

# Fit a GAM model
gam1 <- gam(Y ~ s(X, k=4), data=data)
summary(gam1)

# GAM with multiple smooth terms
gam2 <- gam(Y ~ s(X, k=4) + s(Z), data=data)  # Z = another predictor
summary(gam2)

# Load libraries
library(catdata)
library(rpart)
library(rpart.plot)
library(randomForest)
library(MLmetrics)
library(MASS)

### ========== CLASSIFICATION TREE ==========
data(heart)
Heartdata <- as.data.frame(heart)

# Split data
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(Heartdata), replace=TRUE, prob=c(0.7, 0.3))
train <- Heartdata[sample,]
test <- Heartdata[!sample,]

# Build decision tree
tree.heart <- rpart(y ~ ., data=train, method='class')
rpart.plot(tree.heart)

# Predict
ypred <- predict(tree.heart, test, type='class')
table(Predicted = ypred, Actual = test$y)

# Metrics
ConfusionMatrix(ypred, test$y)
Accuracy(ypred, test$y)
Precision(ypred, test$y)
Recall(ypred, test$y)
F1_Score(ypred, test$y)

# Prune tree
plotcp(tree.heart)
opt_index <- which.min(tree.heart$cptable[, "xerror"])
opt_cp <- tree.heart$cptable[opt_index, "CP"]
pruned_tree <- prune(tree.heart, cp=opt_cp)
rpart.plot(pruned_tree)

# Predict with pruned tree
ypred <- predict(pruned_tree, test, type='class')
ConfusionMatrix(ypred, test$y)

### ========== RANDOM FOREST CLASSIFICATION ==========
rf.heart <- randomForest(factor(y) ~ ., data=train, mtry=6, importance=TRUE)
ypred_rf <- predict(rf.heart, test, type='class')

# Metrics
ConfusionMatrix(ypred_rf, test$y)
Accuracy(ypred_rf, test$y)
Precision(ypred_rf, test$y)
Recall(ypred_rf, test$y)
F1_Score(ypred_rf, test$y)

### ========== REGRESSION TREE ==========
data("Boston")
sample <- sample(c(TRUE, FALSE), nrow(Boston), replace=TRUE, prob=c(0.7, 0.3))
train_boston <- Boston[sample,]
test_boston <- Boston[!sample,]

# Fit regression tree
tree.boston <- rpart(medv ~ ., data=train_boston, method='anova')
rpart.plot(tree.boston)

# Predict
ypred_boston <- predict(tree.boston, test_boston)

# Metrics
MAE(ypred_boston, test_boston$medv)
MSE(ypred_boston, test_boston$medv)
RMSE(ypred_boston, test_boston$medv)
MAPE(ypred_boston, test_boston$medv)

# R-squared
ss_total <- sum((test_boston$medv - mean(test_boston$medv))^2)
ss_res <- sum((test_boston$medv - ypred_boston)^2)
r_squared <- 1 - (ss_res / ss_total)
print(r_squared)

### ========== RANDOM FOREST REGRESSION ==========
rf.reg <- randomForest(medv ~ ., data=train_boston)
ypred_rf_reg <- predict(rf.reg, test_boston)

MAE(ypred_rf_reg, test_boston$medv)
MSE(ypred_rf_reg, test_boston$medv)
RMSE(ypred_rf_reg, test_boston$medv)
MAPE(ypred_rf_reg, test_boston$medv)

# Load Libraries
library(e1071)
library(caret)
library(GGally)
library(ggplot2)
library(clue)         # Hungarian Algorithm
library(mclust)       # Adjusted Rand Index
library(factoextra)   # PCA visualization

# Load and View Dataset
data(iris)
head(iris)
summary(iris)
str(iris)

# ===============================
# PART 1: SVM Classification
# ===============================
# Train SVM model with RBF kernel
svm_model <- svm(Species ~ ., data = iris, kernel = "radial", cost = 1, epsilon = 0.1)
summary(svm_model)

# Predict and Evaluate
pred <- predict(svm_model, iris)
confusionMatrix(pred, iris$Species)

# Misclassification Rate & Accuracy
tab <- table(Predicted = pred, Actual = iris$Species)
misc_class <- 1 - sum(diag(tab)) / sum(tab)
accuracy <- 1 - misc_class
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

# ===============================
# PART 2: SVM Hyperparameter Tuning
# ===============================
set.seed(123)
tune_result <- tune(svm, Species ~ ., data = iris, ranges = list(gamma = c(0.01, 0.1, 1), cost = 2^(2:7)))
summary(tune_result)
best_model <- tune_result$best.model
confusionMatrix(predict(best_model, iris), iris$Species)

# ===============================
# PART 3: Compare SVM Kernels
# ===============================
svm_linear <- svm(Species ~ ., data = iris, kernel = "linear")
svm_poly   <- svm(Species ~ ., data = iris, kernel = "polynomial", degree = 3)
svm_rbf    <- svm(Species ~ ., data = iris, kernel = "radial")

confusionMatrix(predict(svm_linear, iris), iris$Species)
confusionMatrix(predict(svm_poly, iris), iris$Species)
confusionMatrix(predict(svm_rbf, iris), iris$Species)

# ===============================
# PART 4: K-Means Clustering
# ===============================
# Remove label column and scale features
true_labels <- as.numeric(iris$Species)
iris_features <- iris[, -5]
iris_scaled <- as.data.frame(lapply(iris_features, function(x) (x - min(x)) / (max(x) - min(x))))

# Elbow Method (Scree Plot)
wss <- sapply(1:10, function(k) {
  set.seed(123)
  kmeans(iris_scaled, centers = k, nstart = 25)$tot.withinss
})
plot(1:10, wss, type = "b", pch = 19, xlab = "Number of Clusters (K)", ylab = "Total WSS", main = "Scree Plot")
abline(v = 3, col = "red", lty = 2)

# K-Means (K = 3)
set.seed(123)
kmeans_result <- kmeans(iris_scaled, centers = 3, nstart = 25)

# Hungarian Algorithm to align clusters
conf_matrix <- table(kmeans_result$cluster, true_labels)
mapping <- solve_LSAP(conf_matrix, maximum = TRUE)
aligned_clusters <- as.numeric(factor(kmeans_result$cluster, levels = as.integer(mapping)))

# Accuracy & Adjusted Rand Index
final_conf <- table(Predicted = aligned_clusters, Actual = true_labels)
print(final_conf)

accuracy <- sum(aligned_clusters == true_labels) / length(true_labels)
ari <- adjustedRandIndex(aligned_clusters, true_labels)

cat("K-Means Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Adjusted Rand Index (ARI):", round(ari, 3), "\n")

# ===============================
# PART 5: PCA + K-Means
# ===============================
pca_result <- prcomp(iris_scaled, center = TRUE, scale. = TRUE)

# Scree Plot
fviz_eig(pca_result, addlabels = TRUE, main = "PCA Scree Plot")

# Cumulative Variance Plot
cum_var <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
plot(1:length(cum_var), cum_var, type = "b", pch = 19, col = "blue",
     xlab = "Principal Components", ylab = "Cumulative Variance Explained",
     main = "Cumulative Variance Plot", ylim = c(0, 1.1))

