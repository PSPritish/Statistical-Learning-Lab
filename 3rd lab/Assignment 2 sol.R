View(diabetes)
df <- diabetes
View(df)
head(df)
dim(df)
str(df)
plot(df)
table(df$Pregnancies)
boxplot(Glucose ~ Outcome , data = df)
library(ggplot2)
ggplot(diabetes, aes(x = DiabetesPedigreeFunction, y = Outcome)) +
  geom_jitter(alpha = 0.5, color = "blue") +
  labs(title = "Scatter Plot: DiabetesPedigreeFunction vs Outcome", x = "DiabetesPedigreeFunction", y = "Outcome")

ggplot(diabetes, aes(x = factor(Outcome), y = Insulin, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Box Plot: Insulin vs Outcome", x = "Outcome", y = "Insulin") +
  theme_minimal()

num_data <- df[,sapply(df,is.numeric)] #select numeric columns
cor_matrix <- cor(num_data , use = "complete.obs")
print(cor_matrix)


ggplot(diabetes, aes(x = Glucose, y = BMI, shape = factor(Outcome))) +
  geom_point(size = 3, alpha = 0.7,) +  # Points with different shapes based on Outcome
  labs(
    title = "Scatter Plot: Glucose vs BMI by Outcome",
    x = "Glucose",
    y = "BMI",
    shape = "Outcome"
  ) +
  theme_minimal() +  # Minimal theme for a clean look
  scale_shape_manual(values = c(16, 18)) 


# Load required library
library(ggplot2)

# Define the input columns (exclude the Outcome column)
input_columns <- c("Glucose", "BMI", "BloodPressure", "SkinThickness", "Insulin", 
                   "Pregnancies", "DiabetesPedigreeFunction", "Age")
trn <- sample(dim(df)[1], 615) # 80% of 768
trn
df_train <- df[trn,]
df_test <- df[-trn,]
head(df_test)
df_test <- subset(df_test,select = -`Outcome`)
head(df_test)
dim(df_train)
dim(df_test)
#fit logistic regression model

M1  = glm(Outcome ~. , family = binomial , data = df_train)
summary(glm.fits)
#predict using test dataset
pred <- predict(glm.fits , df_test , type = "response")
pred_class <- ifelse(pred>=0.5 , 1 , 0)

#Create confusion matrix
conf_matrix <- table(df[-trn,]$Outcome , pred_class)

#Test Accuracy
mean(pred_class == df[-trn,]$Outcome)

tp <- conf_matrix[2, 2]  # True Positives
fp <- conf_matrix[1, 2]  # False Positives
fn <- conf_matrix[2, 1]  # False Negatives

# Precision
precision <- tp / (tp + fp)

# Recall
recall <- tp / (tp + fn)
# F1-Score
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1_score, "\n")

M2 = glm(Outcome ~ Pregnancies + Glucose + BMI  , family = binomial , data = df_train)
summary(M2)
chisquare = deviance(M1) - deviance(M2)
dof = 8 - 3
qchisq(0.95,dof)(cols = -Outcome , names_to = "variable", values_to = "Value")

library(caret)
library(dplyr)
library(tidyr)
boxplot_data <- diabetes%>% pivot_longer(cols = -Outcome , names_to = "Variable", values_to = "Value")
ggplot(boxplot_data, aes(x= as.factor(Outcome), y = Value , fill=as.factor(Outcome))) +geom_boxplot() + facet_wrap(~Variable , scales = "free") + theme_minimal() + labs(x = "Outcome" , y= "Value" , fill = "Outcome")
