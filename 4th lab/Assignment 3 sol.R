View(diabetes)
df <- diabetes
head(df)
str(df)
sum(is.na(df))
dim(df)
plot(df)
df_index = sample(nrow(df),615)
df_train = df[df_index,]
df_test = df[-df_index,]
library(ISLR)
library(ggplot2)
library(MASS)
attach(df)
lda_fit1 <- lda(Outcome ~ ., data = df_train)
lda_fit1
pred <- predict(lda_fit1 , newdata = df_test )
pred.df <- data.frame(pred)
head(pred.df)
table(Actual = df_test$Outcome,Predicted = pred.df$class)

plot(lda_fit1) 
library(ggplot2)

# Predict the LDA values
lda_values <- predict(lda_fit1)

# Create a dataframe with predictions
lda_df <- data.frame(LD1 = lda_values$x[,1], Outcome = df_train$Outcome)

# Plot the LDA separation
ggplot(lda_df, aes(x = LD1, fill = as.factor(Outcome))) +
  geom_density(alpha = 0.5) +
  labs(title = "LDA Separation",x = "Linear Discriminant (LD1)" , fill = "Outcome") +
  theme_minimal()

