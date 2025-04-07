View(manufacturing)
df <- manufacturing
head(df)
dim(df)
plot(df)
cor_matrix <- cor(df)
cor_matrix
library(ggcorrplot)
ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower", lab = TRUE)
library(ggplot2)
library(reshape2)
cor_data <- melt(cor_matrix)

# Create the heatmap
ggplot(cor_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +  # Heatmap tiles
  scale_fill_gradient2() +
  labs(title = "Correlation Heatmap", fill = "Correlation") +
  theme_minimal()

model <- lm(`Quality Rating` ~ ., data = df)
summary(model)
aov(model)
resi <-  residuals(model)
resi
qqnorm(resi)
train <- sample(nrow(df),3165) #80% of 3956
train_df <- df[train,]
test_df <- df[-train,]
fit2 <- lm(`Quality Rating` ~. , data = train_df)

dummy <- subset(test_df,select = -`Quality Rating`)
pred <- predict(fit2, newdata = dummy)
rmse <- sqrt(mean(test_df$`Quality Rating` - pred)^2)  
rmse
