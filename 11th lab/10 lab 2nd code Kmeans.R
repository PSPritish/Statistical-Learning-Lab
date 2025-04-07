library(ggplot2)
library(clue) # For Hungarian Algorithm (cluster-label matching)
library(mclust) # For adjusted Rand Index 
library(factoextra) # Fro enhanced Visualization


data(iris)
head(iris)

# Remove Species column (Store for later evaluation)
true_labels <- as.numeric(iris$Species) # Convert to numeric (1,2,3)
iris_features <- iris[,-5] # Features only

# Min-Max Scaling
min_max_scale <- function(x){
  (x-min(x)) / (max(x)-min(x))
}

# Apply min-max scaling to all features
iris_scaled <- as.data.frame(lapply(iris_features, min_max_scale))

head(iris_scaled)

# Find Optimal K (Scree Plot)
# Compute total within-cluster sum of squares (WSS) for K=1 to 100

wss <- sapply(1:10, function(k){
  set.seed(123)
  kmeans(iris_scaled,k,nstart = 25)$tot.withinss
})

# Plot scree plot (elbow method)
plot(1:10, wss, type="b",pch=1,
     xlab="Number of Clusters (K)",
     ylab="Total Within-Cluster",
     main="Scree Plot for Optimal K")
abline(v = 3 , lty = 2 , col = "red")

# K means Clustering ( K = 3)
set.seed(123)
k <- 3
kmeans_results <- kmeans(iris_scaled,centers = k , nstart = 25)


# Evaluate Clustering vs True Labels

# Align clusters to true labels (Hungarian algorithm)
confusion <- table(kmeans_results$cluster, true_labels)
confusion # Here cost matrix is confusion matrix

mapping <- solve_LSAP(confusion, maximum = TRUE)

aligned_clusters <- as.numeric(factor(kmeans_results$cluster, levels= as.integer(mapping)))

# Confustion matrix
confusion_matrix <- table(Predicted = aligned_clusters , Actual = true_labels)
print(confusion_matrix)

# accuracy calculation
accuracy <- sum(aligned_clusters == true_labels)/length(true_labels)
cat("\nAccuracy:", round(accuracy*100,2),"%n")

# adjusted Rand Index (ARI)
ari <- adjustedRandIndex(aligned_clusters, true_labels)
cat("Adjusted Rand Index (ARI):", round(ari,3),"\n")


# PCA WITH K-MEANS CLUSTERING
pca_result <- prcomp(iris_scaled)

var_explained <- pca_result$sdev^2 /sum(pca_result$sdev^2)
var_explained

fviz_eig(pca_result,addlabels = TRUE, main = "PCA Scree Plot (Variance per Component)")
# Get cumulative variance manually
cum_var <- cumsum(pca_result$sdev^2/sum(pca_result$sdev^2))         
cum_var


plot(1:length(cum_var),cum_var,
     type="b",
     pch= 19,
     col="blue",
     xlab="Principal Components",
     ylab="Cumulative Variance",
     main = "Cumulative Variance Explained",
     ylim=c(0,1.1),
     xaxt = "n")


