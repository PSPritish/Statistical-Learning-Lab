data("airquality")
View(airquality)
dim(airquality)
head(airquality)
tail(airquality)
colnames(airquality)
mean(airquality$Ozone)
colSums(is.na(airquality))
str(airquality)
summary(airquality)
summary(airquality$Ozone)

#Handling missing values
mean(airquality$Ozone)
is.na(airquality$Ozone)
mean(airquality$Ozone, na.rm = TRUE )
#airquality$Ozone[is.na(airquality$Ozone)] <- mean(airquality$Ozone, na.rm = TRUE )
install.packages("mice")
library(mice)
md.pattern(airquality)
install.packages("VIM")
methods(mice)
library(VIM)
aggr_plot <- aggr(airquality, col=c('navyblue','red'),numbers = TRUE , sortVars=TRUE , labels = names(data),cex.axix = .7,gap=3,ylab = c("Histogram of missing data","Pattern"))
impute_object <- mice(airquality , method = "pmm" , m = 1)
impute_data <- complete(impute_object, action="long")
head(impute_data)
summary(impute_object)

#Descriptive statistics
duplicated(airquality)
mean(airquality$Temp)










# Visualise the data 
install.packages("ggplot2")
library(ggplot2)
plot(airquality)
plot(Ozone~Temp,airquality)
plot(airquality$Temp,xlab = "observation number" , ylab = "Temperature value" , pch = 15 , col = "Red",main = "Scatter plot")
boxplot(airquality)
boxplot(airquality$Ozone)
boxplot(Ozone ~ Month , airquality , xlab = "Month" , ylab = "Ozone (ppb)")
palette()


#Join the two dataset
df1 = data.frame(StudentId = c(101:106) , Product = c("Hindi","English","Maths","Science", "Political Science", "Physics"))
df1
df2 = data.frame(StudentId = c(102,104,106,107,108),state = c("Mangalore","Mysore","Pune","Dehradun","Delhi"))
df2
dfi = merge(x = df1 , y = df2 , by = "StudentId" ,all = FALSE) #if all is true then all will be combined
dfi
dfm = merge(x = df1 , y = df2 , by = "StudentId" ,all = TRUE)
dfm
dfl = merge(x = df1 , y = df2 , by = "StudentId" , all.x = TRUE)
dfl
dfr = merge(x = df1 , y = df2 , by = "StudentId" , all.y = TRUE)
dfr
