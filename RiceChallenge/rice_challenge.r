# Tutorial script
train <- read.csv("rice_train.csv")
test <- read.csv("rice_test.csv")

# fit with linear regression
fit = lm(Class ~ ., data=train)

# vector with prediction for each row in test dataframe
yhat = (predict(fit, newdata=test)>1.5)+1

# principal components test dataset
pc <- prcomp(test, center = TRUE, scale. = TRUE)

# plot two principal components with colouring the predicted class with yhat
library(ggplot2)

pc_df <- as.data.frame(pc$x[, 1:2])
pc_df$PredictedClass <- factor(yhat)

ggplot(pc_df, aes(PC1, PC2, color = PredictedClass)) +
  geom_point() +
  labs(title = "PCA: coloured by classess predicted using lm()")
