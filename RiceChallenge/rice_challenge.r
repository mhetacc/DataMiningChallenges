# Tutorial script
rice_train <- read.csv("rice_train.csv")
rice_test <- read.csv("rice_test.csv")

linearfit <- function(){
  # fit with linear regression
  fit = lm(Class ~ ., data=rice_train)

  # vector with prediction for each row in test dataframe
  yhat = (predict(fit, newdata=rice_test)>1.5)+1
}

pcomp1 <- function(){
  pc <- prcomp(rice_test, center = TRUE, scale. = TRUE)

  # plot two principal components with colouring the predicted class with yhat
  library(ggplot2)

  pc_df <- as.data.frame(pc$x[, 1:2])
  pc_df$PredictedClass <- factor(yhat)

  ggplot(pc_df, aes(PC1, PC2, color = PredictedClass)) +
    geom_point() +
    labs(title = "PCA: coloured by classess predicted using lm()")


  # plot two principal components with colouring the predicted class with cheat df
  library(ggplot2)

  pc_df <- as.data.frame(pc$x[, 1:2])
  pc_df$Class <- factor(rice_test_with_class$Class)

  ggplot(pc_df, aes(PC1, PC2, color = Class)) +
  geom_point() +
  labs(title = "PCA: coloured by true classes")
}

## Prediction ##

ridgeregression <- function(){
  ## Ridge
  x <- model.matrix(Class ~ ., data=rice_train)[, -1]  
  y <- rice_train$Class 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  # built-in cross-validation function for ridge lambda selection
  set.seed(1) # for reproducibility 
  library(glmnet)
  cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
  plot(cv.out)
  bestlam <- cv.out$lambda.min
  bestlam
  cv_mse <- min(cv.out$cvm)
  cv_rmse <- sqrt(cv_mse)
  cv_rmse


  # fit ridge model with best lambda
  out <- glmnet(x, y, alpha = 0)
  #predict(out, type = "coefficients", s = bestlam)[1:20, ]

  # lastly predict yhat on test set 
  x_test <- model.matrix(~ ., data=rice_test)[, -1]  
  yhat <- (predict(out, newx=x_test, s = bestlam)>1.5)+1
}


