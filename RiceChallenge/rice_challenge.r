setwd("~/Documents/Data Mining/DataMiningChallenges/RiceChallenge")
rice_train <- read.csv("rice_train.csv")
rice_test <- read.csv("rice_test.csv")

linearfit <- function(){
  x <- model.matrix(Class ~ ., data=rice_train)[, -1]  
  y <- rice_train$Class 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]
  # fit with linear regression
  linear_fit = lm(Class ~ ., data=rice_train, subset = train)

  pred <- predict(linear_fit, newdata = rice_train[test, ])
  mean((pred - y.test)^2)

  # vector with prediction for each row in test dataframe
  yhat = (predict(fit, newdata=rice_test)>1.5)+1

  #### Collinearity ####
  rice_train$Combined <- rowMeans(rice_train[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])
  #rice_test$Combined <- rowMeans(rice_test[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])

  linear_fit = lm(Class ~ Minor_Axis_Length+Eccentricity+Extent+Combined, data=rice_train, subset = train)

  pred <- predict(linear_fit, newdata = rice_train[test, ])
  mean((pred - y.test)^2)
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


  # fit ridge model with best lambda and plot it
  ridge_fit <- glmnet(x, y, alpha = 0)

  plot(ridge_fit, xvar = "lambda")
  legend(
    "bottomleft",
    legend = colnames(x),
    col = 1:ncol(x),
    lty = 1,
    cex = 0.8
  )

  # lastly predict yhat on test set 
  x_test <- model.matrix(~ ., data=rice_test)[, -1]  
  yhat <- (predict(ridge_fit, newx=x_test, s = bestlam)>1.5)+1
}

lassoregression <- function(){
  ## Lasso
  x <- model.matrix(Class ~ ., data=rice_train)[, -1]  
  y <- rice_train$Class 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  # built-in cross-validation function for lasso lambda selection
  set.seed(1)
  cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
  plot(cv.out)
  bestlam <- cv.out$lambda.min
  cv_mse <- min(cv.out$cvm)
  cv_rmse <- sqrt(cv_mse)
  cv_rmse

  ## same as of cv.out
  # lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
  # mean((lasso.pred - y.test)^2)


  # fit ridge model with best lambda
  lasso_fit <- glmnet(x, y, alpha = 1)
  
  plot(lasso_fit, xvar = "lambda")
  legend(
    "bottomleft",
    legend = colnames(x),
    col = 1:ncol(x),
    lty = 1,
    cex = 0.8
  )

  # lastly predict yhat on test set 
  x_test <- model.matrix(~ ., data=rice_test)[, -1]  
  yhat <- (predict(lasso_fit, newx=x_test, s = bestlam)>1.5)+1
}


pcr <- function(){
  x <- model.matrix(Class ~ ., data=rice_train)[, -1]  
  y <- rice_train$Class 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  library(pls)
  set.seed(1)

  # fit principal component regression with cross validation
  pcr.fit <- pcr(Class ~ ., data = rice_train, subset = train,
    scale = TRUE, validation = "CV")
  validationplot(pcr.fit, val.type = "MSEP")

  # compute mse
  pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 2)
  mean((pcr.pred - y.test)^2)

  pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 7)
  mean((pcr.pred - y.test)^2)
 
  # predict on whole dataset and predict yhat
  pcr.fit <- pcr(Class ~ ., data = rice_train, scale = TRUE, ncomp = 6)
  yhat <- (predict(pcr.fit, newdata=rice_test, ncomp = 6)>1.5)+1
}


pls <- function(){
  x <- model.matrix(Class ~ ., data=rice_train)[, -1]  
  y <- rice_train$Class 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  library(pls)
  set.seed(2)

  # fit principal component regression with cross validation
  pls.fit <- plsr(Class ~ ., data = rice_train, subset = train, scale = TRUE, validation = "CV")
  validationplot(pls.fit, val.type = "MSEP")
  
  # compute mse
  pls.pred <- predict(pls.fit, x[test, ], ncomp = 2)
  mean((pls.pred - y.test)^2)

  pls.pred <- predict(pls.fit, x[test, ], ncomp = 6)
  mean((pls.pred - y.test)^2)
 
  # predict on whole dataset and predict yhat
  pls.fit <- pcr(Class ~ ., data = rice_train, scale = TRUE, ncomp = 7)
  yhat <- (predict(pls.fit, newdata=rice_test, ncomp = 6)>1.5)+1
}

llr <- function(){

  library(locfit)

  fit <- locfit(Class ~., data = rice_train)
  # fatal error: eig_dec not converged
}

loess <- function(){
  set.seed(1)
  x <- model.matrix(Class ~ ., data=rice_train)[, -1]  
  y <- rice_train$Class 

  # split training set to estimate test error

  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  # fit different fits to find best span
  loess_fit1 <- loess(Class ~ Area + Perimeter + Convex_Area + Major_Axis_Length, data = rice_train, subset = train, span = 0.1)

  loess_fit2 <- loess(Class ~ Area + Perimeter + Convex_Area + Major_Axis_Length, data = rice_train, subset = train, span = 0.5)

  loess_fit3 <- loess(Class ~ Area + Perimeter + Convex_Area + Major_Axis_Length, data = rice_train, subset = train, span = 0.9)

  # different predictions
  pred1 <- predict(loess_fit1, newdata = rice_train[test, ])
  mean((pred1 - y.test)^2, na.rm = TRUE)

  pred2 <- predict(loess_fit2, newdata = rice_train[test, ])
  mean((pred2 - y.test)^2, na.rm = TRUE)

  pred3 <- predict(loess_fit3, newdata = rice_train[test, ])
  mean((pred3 - y.test)^2, na.rm = TRUE)

  ####### with less predictors ######
  loess_fit1.1 <- loess(Class ~ Area + Perimeter, data = rice_train, subset = train, span = 0.1)
  pred1.1 <- predict(loess_fit1.1, newdata = rice_train[test, ])
  mean((pred1.1 - y.test)^2, na.rm = TRUE)

  loess_fit1.2 <- loess(Class ~ Area + Perimeter, data = rice_train, subset = train, span = 0.5)
  pred1.2 <- predict(loess_fit1.2, newdata = rice_train[test, ])
  mean((pred1.2 - y.test)^2, na.rm = TRUE)

  loess_fit1.3 <- loess(Class ~ Area + Perimeter, data = rice_train, subset = train, span = 0.9)
  pred1.3 <- predict(loess_fit1.3, newdata = rice_train[test, ])
  mean((pred1.3 - y.test)^2, na.rm = TRUE)
  
  #bestspan = NaN
  
  # predict on whole dataset and predict yhat
  #loess_fit <- loess(Class ~ ., data = rice_train, span = bestspan)
  yhat <- (predict(loess_fit1.1, newdata=rice_test, span = 0.1)>1.5)+1
}

knn <- function(){
  set.seed(1)
  
  # load library and set validation method
  library(caret)
  train.control <- trainControl(method  = "LOOCV")

  ### FOR CLASSIFICATION ###
  rice_train$Class <- as.factor(rice_train$Class) # necessary, caret will automatically do classification 

  knn_fit <- train(
    Class~ .,
    method     = "knn",
    tuneGrid   = expand.grid(k = 1:20),
    trControl  = train.control,
    preProcess = c("center","scale"),    # normalized
    metric     = "Accuracy",
    data       = rice_train)
  
  knn_predict <- predict(knn_fit, newdata = x[test, ])
  #mean((knn_predict - y.test)^2)
  confusionMatrix(knn_predict, y.test)

  # predict actual yhat
  yhat <- predict(knn_fit, newdata=rice_test)

  ### FOR REGRESSION ###
  knn_fit <- train(
    Class~ .,
    method     = "knn",
    tuneGrid   = expand.grid(k = 1:20),
    trControl  = train.control,
    preProcess = c("center","scale"),    # normalized
    metric     = "RMSE",
    data       = rice_train)

  knn_fit

  # predict actual yhat
  yhat <- (predict(knn_fit, newdata=rice_test)>1.5)+1
}


random_forest_parallel_ranger<-function(){
  # ranger is a faster implementation for the random forest
  
  set.seed(1)
  
  # collinearity
  rice_train$Combined <- rowMeans(rice_train[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])
  
  
  # leverage multicores R capabilites
  library(doParallel)
  cl <- makePSOCKcluster(detectCores() - 1)       # use all available cores except one
registerDoParallel(cl)

  # load library and set validation method
  library(caret)
  train.control <- trainControl(method  = "LOOCV")

  ### FOR CLASSIFICATION ###
  rice_train$Class <- as.factor(rice_train$Class) # necessary, caret will automatically do classification 

  rf_fit <- train(
    Class~ .,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "Accuracy",
    data       = rice_train,
    allowParallel = TRUE
    )
  
  rf_predict <- predict(rf_fit, newdata = x[test, ])
  #mean((rf_predict - y.test)^2)
  confusionMatrix(rf_predict, y.test)

  # predict actual yhat
  yhat <- predict(rf_fit, newdata=rice_test)

  ### FOR REGRESSION ###
  rf_fit <- train(
    Class~ .,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "RMSE",
    data       = rice_train,
    allowParallel = TRUE
    )

  rf_fit

  # predict actual yhat
  yhat <- (predict(rf_fit, newdata=rice_test)>1.5)+1


  stopCluster(cl)
}


ranger_parallel_combined_features<-function(){
  set.seed(1)
  
  # collinearity
  rice_train$Combined <- rowMeans(rice_train[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])
  
  
  # leverage multicores R capabilites
  library(doParallel)
  cl <- makePSOCKcluster(detectCores() - 1)       # use all available cores except one
registerDoParallel(cl)

  # load library and set validation method
  library(caret)
  train.control <- trainControl(method  = "LOOCV")

  ### FOR CLASSIFICATION ###
  rice_train$Class <- as.factor(rice_train$Class) # necessary, caret will automatically do classification 

  rf_fit <- train(
    Class~ Minor_Axis_Length + Eccentricity + Extent + Combined,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "Accuracy",
    data       = rice_train,
    allowParallel = TRUE
    )
  
  rf_predict <- predict(rf_fit, newdata = x[test, ])
  #mean((rf_predict - y.test)^2)
  confusionMatrix(rf_predict, y.test)

  # predict actual yhat
  yhat <- predict(rf_fit, newdata=rice_test)

  ### FOR REGRESSION ###
  rf_fit <- train(
    Class~ Minor_Axis_Length + Eccentricity + Extent + Combined,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "RMSE",
    data       = rice_train,
    allowParallel = TRUE
    )

  rf_fit

  # predict actual yhat
  yhat <- (predict(rf_fit, newdata=rice_test)>1.5)+1


  stopCluster(cl)
}



