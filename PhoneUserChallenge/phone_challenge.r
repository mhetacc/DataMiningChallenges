setwd("~/Documents/Data Mining/DataMiningChallenges/PhoneUserChallenge")
phone_train <- read.csv("phone_train.csv")
phone_test <- read.csv("phone_validation.csv")


skewnesscheck <- function(){
  library(e1071)

  numeric_cols <- sapply(phone_train, is.numeric)
  skewed_features <- sapply(phone_train[, numeric_cols], function(x) skewness(x, na.rm = TRUE))

  # Identify highly skewed features (e.g., skewness > 1 or < -1)
  highly_skewed <- names(skewed_features[abs(skewed_features) > 1])

  # Print the highly skewed features and their skewness values
  print(highly_skewed)
  print(skewed_features[highly_skewed])
}


plotcalls_over_everything <- function(){

#######  NUMBER OF CALLS  ############
months <- paste0("q0", 1:9)

for (m in months) {
  peak_col    <- paste0(m, ".out.ch.peak")
  offpeak_col <- paste0(m, ".out.ch.offpeak")
  total_col   <- paste0(m, ".out.ch.tot")   # new column name

  phone_train[[total_col]] <- rowSums(phone_train[, c(peak_col, offpeak_col)], na.rm = TRUE)
}

# PLOT NUM OF CALLS OVER MONTHS
months <- paste0("q0", 1:9, ".out.ch.tot")  # your total call columns

# Sum of calls per month
monthly_totals <- colSums(phone_train[, months], na.rm = TRUE)

# Create a dataframe for plotting
df_plot <- data.frame(
  month = months,
  total_calls = monthly_totals
)

# plot
library(ggplot2)

  # Plot with custom labels
ggplot(df_plot, aes(x = month, y = total_calls)) +
  geom_col(fill = "steelblue") +
  labs(title = "Total Calls per Month",
       x = "Month",
       y = "Total Calls") +
  scale_x_discrete(labels = paste("Month", 1:9)) +
  theme_minimal()

# NUM CALLS OVER SEX

# NUM CALLS OVER PLAN

# NUM CALLS OVER PAYMETHOD

########## CALL TIME ############

months <- paste0("q0", 1:9)

for (m in months) {
  peak_col    <- paste0(m, ".out.dur.peak")
  offpeak_col <- paste0(m, ".out.dur.offpeak")
  total_col   <- paste0(m, ".out.dur.tot")   # new column name

  phone_train[[total_col]] <- rowSums(phone_train[, c(peak_col, offpeak_col)], na.rm = TRUE)
}

# PLOT CALL TIME OVER MONTHS
months <- paste0("q0", 1:9, ".out.dur.tot")  # your total call columns

# Sum of calls per month
monthly_totals <- colSums(phone_train[, months], na.rm = TRUE)

# Create a dataframe for plotting
df_plot <- data.frame(
  month = months,
  total_time = monthly_totals
)

# plot
library(ggplot2)

  # Plot with custom labels
ggplot(df_plot, aes(x = month, y = total_time)) +
  geom_col(fill = "steelblue") +
  labs(title = "Call Time per Month",
       x = "Month",
       y = "Call Time") +
  scale_x_discrete(labels = paste("Month", 1:9)) +
  theme_minimal()

# CALL TIME OVER SEX

# make total 
months <- paste0("q0", 1:9, ".out.dur.tot")
phone_train$total_call_time_months <- rowSums(phone_train[, months])

#tot_by_sex <- aggregate(total_call_time_months ~ sex, data = phone_train, sum, na.rm = TRUE)

library(dplyr)
avg_by_sex <- phone_train %>%
  group_by(sex) %>%
  summarize(
    n_customers   = n(),
    total_call_time   = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time = total_call_time / n_customers
  )
avg_by_sex
# remove top 1%

library(dplyr)

# Compute 99th percentile threshold
threshold <- quantile(phone_train$total_call_time_months, 0.99, na.rm = TRUE)

# Filter out top 1% and summarize
avg_by_sex <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(sex) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
  avg_by_sex

# Log compress
library(dplyr)

avg_log_by_sex <- phone_train %>%
  group_by(sex) %>%
  summarize(
    n_customers        = n(),
    total_call_time    = sum(total_call_time_months, na.rm = TRUE),
    avg_log_call_time  = mean(log(total_call_time_months + 1), na.rm = TRUE)
  )
avg_log_by_sex

# CALL TIME OVER AGE

threshold <- quantile(phone_train$total_call_time_months, 0.99, na.rm = TRUE)

# Filter out top 1% and summarize
avg_by_age <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(age) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
avg_by_age

ggplot(avg_by_age, aes(x = age, y = avg_call_time)) +
  geom_col() +
  labs(
    x = "Age",
    y = "Call Time (9-Months Span)",
    title = "Average Call Time by Age"
  ) +
  theme_minimal()

# CALL TIME OVER PLAN

# Compute 99th percentile threshold
threshold <- quantile(phone_train$total_call_time_months, 0.99, na.rm = TRUE)

# Filter out top 1% and summarize
avg_by_plan <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(tariff.plan) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
avg_by_plan
# CALL TIME OVER PAYMETHOD

# Compute 99th percentile threshold
threshold <- quantile(phone_train$total_call_time_months, 0.99, na.rm = TRUE)

# Filter out top 1% and summarize
avg_by_pymethod <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(payment.method) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
avg_by_pymethod
# CALL TIME OVER ZONE

avg_by_zone <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(activation.zone) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
  avg_by_zone

# CALL TIME OVER CHANNEL

avg_by_channel <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(activation.channel) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
avg_by_channel

# CALL TIME OVER VALUEADD 1

avg_by_addV1 <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(vas1) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
  avg_by_addV1

# CALL TIME OVER VALUEADD 2

avg_by_addV2 <- phone_train %>%
  filter(total_call_time_months <= threshold) %>%  # remove top 1%
  group_by(vas2) %>%
  summarize(
    n_customers     = n(),
    total_call_time = sum(total_call_time_months, na.rm = TRUE),
    avg_call_time   = total_call_time / n_customers
  )
  avg_by_addV2


}

linearfit <- function(){
  x <- model.matrix(y ~ ., data=phone_train)[, -1]  
  y <- phone_train$y 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]
  # fit with linear regression
  linear_fit = lm(y ~ ., data=phone_train, subset = train)

  pred <- predict(linear_fit, newdata = phone_train[test, ])
  mean((pred - y.test)^2)

  linear_fit_log = lm(log(y+1) ~ ., data=phone_train, subset = train)

  pred <- exp(predict(linear_fit_log, newdata = phone_train[test, ]))-1
  mean((pred - y.test)^2)


  linear_fit_loglog = lm(y ~ ., data=phone_train, subset = train)
  


  # vector with prediction for each row in test dataframe
  yhat = exp(predict(linear_fit, newdata=phone_test))

  #### Collinearity ####
  phone_train$Combined <- rowMeans(phone_train[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])
  #phone_test$Combined <- rowMeans(phone_test[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])

  linear_fit = lm(y ~ Minor_Axis_Length+Eccentricity+Extent+Combined, data=phone_train, subset = train)

  pred <- predict(linear_fit, newdata = phone_train[test, ])
  mean((pred - y.test)^2)
}

pcomp1 <- function(){
  pc <- prcomp(phone_test, center = TRUE, scale. = TRUE)

  # plot two principal components with colouring the predicted class with yhat
  library(ggplot2)

  pc_df <- as.data.frame(pc$x[, 1:2])
  pc_df$Predictedy <- factor(yhat)

  ggplot(pc_df, aes(PC1, PC2, color = Predictedy)) +
    geom_point() +
    labs(title = "PCA: coloured by classess predicted using lm()")


  # plot two principal components with colouring the predicted class with cheat df
  library(ggplot2)

  pc_df <- as.data.frame(pc$x[, 1:2])
  pc_df$y <- factor(phone_test_with_class$y)

  ggplot(pc_df, aes(PC1, PC2, color = y)) +
  geom_point() +
  labs(title = "PCA: coloured by true classes")
}

## Prediction ##

ridgeregression <- function(){
  ## Ridge
  x <- model.matrix(y ~ ., data=phone_train)[, -1]  
  y <- phone_train$y 

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
  x_test <- model.matrix(~ ., data=phone_test)  
  yhat <- (predict(ridge_fit, newx=x_test, s = bestlam)>1.5)+1
}

lassoregression <- function(){
  ## Lasso
  x <- model.matrix(y ~ ., data=phone_train)[, -1]  
  y <- phone_train$y 

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
  x_test <- model.matrix(~ ., data=phone_test)  
  yhat <- (predict(lasso_fit, newx=x_test, s = bestlam)>1.5)+1
}


pcr <- function(){
  x <- model.matrix(y ~ ., data=phone_train)[, -1]  
  y <- phone_train$y 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  library(pls)
  set.seed(1)

  # fit principal component regression with cross validation
  pcr.fit <- pcr(y ~ ., data = phone_train, subset = train,
    scale = TRUE, validation = "CV")
  validationplot(pcr.fit, val.type = "MSEP")

  # compute mse
  pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 2)
  mean((pcr.pred - y.test)^2)

  pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 7)
  mean((pcr.pred - y.test)^2)
 
  # predict on whole dataset and predict yhat
  pcr.fit <- pcr(y ~ ., data = phone_train, scale = TRUE, ncomp = 6)
  yhat <- (predict(pcr.fit, newdata=phone_test, ncomp = 6)>1.5)+1
}


pls <- function(){
  x <- model.matrix(y ~ ., data=phone_train)[, -1]  
  y <- phone_train$y 

  # split training set to estimate test error
  set.seed(1)
  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  library(pls)
  set.seed(2)

  # fit principal component regression with cross validation
  pls.fit <- plsr(y ~ ., data = phone_train, subset = train, scale = TRUE, validation = "CV")
  validationplot(pls.fit, val.type = "MSEP")
  
  # compute mse
  pls.pred <- predict(pls.fit, x[test, ], ncomp = 2)
  mean((pls.pred - y.test)^2)

  pls.pred <- predict(pls.fit, x[test, ], ncomp = 6)
  mean((pls.pred - y.test)^2)
 
  # predict on whole dataset and predict yhat
  pls.fit <- pcr(y ~ ., data = phone_train, scale = TRUE, ncomp = 7)
  yhat <- (predict(pls.fit, newdata=phone_test, ncomp = 6)>1.5)+1
}

llr <- function(){

  library(locfit)

  fit <- locfit(y ~., data = phone_train)
  # fatal error: eig_dec not converged
}

loess <- function(){
  set.seed(1)
  x <- model.matrix(y ~ ., data=phone_train)[, -1]  
  y <- phone_train$y 

  # split training set to estimate test error

  train <- sample(1:nrow(x), nrow(x) / 2)
  test <- (-train)
  y.test <- y[test]

  # fit different fits to find best span
  loess_fit1 <- loess(y ~ Area + Perimeter + Convex_Area + Major_Axis_Length, data = phone_train, subset = train, span = 0.1)

  loess_fit2 <- loess(y ~ Area + Perimeter + Convex_Area + Major_Axis_Length, data = phone_train, subset = train, span = 0.5)

  loess_fit3 <- loess(y ~ Area + Perimeter + Convex_Area + Major_Axis_Length, data = phone_train, subset = train, span = 0.9)

  # different predictions
  pred1 <- predict(loess_fit1, newdata = phone_train[test, ])
  mean((pred1 - y.test)^2, na.rm = TRUE)

  pred2 <- predict(loess_fit2, newdata = phone_train[test, ])
  mean((pred2 - y.test)^2, na.rm = TRUE)

  pred3 <- predict(loess_fit3, newdata = phone_train[test, ])
  mean((pred3 - y.test)^2, na.rm = TRUE)

  ####### with less predictors ######
  loess_fit1.1 <- loess(y ~ Area + Perimeter, data = phone_train, subset = train, span = 0.1)
  pred1.1 <- predict(loess_fit1.1, newdata = phone_train[test, ])
  mean((pred1.1 - y.test)^2, na.rm = TRUE)

  loess_fit1.2 <- loess(y ~ Area + Perimeter, data = phone_train, subset = train, span = 0.5)
  pred1.2 <- predict(loess_fit1.2, newdata = phone_train[test, ])
  mean((pred1.2 - y.test)^2, na.rm = TRUE)

  loess_fit1.3 <- loess(y ~ Area + Perimeter, data = phone_train, subset = train, span = 0.9)
  pred1.3 <- predict(loess_fit1.3, newdata = phone_train[test, ])
  mean((pred1.3 - y.test)^2, na.rm = TRUE)
  
  #bestspan = NaN
  
  # predict on whole dataset and predict yhat
  #loess_fit <- loess(y ~ ., data = phone_train, span = bestspan)
  yhat <- (predict(loess_fit1.1, newdata=phone_test, span = 0.1)>1.5)+1
}

knn <- function(){
  set.seed(1)

  # leverage multicores R capabilites
  library(doParallel)
  cl <- makePSOCKcluster(detectCores() - 1)       # use all available cores except one
  registerDoParallel(cl)

  
  # load library and set validation method
  library(caret)
  train.control <- trainControl(method  = "LOOCV")

  ### FOR CLASSIFICATION ###
  phone_train$y <- as.factor(phone_train$y) # necessary, caret will automatically do classification 

  knn_fit <- train(
    y~ .,
    method     = "knn",
    tuneGrid   = expand.grid(k = 1:20),
    trControl  = train.control,
    preProcess = c("center","scale"),    # normalized
    metric     = "Accuracy",
    data       = phone_train,
    allowParallel = TRUE
    )
  
  knn_predict <- predict(knn_fit, newdata = x[test, ])
  #mean((knn_predict - y.test)^2)
  confusionMatrix(knn_predict, y.test)

  # predict actual yhat
  yhat <- predict(knn_fit, newdata=phone_test)

  ### FOR REGRESSION ###
  knn_fit <- train(
    y~ .,
    method     = "knn",
    tuneGrid   = expand.grid(k = 1:20),
    trControl  = train.control,
    preProcess = c("center","scale"),    # normalized
    metric     = "RMSE",
    data       = phone_train,
    allowParallel = TRUE
    )

  knn_fit

  # predict actual yhat
  yhat <- (predict(knn_fit, newdata=phone_test)>1.5)+1



  stopCluster(cl)
}


random_forest_parallel_ranger<-function(){
  # ranger is a faster implementation for the random forest
  
  set.seed(1)
  
  # collinearity
  phone_train$Combined <- rowMeans(phone_train[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])
  
  
  # leverage multicores R capabilites
  library(doParallel)
  cl <- makePSOCKcluster(detectCores() - 1)       # use all available cores except one
  registerDoParallel(cl)

  # load library and set validation method
  library(caret)
  train.control <- trainControl(method  = "LOOCV")

  ### FOR CLASSIFICATION ###
  phone_train$y <- as.factor(phone_train$y) # necessary, caret will automatically do classification 

  rf_fit <- train(
    y~ .,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "Accuracy",
    data       = phone_train,
    allowParallel = TRUE
    )
  
  rf_predict <- predict(rf_fit, newdata = x[test, ])
  #mean((rf_predict - y.test)^2)
  confusionMatrix(rf_predict, y.test)

  # predict actual yhat
  yhat <- predict(rf_fit, newdata=phone_test)

  ### FOR REGRESSION ###
  rf_fit <- train(
    y~ .,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "RMSE",
    data       = phone_train,
    allowParallel = TRUE
    )

  rf_fit

  # predict actual yhat
  yhat <- (predict(rf_fit, newdata=phone_test)>1.5)+1


  stopCluster(cl)
}


ranger_parallel_combined_features<-function(){
  set.seed(1)
  
  # collinearity
  phone_train$Combined <- rowMeans(phone_train[, c("Area","Perimeter","Major_Axis_Length","Convex_Area")])
  
  
  # leverage multicores R capabilites
  library(doParallel)
  cl <- makePSOCKcluster(detectCores() - 1)       # use all available cores except one
registerDoParallel(cl)

  # load library and set validation method
  library(caret)
  train.control <- trainControl(method  = "LOOCV")

  ### FOR CLASSIFICATION ###
  phone_train$y <- as.factor(phone_train$y) # necessary, caret will automatically do classification 

  rf_fit <- train(
    y~ Minor_Axis_Length + Eccentricity + Extent + Combined,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "Accuracy",
    data       = phone_train,
    allowParallel = TRUE
    )
  
  rf_predict <- predict(rf_fit, newdata = x[test, ])
  #mean((rf_predict - y.test)^2)
  confusionMatrix(rf_predict, y.test)

  # predict actual yhat
  yhat <- predict(rf_fit, newdata=phone_test)

  ### FOR REGRESSION ###
  rf_fit <- train(
    y~ Minor_Axis_Length + Eccentricity + Extent + Combined,
    method     = "ranger",
    tuneLength = 10,
    trControl  = train.control,
    metric     = "RMSE",
    data       = phone_train,
    allowParallel = TRUE
    )

  rf_fit

  # predict actual yhat
  yhat <- (predict(rf_fit, newdata=phone_test)>1.5)+1


  stopCluster(cl)
}



