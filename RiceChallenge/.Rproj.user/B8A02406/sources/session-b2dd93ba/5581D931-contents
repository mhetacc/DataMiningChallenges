# Tutorial script
train <- read.csv("rice_train.csv")
test <- read.csv("rice_test.csv")

# fit with linear regression
fit = lm(Class ~ ., data=train)

# vector with prediction for each row in test dataframe
yhat = (predict(fit, newdata=test)>1.5)+1

