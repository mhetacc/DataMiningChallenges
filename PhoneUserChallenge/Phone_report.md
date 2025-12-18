# Preliminary Observations

## Dataset

### Monthly Call Time Skewness

Call time data is usually positively skewed, with many users having low usage and a small number of user with very high usage (e.g.,50.000 seconds).
This is easily verifiable at a glance if we try to plot the monthly call time or, better yet, with an histogram.

![](phone_y_plot.png)
![](phone_y_hist.png)

A common method to compress such data is by applying a logarithmic function to it.

![](phone_log_y_hist.png)


# Prediction

## Linear Regression

### Heteroskedasticity

One preliminary way to check whetere there can be some form of heteroskedasticity is by computing a residual plot.

```{r}
fit <- lm(y ~ ., data = train)

plot(fitted(fit), resid(fit),
     xlab = "Fitted values",
     ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, lty = 2)
```

As we can see there is a evident funnel shape, often sign of heteroskedasticity. To prove it more formally we can check p-values.

```{r}
library(lmtest)
bptest(fit)
```

## KNN

Let's start with a baseline prediction using a simple KNN model for regression. I first tried to fit all the data, without logarithmically transform the target, but I had to change the validation method from Leave-One-Out-Cross-Validation to just Cross-Validation for lack of memory.

```{rbash}
k-Nearest Neighbors 

10000 samples
   98 predictor

Pre-processing: centered (100), scaled (100) 
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 8999, 9000, 9001, 9000, 8999, 8999, ... 
Resampling results across tuning parameters:

  k   RMSE      Rsquared   MAE     
   1  4510.530  0.4594312  1258.249
      ........  ........   ........
   8  4045.793  0.5674593  1123.891
      ........  ........   ........
  20  4140.494  0.5695263  1132.276

RMSE was used to select the optimal model using the smallest value.
The final value used for the model was k = 8.
```

It is quite evident that $MSE = 16368440.9988$ is a terrible result. I then tried to train the model on target $log(y+1)$ and got much better results: with $k=15$, $RMSE=2.367258$ and $MSE=5.60391043856$. Thus i predicted *yhat* as follows.

```{r}
yhat <- exp(predict(knn_fit, newdata=phone_test)-1)
```