# Preliminary Observations

## Scatterplot

![](../RiceChallenge/ricetest_scatterplot_1080.png)

## Principal Components

```{r}
pc <- prcomp(test)
```

```{bash}
Importance of components:
                             PC1      PC2      PC3   PC4    PC5     PC6      PC7
Standard deviation     2489.7639 58.37044 11.91918 2.004 0.4882 0.07269 0.003174
Proportion of Variance    0.9994  0.00055  0.00002 0.000 0.0000 0.00000 0.000000
Cumulative Proportion     0.9994  0.99998  1.00000 1.000 1.0000 1.00000 1.000000
```
