Machine Learning
========================================================

The ideia of this project is to predict 20 new observations based on the training data provided. Thisis part of the Coursera project of Machine Learning.

The data sets were downloaded with this code:

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    "pml-training.csv")
```

```
## Error: esquema de URL não suportado
```

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    "pml-testing.csv")
```

```
## Error: esquema de URL não suportado
```


After that it's important to import the files and load some useful libraries


```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 2.15.3
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.0.2 r169 Copyright (c) 2006-2013 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 2.15.3
```

```
## Loading required package: rpart
```

```r
library(AppliedPredictiveModeling)
```

```
## Warning: package 'AppliedPredictiveModeling' was built under R version
## 2.15.3
```

```r
library(pgmm)
library(caret)
```

```
## Warning: package 'caret' was built under R version 2.15.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 2.15.3
```

```r

study = read.csv("pml-training.csv")
final.test = read.csv("pml-testing.csv")
```


I've opted for dividing the training data into two. As my analisys can take a while, the actual training data set is only 30% of the total data.


```r
set.seed(2525)
train.sample = sample(1:dim(study)[1], size = dim(study)[1] * 0.3, replace = F)
train = study[train.sample, ]
test = study[-train.sample, ]
```



A new data set is created, in this data set columns that had missing values were removed

```r
x = cbind(train$roll_belt, train$pitch_belt, train$yaw_belt, train$total_accel_belt, 
    train$gyros_belt_x, train$gyros_belt_y, train$gyros_belt_z, train$accel_belt_x, 
    train$accel_belt_y, train$accel_belt_z, train$magnet_belt_x, train$magnet_belt_y, 
    train$magnet_belt_z, train$roll_arm, train$pitch_arm, train$yaw_arm, train$total_accel_arm, 
    train$gyros_arm_x, train$gyros_arm_y, train$gyros_arm_z, train$accel_arm_x, 
    train$accel_arm_y, train$accel_arm_z, train$magnet_arm_x, train$magnet_arm_y, 
    train$magnet_arm_z, train$roll_dumbbell, train$pitch_dumbbell, train$yaw_dumbbell, 
    train$total_accel_dumbbell, train$gyros_dumbbell_x, train$gyros_dumbbell_y, 
    train$gyros_dumbbell_z, train$accel_dumbbell_x, train$accel_dumbbell_y, 
    train$accel_dumbbell_z, train$magnet_dumbbell_x, train$magnet_dumbbell_y, 
    train$magnet_dumbbell_z, train$roll_forearm, train$pitch_forearm, train$yaw_forearm, 
    train$total_accel_forearm, train$gyros_forearm_x, train$gyros_forearm_y, 
    train$gyros_forearm_z, train$accel_forearm_x, train$accel_forearm_y, train$accel_forearm_z, 
    train$magnet_forearm_x, train$magnet_forearm_y, train$magnet_forearm_z)
```



We still have many columns, so in a try to reduce the number of columns without losing important information for the model a Principal Components analisys is performed:


```r
c1 = preProcess(x, method = "pca", thresh = 0.5)
c2 = preProcess(x, method = "pca", thresh = 0.6)
c3 = preProcess(x, method = "pca", thresh = 0.7)
c4 = preProcess(x, method = "pca", thresh = 0.75)
c5 = preProcess(x, method = "pca", thresh = 0.8)
c6 = preProcess(x, method = "pca", thresh = 0.9)
c7 = preProcess(x, method = "pca", thresh = 0.95)

# number of components by % of variance explained
rbind(c(0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95), c(c1$numComp, c2$numComp, c3$numComp, 
    c4$numComp, c5$numComp, c6$numComp, c7$numComp))
```

```
##      [,1] [,2] [,3]  [,4] [,5] [,6]  [,7]
## [1,]  0.5  0.6  0.7  0.75  0.8  0.9  0.95
## [2,]  5.0  7.0  9.0 11.00 13.0 20.0 26.00
```


The new data set c6. that explains 0.9 the variance was the chosen one. The transformation is applied to the training data set as well as some adjusts.


```r
comp = predict(c6, x)
x1 = cbind(comp, train$classe)
x2 = as.data.frame(x1)
d = dim(x1)[2]
vs = rep("V", d - 1)
ids = (1:(d - 1))
labels = cbind(paste(vs, ids, sep = ""))
labels = c(labels, "classe")
names(x2) = labels[1:d]
x2$classe = as.factor(x2$classe)
```


It's time to run the model. Having categorical answer and made me choose the Random Forest method, as it is very good at predictions with this number of observations and is doesn't have a computational cost as big as other good technics could have.


```r
set.seed(2525)
modFit = train(classe ~ ., data = x2, method = "rf")
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 2.15.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'e1071' was built under R version 2.15.3
```


A Backtest is executed to it's possibel to see that the model has a good perforace


```r
est.train = predict(modFit, newdata = x2)
t = table(x2$classe, est.train)
g.result = sum(diag(t))/sum(t)
result = diag(t)/c(sum(t[, 1]), sum(t[, 2]), sum(t[, 3]), sum(t[, 4]), sum(t[, 
    5]))
result
```

```
## 1 2 3 4 5 
## 1 1 1 1 1
```

```r
g.result
```

```
## [1] 1
```

```r
t
```

```
##    est.train
##        1    2    3    4    5
##   1 1716    0    0    0    0
##   2    0 1149    0    0    0
##   3    0    0  971    0    0
##   4    0    0    0  975    0
##   5    0    0    0    0 1075
```


Iy's time to apply the model in a test data set to verify if this model is really good or we have any problem, like a overfit.

The same transformations have to be made to the test data set


```r
y = cbind(test$roll_belt, test$pitch_belt, test$yaw_belt, test$total_accel_belt, 
    test$gyros_belt_x, test$gyros_belt_y, test$gyros_belt_z, test$accel_belt_x, 
    test$accel_belt_y, test$accel_belt_z, test$magnet_belt_x, test$magnet_belt_y, 
    test$magnet_belt_z, test$roll_arm, test$pitch_arm, test$yaw_arm, test$total_accel_arm, 
    test$gyros_arm_x, test$gyros_arm_y, test$gyros_arm_z, test$accel_arm_x, 
    test$accel_arm_y, test$accel_arm_z, test$magnet_arm_x, test$magnet_arm_y, 
    test$magnet_arm_z, test$roll_dumbbell, test$pitch_dumbbell, test$yaw_dumbbell, 
    test$total_accel_dumbbell, test$gyros_dumbbell_x, test$gyros_dumbbell_y, 
    test$gyros_dumbbell_z, test$accel_dumbbell_x, test$accel_dumbbell_y, test$accel_dumbbell_z, 
    test$magnet_dumbbell_x, test$magnet_dumbbell_y, test$magnet_dumbbell_z, 
    test$roll_forearm, test$pitch_forearm, test$yaw_forearm, test$total_accel_forearm, 
    test$gyros_forearm_x, test$gyros_forearm_y, test$gyros_forearm_z, test$accel_forearm_x, 
    test$accel_forearm_y, test$accel_forearm_z, test$magnet_forearm_x, test$magnet_forearm_y, 
    test$magnet_forearm_z)
comp = predict(c6, y)

y1 = cbind(comp, test$classe)
y2 = as.data.frame(y1)
d = dim(y1)[2]
vs = rep("V", d - 1)
ids = (1:(d - 1))
labels = cbind(paste(vs, ids, sep = ""))
labels = c(labels, "classe")
names(y2) = labels[1:d]
y2$classe = as.factor(y2$classe)

set.seed(2525)
est.test = predict(modFit, newdata = y2)
```


we can avalute the model performace with the code below


```r
t = table(y2$classe, est.test)
g.result = sum(diag(t))/sum(t)
result = diag(t)/c(sum(t[, 1]), sum(t[, 2]), sum(t[, 3]), sum(t[, 4]), sum(t[, 
    5]))
result
```

```
##      1      2      3      4      5 
## 0.9640 0.9188 0.8900 0.9439 0.9823
```

```r
g.result
```

```
## [1] 0.9415
```

```r
t
```

```
##    est.test
##        1    2    3    4    5
##   1 3746   43   27   34   14
##   2   81 2466   92    1    8
##   3   33   99 2258   48   13
##   4   19   18  125 2071    8
##   5    7   58   35   40 2392
```


The result seems acceptable, so the result will be aplied to the final test data for submission

```r
z = cbind(final.test$roll_belt, final.test$pitch_belt, final.test$yaw_belt, 
    final.test$total_accel_belt, final.test$gyros_belt_x, final.test$gyros_belt_y, 
    final.test$gyros_belt_z, final.test$accel_belt_x, final.test$accel_belt_y, 
    final.test$accel_belt_z, final.test$magnet_belt_x, final.test$magnet_belt_y, 
    final.test$magnet_belt_z, final.test$roll_arm, final.test$pitch_arm, final.test$yaw_arm, 
    final.test$total_accel_arm, final.test$gyros_arm_x, final.test$gyros_arm_y, 
    final.test$gyros_arm_z, final.test$accel_arm_x, final.test$accel_arm_y, 
    final.test$accel_arm_z, final.test$magnet_arm_x, final.test$magnet_arm_y, 
    final.test$magnet_arm_z, final.test$roll_dumbbell, final.test$pitch_dumbbell, 
    final.test$yaw_dumbbell, final.test$total_accel_dumbbell, final.test$gyros_dumbbell_x, 
    final.test$gyros_dumbbell_y, final.test$gyros_dumbbell_z, final.test$accel_dumbbell_x, 
    final.test$accel_dumbbell_y, final.test$accel_dumbbell_z, final.test$magnet_dumbbell_x, 
    final.test$magnet_dumbbell_y, final.test$magnet_dumbbell_z, final.test$roll_forearm, 
    final.test$pitch_forearm, final.test$yaw_forearm, final.test$total_accel_forearm, 
    final.test$gyros_forearm_x, final.test$gyros_forearm_y, final.test$gyros_forearm_z, 
    final.test$accel_forearm_x, final.test$accel_forearm_y, final.test$accel_forearm_z, 
    final.test$magnet_forearm_x, final.test$magnet_forearm_y, final.test$magnet_forearm_z)

comp = predict(c6, z)

z1 = comp
z2 = as.data.frame(z1)
d = dim(z1)[2]
vs = rep("V", d)
ids = (1:(d))
labels = cbind(paste(vs, ids, sep = ""))

names(z2) = labels[1:d]

set.seed(2525)
est.final.test = predict(modFit, newdata = z2)
est.final.test
```

```
##  [1] 3 1 3 1 1 2 4 2 1 1 2 3 2 1 5 5 1 2 2 2
## Levels: 1 2 3 4 5
```

```r

answers = NULL
for (i in 1:length(est.final.test)) {
    if (est.final.test[i] == 1) {
        answers = c(answers, "A")
    }
    if (est.final.test[i] == 2) {
        answers = c(answers, "B")
    }
    if (est.final.test[i] == 3) {
        answers = c(answers, "C")
    }
    if (est.final.test[i] == 4) {
        answers = c(answers, "D")
    }
    if (est.final.test[i] == 5) {
        answers = c(answers, "E")
    }
    
}
answers
```

```
##  [1] "C" "A" "C" "A" "A" "B" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```


The last thing is to create the files for submission with the function given with the problem:

```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```



  
