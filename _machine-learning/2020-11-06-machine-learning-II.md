---
layout: machine-learning
title: Machine Learning - II
description: "Concepts In Machine Learning"
date:   2020-11-06 15:43:52 +0530
---
{% include mathjax.html %}

**Machine Learning Concepts**

When designing features or algorithms for learning features, our goal is to separate the **factors of variation** that explain the observed data.

**Features** are components in your dataset which helps in building ML Algorithm. There are different types features like Categorical, Numerical, Ordinal etc. So before applying algorithm on a dataset, we need to convert the dataset into a format, which is consumable by algorithm. One such vital technique is handling categorical value, by converting it into an one-hot encoding, similar to mapping category to numeric value.

<center>
<img src="{{site.url}}/assets/images/ml/ml_things.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="1000" height="600"/><br>
<p><b>Figure 2:</b> Terms in Machine Learning</p> 
</center>

**Feature Engineering**

Any problem in Machine Learning or Deep Learning requires some amount of feature engineering, one cannot simply do a ```model.fit(x, y)``` and get SOTA results. Feature Engineering requires creativity and understanding of the domain. For instance, cost of house of size 'a' in X locality is priced higher than same size house in Y locality, even though locality is not a feature in your raw dataset.

**One Hot Encoding**

There are few algorithms like decision trees, which takes categorical feature like color with values as 'red', 'blue' etc as inputs but majority of the algorithms requires feature values to be numerical. To convert these features, we can use one hot encoding. 

```python
      orange = [0, 0, 1]
         red = [1, 0, 0]    
        blue = [0, 1, 0]
```
One hot encoding increases the dimensionality of the feature vector, but transforming features like colors - red as 1, blue as 2 etc would bring in 'order' to the  values of the colors feature and would mess the algorithm's decision making while modelling.

**Handling Missing Values**

The raw data comes in different problems with it, sometimes the features values are missed if the dataset is prepared manually. To overcome such missing value problem, one do the following,

* Drop the samples, if the dataset is big enough.
* Perform Data Imputation to fillin the gaps of missing values.
* Few algorithms are robust to missing values.

**Data Imputation**

* One way to impute is to find the mean of the features and replace the missing values.(Careful if outliers are present).
* Replace with value which is outside the range of features i.e. if feature x is [0, 1] then replace missing value with -1 or 2. It provides a distinct feature values for this sample alone.
* Replace with value which is in middle of the range i.e. if feature x is [-1, 1] then replace missing value with 0. It makes algorithm to get less affected by 0.
   
**Scaling** of a feature is an important task before building/applying a ML algorithm.
    
**Why Scaling?**
   
   For instance, If an algorithm is not using feature scaling, then it can consider the value 300 meter to be greater than 5 km, which is not true and in this case, the algorithm will give wrong predictions by giving more importance to 300 meter than 3 km. So, we use Feature Scaling to normalize the values of it.

**Types of Scaling**
   
* Min-Max Scaling or Normalization
* Mean Normalization
* Standardization or Z-score Normalization

**Min-Max Scaling** helps in rescaling the feature values into the range of [0,1] or [-1,1].

<p>$$x' = {x - min(x)  \over max(x) - min(x)}$$</p>

**Mean Normalization** 

<p>$$x' = {x - average(x) \over max(x) - min(x)}$$</p>

**Standardization** of features makes the value of each feature in the data to have zero mean and unit variance. It is a widely used normalization technique as major algorithms like SVM, Neural Nets and Logistic Regressions follow such standardization.

<p>$$x' = {x - \mu \over \sigma}$$</p>

**Bias** refers to the tendency of a measurement process to over- or under-estimate the value of a population parameter. In survey sampling, for example, bias would be the tendency of a sample statistic to systematically over- or under-estimate a population parameter.

**Underfitting** occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data. Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough. Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.

**Variance**, in the context of Machine Learning, is a type of error that occurs due to a model's sensitivity to small fluctuations in the training set. High variance would cause an algorithm to model the noise in the training set. This is most commonly referred to as overfitting.

**Overfitting** refers to a model that models the training data too well. Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.

**Batch Gradient Descent** refers how a gradient changes are applied to the weight matrix. If we have 1000 data points, then the model is trained on 1000 data points before any update is made to the weight of the model.

**Stochastic Gradient Descent** refers how a gradient changes are applied to the weight matrix. If we have 1000 data points, then the model is trained on 1 data point and an update is made to the weight of the model.

**Mini-Batch Gradient Descent** refers how a gradient changes are applied to the weight matrix. If we have 1000 data points, then we assign a value to batch_size, if batch_size is 10, then the model is trained on 10 data point and an update is made to the weight of the model. This happens iteratively taking 10 data points and updating.

**Regularization**

We have a simple dataset with two features and a target variable, we can either use simple model with 2 coefficients for two variables or else we use a complex model which will have more coefficients and will overfit the simple dataset. The complex model will not generalize for the new data as it is a overfit model. To overcome and choose a simple model, we can use regularization.

**L1 Regularization**

While calculating the error(E) we add the absolute value of the coefficients of the model. 

In simple model case, we have 2 coefficients $$w_1, w_2$$ then,

<p>$$TotalError = E + |w_1| + |w_2|$$</p>

In Complex model case, lets say have 5 coefficients $$w_1, w_2, w_3, w_4, w_5$$ then,

<p>$$TotalError = E + |w_1| + |w_2| + |w_3| + |w_4| + |w_5|$$</p>
    
So we get smaller error for simple model and will use the same for the generalization.

**L2 Regularization**

While calculating the error(E), we square the value of the coefficients of the model. 

In simple model case, we have 2 coefficients $$w_1, w_2$$ then,

<p>$$TotalError = E + (w_1)^2 + (w_2)^2$$</p>

In Complex model case, lets say have 5 coefficients $$w_1, w_2, w_3, w_4, w_5$$ then,

<p>$$TotalError = E + (w_1)^2 + (w_2)^2 + (w_3)^2 + (w_4)^2 + (w_5)^2$$</p>

So we get smaller error for simple model and will use the same for the generalization. More the number of parameters, more complex the model is.

**How to select Regularization's parameter?**
Based on the complexity of the data, the model tends to be complex. So the lambda value acts like a switch either to increase the complexity or not. If we keep a small value of lambda and multiple it, with the complexity part of the model i.e. "w" parameters then we get smaller error compared to the simple model with its "w" parameters. And if lambda is large, then we punish the complexity part highly and thus making the complex model with great error.

The complexity of the model is defined by the number of the parameters(w) in the equation.

<p align='center'>. . . . .</p>
