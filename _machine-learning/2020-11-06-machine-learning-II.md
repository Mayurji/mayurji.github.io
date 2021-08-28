---
layout: machine-learning
title: Machine Learning Series - Part II
description: "Concepts In Machine Learning"
date:   2020-11-06 15:43:52 +0530
---
{% include mathjax.html %}

**Machine Learning Concepts**

When designing features or algorithms for learning features, our goal is to separate the **factors of variation** that explain the observed data.

**Features** are the components in our dataset which helps in building ML Algorithms. There are different types of features like Categorical, Numerical, Ordinal, etc. So before applying an algorithm to a dataset, we need to convert the dataset into a format, which is consumable by an algorithm. For instance, we can handle categorical value, by converting it into a one-hot encoding, similar to mapping category to a numeric value.

<center>
<img src="{{site.url}}/assets/images/ml/ml_things.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/><br>
<p><b>Figure 1:</b> Terms in Machine Learning</p> 
</center>

**Feature Engineering**

Any problem in Machine Learning or Deep Learning requires some amount of feature engineering, one cannot simply do a ``` model.fit(x, y)``` and get SOTA results. Feature Engineering requires creativity and understanding of the domain. For instance e, the cost of a house of size 'a' in X locality is priced higher than the same size house in Y locality, even though locality is not a feature in your raw dataset.

**One Hot Encoding**

Only a few models like decision trees can handle the categorical features like color with values as red, blue, etc without performing preprocessing to numerical form. However, the majority of the algorithms require feature values to be numerical. To convert these features, we can use a one-hot encoding. 

```python

      orange = [0, 0, 1]
         red = [1, 0, 0]    
        blue = [0, 1, 0]
```
One hot encoding increases the dimensionality of the feature vector, but transforming features like colors - red as 1, blue as 2, etc would bring in 'order' to the values of the colors feature and would mess the algorithm's decision making while modeling.

**Handling Missing Values**

The raw data comes in different problems with it, sometimes the values of the features are missed if the dataset is prepared manually. To overcome such missing value problem, one does the following,

* Drop the samples, if the dataset is big enough.
* Perform Data Imputation to filling the gaps of missing values.
* Few algorithms are robust to missing values.

**Data Imputation**

* One way to impute is to find the mean of the features and replace the missing values. (Careful if outliers are present).
* Replace with a value that is outside the range of features i.e. if feature x is [0, 1] then replace the missing value with -1 or 2. It provides a distinct feature value for this sample alone.
* Replace with a value that is in the middle of the range i.e. if feature x is [-1, 1] then replace the missing value with 0. It makes the algorithm to get less affected by 0.
   
**Scaling** 
    
**Why Scaling?**
   
   Scaling a feature is an important task before building/applying an ML algorithm. For instance, If an algorithm is not using feature scaling, then it can consider the value 300 meters to be greater than 5 km, which is not true and in this case, the algorithm will give wrong predictions by giving more importance to 300 meters than 3 km. So, we use Feature Scaling to normalize its values.

**Types of Scaling**
   
* Min-Max Scaling or Normalization
* Mean Normalization
* Standardization or Z-score Normalization

*Min-Max Scaling* helps in rescaling the feature values into the range of [0,1] or [-1,1].

<p>$$x' = {x - min(x)  \over max(x) - min(x)}$$</p>

*Mean Normalization*

<p>$$x' = {x - average(x) \over max(x) - min(x)}$$</p>

*Standardization* of features makes the value of each feature in the data have zero mean and unit variance. It is a widely used normalization technique as major algorithms like SVM, Neural Nets and Logistic Regressions follow such standardization.

<p>$$x' = {x - \mu \over \sigma}$$</p>


<p align='center'>. . . . .</p>

<center>
<img src="{{site.url}}/assets/images/ml/bias-variance.jpg"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p><b>Figure 2:</b> Model Complexity vs Error</p> 
</center>

**Bias** in the statistical term refers to the tendency of a measurement process to over-or under-estimate the value of a population parameter. In survey sampling, for example, the bias would be the tendency of a sample statistic to systematically over-or under-estimate a population parameter. In machine learning, a model is said to be suffering from bias when the model does not perform well on the training set i.e. the model is unable to recognize the data pattern of the training set, making it too simplistic. As the model complexity is increased, the bias is reduced.

**Underfitting** occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data. Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough. Specifically, underfitting occurs if the model or an algorithm shows low variance but high bias.

**Variance**, in the context of Machine Learning, is a type of error that occurs due to a model's sensitivity to small fluctuations in the training set. The high variance would cause an algorithm to model the noise in the training set. This is most commonly referred to as overfitting.

As seen in figure 2, as the model complexity is increased, the variance increases causing the model to overfit. To avoid such a scenario, we keep track of validation loss, which is high if the model is overfitting. In Deep Learning, we perform *Early Stopping* to avoid overfitting.

**Overfitting** refers to a model that models the training data too well. Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.


<p align='center'>. . . . .</p>

<center>
<img src="{{site.url}}/assets/images/ml/gradient-descent.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p><b>Figure 3:</b> Batch vs Stochastic vs Mini-Batch</p> 
</center>

**Batch Gradient Descent** refers to how gradient changes(weight updates) are applied to the weight matrix. If we have 1000 data points, then the model is trained on 1000 data points before any update is made to the weight of the model.

**Stochastic Gradient Descent** refers to how gradient changes(weight updates) are applied to the weight matrix. If we have 1000 data points, then the model is trained on 1 data point and an update is made to the weight of the model.

**Mini-Batch Gradient Descent** refers to how gradient changes(weight updates) are applied to the weight matrix. If we have 1000 data points, then we assign a value to batch_size, if batch_size is 10, then the model is trained on 10 data points and an update is made to the weight of the model. This happens iteratively taking 10 data points and updating.


<p align='center'>. . . . .</p>

**Regularization**

We have a simple dataset with two features and a target variable, we can either use a simple model with 2 coefficients for two variables, or else we use a complex model which will have more coefficients and will overfit the simple dataset. The complex model will not generalize for the new data as it is a overfit model. To overcome and choose a simple model, we can use regularization.

**L1 Regularization**

While calculating the error(E) we add the absolute value of the coefficients of the model. In simple model case, we have 2 coefficients $$w_1, w_2$$ then,

<p>$$TotalError = E + |w_1| + |w_2|$$</p>

In Complex model case, lets say have 5 coefficients $$w_1, w_2, w_3, w_4, w_5$$ then,

<p>$$TotalError = E + |w_1| + |w_2| + |w_3| + |w_4| + |w_5|$$</p>
    
So we get a smaller error for the simple model and will use the same for the generalization.

**L2 Regularization**

While calculating the error(E), we square the value of the coefficients of the model. In simple model case, we have 2 coefficients $$w_1, w_2$$ then,

<p>$$TotalError = E + w_1^2 + w_2^2$$</p>

In Complex model case, lets say have 5 coefficients $$w_1, w_2, w_3, w_4, w_5$$ then,

<p>$$TotalError = E + w_1^2 + w_2^2 + w_3^2 + w_4^2 + w_5^2$$</p>

So we get a smaller error for the simple model and will use the same for the generalization. More the number of parameters, the more complex the model is.

**How to select Regularization's parameter $$\lambda$$?**

Based on the complexity of the data, the model tends to be complex. So the lambda value acts as a switch either to increase the complexity or not. If we keep a small value of lambda and multiple it, with the complexity part of the model i.e. "w" parameters then we get a smaller error compared to the simple model with its "w" parameters. And if lambda is large, then we punish the complexity part highly and thus making the complex model with great error.

<p>$$TotalError = E + \lambda (w_1^2 + w_2^2 + w_3^2 + w_4^2 + w_5^2)$$ </p>

The complexity of the model is defined by the number of parameters(w) in the equation.

<p align='center'>. . . . .</p>
