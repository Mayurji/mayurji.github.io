---
layout: machine-learning
title: Supervised Learning
description: Classification, Regression, Loss Function, Parameter Update.
date:   2021-02-14 17:43:52 +0530
---
{% include mathjax.html %}

# Supervised Learning

<center>
<img src="{{site.url}}/assets/images/ml/yannis-h-uaPaEM7MiQQ-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="100%" height=auto/><br>
<p><span>Photo by <a href="https://unsplash.com/@yanphotobook?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Yannis H</a> on <a href="https://unsplash.com/s/photos/teaching?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span></p>
</center>

In this blog post, we'll discuss about supervised learning and the class of problems that comes under its umbrella. Before getting into supervised learning, I would highly recommend going through machine learning jargon like what is dataset, target, predictor, model etc. from the [previous blog posts](https://mayurji.github.io/machine-learning/).


### 📌Supervised Observation

In Machine Learning,  if a label or target is available for an observation then such an observation is called **Supervised Observation**. From technical standpoint, given a set of data points X's associated to set of labels or outcomes Y's, we try to build a model that learns to predict *y* from *x.*
$$
\{x^{(1)},...,\ x^{(m)}\} \ and \ \{y^{(1)}, ..., \ y^{(m)}\}
$$

### 📌Type of Prediction

Consider observing animals, when we see an animal before we label it as Dog or Cat, we with our super speed consciousness, we check for features like number of legs, eyes, head, body, whiskers etc and then label it as animal A, B, C etc.

<center>
<img src="{{site.url}}/assets/images/ml/classification.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Supervised Learning - Classification Based Problem</p> 
</center>

Classifying animals or identifying an object or identifying whether a mail is Spam or Not, all these problems are called as **Classification** based Machine Learning problems. Thus, here we predict the predictors/observation as an **one of N finite categories**. Examples- Logistic Regression, SVM etc.

<center>
<img src="{{site.url}}/assets/images/ml/regression(1).png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Supervised Learning - Regression Based Problem</p> 
</center>

Now, consider predicting the price of a stock, we know that stock prices are volatile in nature, meaning it changes from a minute by minute. And the drop or rise in prices are not categorical like +5/+10 or -10/-5, the value changes in decimals as a continuous values.

Problems like stock price prediction, house price prediction etc all come under a class of problem called as **Regression**. Thus, here we predict the predictors/observation as **one of infinite continuous value.** Examples - Linear Regression, Decision Tree based Regression etc.

### 📌Type of Model

<center>
<img src="{{site.url}}/assets/images/ml/Discriminative_Generative.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Supervised Learning - Discriminative and Generative Model</p> 
</center>

A model can learn to classify animals or predict the price of the stock in two different ways. 

First, Discriminative Model, where the model learns to build an decision boundary based on which it classifies predictors as target A or B. Discriminative modeling tries to directly predict y given x. Examples-Linear Regression.
<br>
$$
Discriminative\ Model: P(y\ | \ x)
$$
<br>
Second, Generative Model, where the model learns from the probability distribution of predictors *x* based on given y and then deduce the target *y* given *x*. Example-Naive Bayes.
<br>
$$
Generative \ Model: Estimate \ P(x \ | \ y) \ and \ deduce \ P(y \ | \ x)
$$
<br>
Thus, we can perform classification or regression based on either discriminative approach or generative approach. 

### 📌Loss Function 

Loss function is one of the key component of supervised learning. Each machine learning algorithm has a loss function which helps in learning the pattern of the predictors and then estimating the target *y*. Loss Function takes the estimated y_hat as input for a real data point and compared with real target y and finds the difference between them.

Consider a linear regression problem, the loss function plays the role to generate the best fit line between the predictors such that when a new unseen data point is put into the regression line, it at best tries to identify the best estimate of y for unseen data point. For linear regression, the widely used loss function is Mean Squared Error. It's called error because the estimate is deviated from true value.

<center>
<img src="{{site.url}}/assets/images/ml/mse.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Loss Function - Mean Squared Error</p> 
</center>

*The **least squares method** is a statistical **procedure** to find the best fit for a set of data points by minimizing the sum of  the offsets or residuals of points from the plotted curve. **Least squares** regression is **used** to predict the behavior of dependent variables.*

**Common Loss Function**

* Mean Squared Error
* Mean Absolute Error
* Logistic Loss
* Cross-Entropy Loss
* Hinge Loss

### 📌Parameter Update θ

When a model tries to learn or identify the pattern in the data, it keeps some reference points while learning. These reference points updates every time the new data is seen by the model. Technically these reference points are called as **Coefficients** of the learned equation. 

In Linear Regression, the model learn these parameters while training the model and updates it such that the loss is reduced, and estimated y is brought close to true y. Parameter update is a key factor in developing a generalized model. Learning rate is a hyperparameter that is tuned for finding the proper update.

<center>
<img src="{{site.url}}/assets/images/ml/gd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Parameter Update - Gradient Descent</p> 
</center>

An algorithm that is widely used for parameter update is **Gradient Descent**. Gradient Descent uses partial derivatives to update the parameter θ. Similarly there are other ways to find parameters updates like Likelihood and Newton's Algorithm.

### 📌Evaluation

In supervised learning, the evaluation is quite simple in both the classification and regression problem. Since the dataset is labelled, we can create a separate test set and validation set from our dataset and use the trained model to predict how accurately it classifies on test set by metric using accuracy, precision, recall, f1-score etc. 

In regression problem, we can keep the margin of error that is allowed by the model and measure it using R square, Adjusted R square, MSE etc. I've covered this topic in [Machine Learning Concept](https://mayurji.github.io/machine-learning/2020-11-06-machine-learning-III).

❌**Warning** ❌ Never use the training set to evaluate the model !

I hope covered almost everything required for performing supervised learning. Next we'll deep dive into unsupervised learning.

Thanks for reading
# 🐅

**Reference**

CS229 Machine Learning - Stanford 



