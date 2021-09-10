---
layout: machine-learning
title: Loss Function
description: Loss function is the brain of a learning system.
date:   2021-02-20 17:43:52 +0530
---
{% include mathjax.html %}

In this blog post, we'll discuss the Loss function, parameter θ, and different types of loss functions. I've learned a lot while researching this topic and hope you'll feel the same. Without further a due, let's starts off with loss function. 

In simple terms, the objective of the loss function is to find the difference between or deviation between the actual ground truth of a value and an estimated approximation of the same value.

<p>
$$
Loss \ Function \ = y_{actual} \ - y_{estimate}
$$
</p>

The above equation is the simplest form loss function.

> In more technical and Wikipedia terms - In mathematical optimization and decision theory, a loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost"  associated with the event. An optimization problem seeks to minimize a loss function.

## Why we are learning about Loss Function

The loss function is a key component of any learning mechanism in AI, either be it machine learning or deep learning, or reinforcement learning. The loss function acts as feedback to the system we are building, without feedback the system will never know **where** and **what** should be improved.  

<p>
$$
Trained \ model's function \ M \ = \ \theta_1 x_1 \ + \theta_2 x_2 \ + \theta_3 x_3 \ +\ .....\ +  \theta_{n-1} x_{n-1} \ + \theta_n x_n 
$$
</p>

θ is the parameter of the trained model M. Loss function helps the model in answering the **what and where** question. Answer to the **what** 🔍 question is "θ", which should be improved to reduce the difference between the actual vs estimate. And **where** 🔍 question means which θ, either its 

<p>
$$
\theta_1 \ or \ \theta_{n-1} \ or \ any\ other \ \theta.
$$
</p>

With repeated iteration over the model with a different set of samples of the dataset, we identify the answer to **what and where** questions.

## Role of θ

Consider the iris data with features as sepal width, sepal length, petal width, petal length, and target as variants of iris **setosa, versicolor, and virginica.** It is a multi-class problem with classes of more than two. 

We'll experiment with a simple Logistic Regression fit on iris data with a different number of iteration to converge the estimate with actual.

```python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        import numpy as np

        iris = load_iris #Load Dataset
        x_train, x_test, y_train, y_test = train_test_split(iris()['data'], iris()['target'], test_size=0.2) #train and test split
        LR = LogisticRegression(max_iter=1) # Change max_iter from 1 to 100, to see the effects on learning coefficient θ.
        LR.fit(x_train, y_train)

```

Check out the coefficient of Logistic Regression at max_iter=1, the **LR.coef_** represents three rows, each row represents one class with its feature's **coefficient value θ**, and **intercept_** refers to the bias of the Logistic regression for each class. With the help of θ and intercept, the model creates its decision function.

```python

        """
        LR.coef_
        array([[-0.06626003,  0.00156274, -0.11994002, -0.04727278],
            [ 0.03090311,  0.00097671,  0.03840425,  0.0103922 ],
            [ 0.03535691, -0.00253945,  0.08153577,  0.03688058]])

        LR.intercept_
        array([ 4.83768518e-18, -1.00170102e-03,  1.00170102e-03])

        LR.decision_function(np.array([x_train[3]]))
        array([[-3.49185211,  1.63301032,  1.85884179]])
        """

```

To arrive at a decision function for sample no. 3 i.e. x_train[3], do the following calculation with coefficient and intercept.

```python

        y_estimate = np.sum(np.multiply(LR.coef_, np.array([x_train[3]])), axis=1) + LR.intercept_
        """
        y_estimate
        array([-3.49185211,  1.63301032,  1.85884179])
        """

```

Each value in y_estimate is the confidence score of the sample for each class. When we iterate the Logistic Regression for max_iter=100, the y_estimate values start to converge and the parameter values are updated as follows

```python

        """
        LR.coef_
        array([[-0.45753939,  0.80624315, -2.38307659, -0.9789476 ],
            [ 0.32666965, -0.29473046, -0.16163767, -0.73956488],
            [ 0.13086974, -0.5115127 ,  2.54471426,  1.71851248]])

        LR.intercept_
        array([ 10.00706074,   2.72893805, -12.73599879])

        y_estimate
        array([  7.64747156,   3.0268779 , -10.67434946])
        """

```

For sample x_train[3], the y_train[3] is 0, thus the confidence score of class 0 is increased from **-3.49 to 7.64** when max_iter=100. With each iteration, the loss is reduced with the help of parameters **θ**.

**Difference Between Loss Function and Cost Function**: The **loss function** computes the error for a single training example, while the **cost function** is the average of the **loss functions** of the entire training set.

## Common Loss Function

* Squared Loss (Mean Square Error)
* Absolute Loss (Mean Absolute Error)
* Hinge Loss
* Log Loss or Cross Entropy Loss

### Mean Squared Error or L2 Loss

MSE of an estimator measures the average of the square of the errors. It is the averaged squared difference between the estimated values and the actual values. 

<p>
$$
MSE \ = \ {1 \over m} \sum_{i=1}^{m} \ (y_i - y_{i}')^2 \\

where, m \ is \ the \ number \ of \ samples \\
y_i \ is \ the \ actual \ value \\
y_{i}' is \ the \ estimated \ value.
$$
</p>

MSE values are mostly positive and not zero, because of the uncertainty of the estimator and also the loss of information during estimation which accounts for actual ground truth.

<center>
<img src="{{site.url}}/assets/images/ml/mse.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Loss Function - Mean Squared Error</p> 
</center>

#### Mean Squared Error Code 

```python

        def MSE(yHat, y):
            return np.sum((yHat - y)**2) / y.size

```

In regression problems, MSE is used to measure the distance between the data point and the predicted regression line. It helps in determine to which extent the model is fit to the data. When the errors are large, it becomes obvious to use MSE. Because the square of a large number means bigger the error distance thus more the penalizing for bigger errors. Thus, making MSE sensitive to outliers.

Also note, that increasing the sample size m leads to a decrease in MSE, because a larger sample size, reduces the variance of the distribution, making it easy to reduce the distance between the estimator and the actual.

### Mean Absolute Error or L1 Loss

MAE is the measure of error between a pair of variables such as predicted vs actual. MAE is the average absolute difference between X and Y. MAE is widely used for forecast error in time series analysis.

<center>
<img src="{{site.url}}/assets/images/ml/mae.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Loss Function - Mean Absolute Error</p> 
</center>

<p>
$$
MAE \ = \ {1 \over m} \sum_{i=1}^{m} \ |y_i - y_{i}'| \\

where, m \ is \ the \ number \ of \ samples \\
y_i \ is \ the \ actual \ value \\
y_{i}' is \ the \ estimated \ value.
$$
</p>

#### Mean Absolute Loss Code

```python

        def L1(yHat, y):
            return np.sum(np.absolute(yHat - y)) / y.size

```

MAE is less sensitive to outliers. [To reduce MAE, minimize the median and to reduce MSE, minimize the mean.](https://forecasters.org/wp-content/uploads/gravity_forms/7-621289a708af3e7af65a7cd487aee6eb/2015/07/Kolassa_Stephan_ISF2015.pdf)

### Hinge Loss

Hinge Loss is used for the Maximum Margin Classifier. The sound of the maximum margin classifier takes us to SVM (support vector machine), where the distance between the data point and decision boundary is kept at max. **The loss function's penalization depends on how badly the data point is misclassified, meaning how far the data point is present on the wrong side of the decision boundary.**

<center>
<img src="{{site.url}}/assets/images/ml/maximum_margin.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Loss Function - Maximum Margin Classifier</p> 
</center>

<p>
$$
Loss \ = \ max(0, \ 1 - \ y \ . \ \hat{y}) \\
where, \ y \ is \ the \ actual \ value \\
\hat{y} \ is \ the \ predicted \ value.
$$
</p>

#### Hinge Loss Code

```python

        def Hinge(yHat, y):
            return np.max(0, y - (1-2*y)*yHat)

```

where *y_hat* is the output of the SVM and *y* is the true class (-1 or 1). Note that the loss is nonzero for misclassified points, as well as correctly classified points that fall within the margin. Hinge Loss is a loss function used for classification problems. [Check out this awesome resource on how to minimize hinge loss](https://math.stackexchange.com/questions/782586/how-do-you-minimize-hinge-loss). Hinge Loss has massive documentation because of its so many variants.

### Log Loss or Cross Entropy Loss

Logistic Loss is also known as Log Loss. It is used in calculating the loss in Logistic Regression. When the number of the class is 2 the cross-entropy is calculated as

<p>
$$
Log \ Loss \ =  - \ {1 \over N}\ \sum_{(x,y) \ \epsilon \ D}\ y \ log( \ p(y') \ ) \ + \ (1 \ - \ y) \ log(1 \ - \ p(y') \ ) \\
where, \ y \ is \ the \ ground \ truth \ label \\
y' \ is \ predicted \ label \ with \ value \ between \ (0, \ 1) \\
p \ is \ the \ probability \ score \ of \ the \ estimator.
$$
</p>

y' is the predicted label and the raw value of y' is > 1 or < 0, to convert it into a probability score we use **sigmoid function** on top of y' to make the raw values as probabilities. By default, the output of the logistics regression model is the probability of the sample being positive, hence the probability score tends to be high and has an ideal score of 1 for positive class and a small probability value for negative class i.e. ideal value 0 for negative class. 

<center>
<img src="{{site.url}}/assets/images/ml/logloss.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Loss Function - Log Loss</p> 
</center>

* When the actual class y is 1: the second term in the Log Loss = 0 and we will be left with the first term 

<p>
   $$
   - y \ log( \ p (y') \ )
   $$
</p>

* When the actual class y is 0: The first term = 0 and the second term will turn into as follows

<p>
$$
- \ (1 \ - \ y\ ) \ log(1 \ - \ p(y') \ )
$$
</p>

By assigning the actual value for y and its estimated probability score, we find if the predicted probability leans towards the class which is close to an actual class, then the loss values are reduced, otherwise the loss is increased. I would encourage you to assign the value of y=1 and p(y')=0.1 and then p(y')=0.9.

When a number of classes> 2 in multiclass classification, we calculate a separate loss for each class label per observation and sum the result.

<p>
$$
−∑_{c=1}^M \ y_{o,c} \ log(p_{o,c}) \\

where \ c - \ number \ of \ classes.
$$
</p>

#### Cross Entropy Loss Code

 ```python

        def CrossEntropy(yHat, y):
            if y == 1:
            return -log(yHat)
            else:
            return -log(1 - yHat)

```

For more information on log loss, [find this amazing blog on Log Loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a).

### Small titbit from Wikipedia for **Selection Of Loss Function**

[W. Edwards Deming](https://en.wikipedia.org/wiki/W._Edwards_Deming) and [Nassim Nicholas Taleb](https://en.wikipedia.org/wiki/Nassim_Nicholas_Taleb) argue that **empirical reality**, not nice mathematical properties, should be the sole basis for selecting loss functions, and real losses often aren't mathematically nice and aren't differentiable, continuous,  symmetric, etc. For example, a person who arrives before a plane gate closure can still make the plane, but a person who arrives after can not, a discontinuity and asymmetry which makes arriving slightly late much more costly than arriving slightly early. In drug dosing, the cost of too little drug maybe lacks efficacy, while the cost of too much may be tolerable toxicity, another example of asymmetry. Traffic, pipes, beams, ecologies, climates, etc. may tolerate increased load or stress with little noticeable change up to a point, then become backed up or break catastrophically. These situations, Deming and Taleb argue, are common in real-life problems, perhaps more common than classical smooth, continuous, symmetric, differentials cases.[[13\]](https://en.wikipedia.org/wiki/Loss_function#cite_note-13)

Long story short, the loss function built should be based on the problem at hand and how small changes in some factors have a significant impact on the system.

If you've liked this post, please don't forget to subscribe to the newsletter.

### Reference

* [ML Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

* Wikipedia

* Sklearn Documentation
