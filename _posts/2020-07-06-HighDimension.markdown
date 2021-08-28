---
layout: post
title:  Dimension Reduction
description: Projecting High Dim. To Low Dim.
category: Blog
date:   2020-07-06 13:43:52 +0530
---

### Topics covered

1. What is Curse of Dimensionality
2. Dimensionality Reduction
3. Why Dimensionality Reduction is important
4. Dimension Reduction Using PCA
5. Techniques to overcome the Curse of Dimensionality

<center>
<img src="{{site.url}}/assets/images/highdimension/highdimension.jpeg" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Curse of Dimensionality</p>
</center>

### Curse Of Dimensionality

It refers to phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience.

### What is Dimensionality Reduction

It is a technique to transform the X’s (1 top)predictors/independent variables into a linear combination of the predictor with a reduced number of transformed Z’s(1 to m) variables, where m < p. What happens when we don’t reduce the dimension? Curse Of Dimensionality.

In machine learning and statistics, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables.

### Why Dimensionality Reduction is important

**Nowadays, Data comes in all forms video, audio, images, texts, etc., with a huge number of features. Is it that all features are relevant ?, NO, not all features are important or relevant. Based on the business requirement or redundancy nature of the data captured we have to reduce the feature size through Feature Selection and Feature Extraction. These techniques not only reduce computation cost but also helps in avoiding misclassification because of a highly correlated variable.**

### Dimension Reduction Using PCA

Here we’ll try to understand PCA by working on MNIST Dataset. Since images have higher dimensions, we’ll be loading a built-in dataset from sklearn.datasets. We make all the import statements respective from loading the dataset to measuring the metrics.

### Loading Packages

```python

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA , TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics as m
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import skimage as img
import seaborn as sns
```

We are loading the digits dataset for our problem. We can notice that we have around 64 features representing the digit.

```python

load_digits = load_digits()
```

We can visualize the all-column (64) value of an image as an 8x8 pixel value in grayscale.

<center>
<img src="{{site.url}}/assets/images/highdimension/visualize.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Dimension Reduction</p>
</center>

The variance of a feature concerning the target variable, explains a lot about the relationship between the feature and the target variable. We have a list of components included in our list over which we try to explain the variance. As the components are increasing the variance also increases.

First, we are iterating over several components to find the best match between variance and component. And as we notice the number of components is increased, the explained variance is also increasing but after a certain value, the increased components don't increase the explained variance which causes saturates the explained variance.

From the results, we can notice how the variance value increases to 1 when all the 63 components are included.

```python
X = load_digits.data
y = load_digits.target
variance = []
components = [4,8,12,16,20,24,28,32,63]
for x in list([4,8,12,16,20,24,28,32,63]):
    dimReduction = PCA(n_components=x)
    X_DR_PCA = dimReduction.fit_transform(X)
    print("Explained Variance with", x ," Components: " ,dimReduction.explained_variance_ratio_.sum())
    variance.append(dimReduction.explained_variance_ratio_.sum())
    X_train,X_test,y_train,y_test = train_test_split(X_DR_PCA,y,test_size=0.25)
    RFC_2 = RandomForestClassifier()
    mnb = GaussianNB()
    RFC_2.fit(X_train,y_train)
    mnb.fit(X_train,y_train)
    y_pred = RFC_2.predict(X_test)
    y_prediction = mnb.predict(X_test)
    print("Accuracy Score with Random Forest Classifier",m.accuracy_score(y_test,y_pred))
    print("Accuracy Score with Gaussian NB",m.accuracy_score(y_test,y_prediction))
    print("--------------------------------------------------")
```
<center>
<img src="{{site.url}}/assets/images/highdimension/results.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Result</p>
</center>

```python

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(variance,components)
for xy in zip(variance, components):
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
plt.xlabel("Variance Explained")
plt.ylabel("Principal Components")
plt.show()
```

<center>
<img src="{{site.url}}/assets/images/highdimension/plots.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Plotting</p>
</center>

```python

pca_1_Comp = PCA(n_components=24)
X_1 = pca_1_Comp.fit_transform(X)
gnb = GaussianNB()

print("Explained Variance: ",pca_1_Comp.explained_variance_ratio_.sum())
X_train,X_test,y_train,y_test = train_test_split(X_1,y,test_size = 0.2,random_state=1)

gnb.fit(X_train,y_train)
y_predict = gnb.predict(X_test)
print("Accuracy: ",m.accuracy_score(y_test,y_predict))
```

**Explained Variance:  0.92**\
**Accuracy:  0.93**

From 64 Feature vectors to 24 Feature, it means we can maintain a good model without losing much information by reducing the variables which are redundant in this case. The changes in variance happen concerning several components, these changes saturate after n_components turns 24. Thus we can assign the n_components as 24 i.e. we can explain the maximum variance of 0.92 with 24 principal components at the accuracy of 93%.

Thus comes the ending to Dimensionality reduction, with a small note to think about dimensionality reduction with the movie Interstellar, Where Cooper stays in five dimensions (hypothetically) and which is difficult to visualize, but we try to understand the scene from our 2 dimension screen.

### Techniques to overcome the Curse of Dimensionality

There are several ways of dimensionality reduction such as feature selection and Feature Extraction.

1. PCA
2. Missing Value Ratio
3. Low Variance Filter
4. Backward Feature Elimination
5. Forward Feature Construction
6. High Correlation Filter

<center>
<img src="{{site.url}}/assets/images/highdimension/hd.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 5: 2D to 1D</p>
</center>


Let’s look at the image shown above. It shows 2 dimensions x1 and x2, which are let us say measurements of an object in Km (x1) and Miles (x2). Now, if you were to use both these dimensions in machine learning, they will convey similar information and introduce a lot of noise in the system, so you are better of just using one dimension. Here we have converted the dimension of data from 2D (from x1 and x2) to 1D (PC1), which has made the data relatively easier to explain.

### Principal Component Analysis

Principal Components Analysis means components that can explain the maximum amount of variance of the features concerning the target variable, if we include all features as components then we get the variance of 1. PCA transforms all the interrelated variables into an uncorrelated variables. Each uncorrelated variable is a Principal Component and each component is a linear combination of the original variable.

Each uncorrelated variable or component holds feature information which is explained as a variance. Each component with its variance adds up to 1. Since each principal component is a combination of the original variable, some principal components explain more variance than others.

The variance explained by one principal component is uncorrelated with other principal components which means with each component we are learning or explaining a new feature. Now raises a question, how many components will be able to explain the maximum variance?. We don’t have any textbook method for calculating the number of components for a given number of features or variables. But We can maintain a variance threshold that needs to explain by the variance of the components.

**Consider we have set a threshold variance of 0.8, and if have ten components with a variance as follows 0.3, 0.25, 0.15, 0.1, 0.08, 0.08, 0.07, 0.07. then we can notice 0.3 is a component with maximum variance and is called as First Principal Component. Now since the threshold is kept at 0.8, we can add up components until it reaches a variance of 0.8.**\
**By adding the first 3 components, we have variance explained at 0.7 and by including the 4th component we reach a variance of 0.8. So we can including 4 components instead of ten components thus reducing the dimension from 10 to 4.**

### Missing Value Ratio

In a Dataset, We have various columns and each column contains values but if data columns contain missing values then we have to think about the feature selection based on the Missing value ratio i.e. we can set a threshold for several Missing values a column may contain and if we have a ratio of Missing value greater than the threshold then we can drop the feature.

Higher the threshold, the more aggressive the drop-in features.

### Low Variance Filter

It is similar to PCA Conceptually i.e. if a column carries very little information or has variance lower than a threshold value then we can drop the feature i.e. variance value acts as Filter for Feature Selection.

Variance is range-dependent, so normalization is required before applying this technique.

### Backward Feature Elimination

In simple terms, If a model is trained on the n-input feature and the error rate is calculated, then again if the model is trained on the n-1 feature and error rate is calculated, now if the error rate is increased by a small value then the feature is dropped from the dataset.

Backward feature Elimination can be performed iteratively to get a better feature.

### Forward Feature Construction

In this Feature Selection process, we train a model with one feature and calculate the performance measure. We kept adding a feature, one by one, and calculate the performance of the performance decreases with increase in Feature, we should drop the feature and if the performance increases with increase in Feature, We iteratively add a feature to the model.

### High Correlation Filter

Here, If the columns present in the dataset are highly correlated then the information becomes redundant and we drop these highly redundant variables from features. We can calculate the **correlation coefficient** between Numerical columns/variables. We can calculate the ‘correlation coefficient’ between Nominal columns/variables.

We can use the **Pearson product-moment coefficient** between Numerical columns/variables. We can use the ‘Pearson Chi squared’ value between Nominal columns/variables. Before doing correlation operation, Perform normalization on the columns as correlation is scaled sensitive.

### Note

Both Forward Feature Construction and Backward Feature Elimination are computationally expensive tasks.
