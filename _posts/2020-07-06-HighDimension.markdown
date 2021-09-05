---
layout: post
title:  Dimension Reduction
description: Projecting high dimensions to low dimensions
category: Blog
date:   2020-07-06 13:43:52 +0530
---

### Curse Of Dimensionality

It refers to phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience.

### Dimensionality Reduction

It is a technique to transform the X’s (1 to p) predictors/independent variables into a linear combination of the predictor with a reduced number of transformed Z’s (1 to m) variables, where m < p. What happens when we don’t reduce the dimension? Curse Of Dimensionality.

In machine learning and statistics, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration into a set of principal variables/components.

### Why Dimensionality Reduction is important

Data comes in all forms video, audio, images, texts, etc. Each with a large number of features, does it mean that all features are relevant? NO, not all features are important or relevant. Based on the business requirement or redundancy nature of the data captured, we have to reduce the feature size through Feature Selection and Feature Extraction. These techniques not only reduce computation cost but also helps in avoiding misclassification because of a highly correlated variable.

### Dimension Reduction Using PCA

We will understand PCA by working on MNIST Dataset. Since images have higher dimensions, we will load the built-in dataset from sklearn.datasets. We make all the import statements respective from loading the dataset to measuring the metrics.

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

        load_digits = load_digits()

```
We are loading the digits dataset for our problem. We can notice that we have 64 features representing a digit. We can visualize all the 64 column values of an image as an 8x8 two-dimensional matrix in grayscale.

<center>
<img src="{{site.url}}/assets/images/highdimension/visualize.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Loading MNIST Image</p>
</center>

A feature's variance concerning the target variable explains a lot about the relationship between the feature and the target variable. We have a list of components included in our list, over which we try to explain the variance. As the components are increasing, the variance also increases.

First, we iterate over several components to find the best match between the variance and component. And we notice that increasing the components also increases the explained variance. But after a certain value, the increased components don't increase the explained variance and it starts saturating.

From the results, when all the components are included in the model, the variance adds up to 1.

```python

        X = load_digits.data
        y = load_digits.target

        variance = []
        components = [4,8,12,16,20,24,28,32,63]

        for x in list([4,8,12,16,20,24,28,32,63]):
            dimReduction = PCA(n_components=x)
            X_DR_PCA = dimReduction.fit_transform(X)
            print("Explained Variance with", x ," Components: " ,dimReduction.explained_variance_ratio_.sum())

            #explained variance ratio
            variance.append(dimReduction.explained_variance_ratio_.sum())
            X_train,X_test,y_train,y_test = train_test_split(X_DR_PCA,y,test_size=0.25)

            #model declaration
            RFC_2 = RandomForestClassifier()
            mnb = GaussianNB()

            #model fitting
            RFC_2.fit(X_train,y_train)
            mnb.fit(X_train,y_train)

            #model prediction
            y_pred = RFC_2.predict(X_test)
            y_prediction = mnb.predict(X_test)

            #accuracy measurement
            print("Accuracy Score with Random Forest Classifier",m.accuracy_score(y_test,y_pred))
            print("Accuracy Score with Gaussian NB",m.accuracy_score(y_test,y_prediction))
            print("--------------------------------------------------")

```
<center>
<img src="{{site.url}}/assets/images/highdimension/results.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Components vs Accuracy</p>
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
<p>Figure 3: Plotting</p>
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

        """
        Explained Variance:  0.92
        Accuracy:  0.93
        """

```
**Understanding The Plot**

The dimension reduction from 64 features to 24 features doesn't affect the results massively. It means we can maintain a good model without losing much information by reducing the variables which are redundant in this case. The changes in variance happen concerning the several components, the variance saturates after n_components turns 24. Thus we can assign the n_components as 24 i.e. we can explain the maximum variance of 0.92 with 24 principal components at the accuracy of 93%.

### Techniques to overcome the Curse of Dimensionality
Feature selection and feature extraction are the common techniques to overcome dimension reduction.

* PCA
* Missing Value Ratio
* Low Variance Filter
* Backward Feature Elimination
* Forward Feature Construction
* High Correlation Filter

<center>
<img src="{{site.url}}/assets/images/highdimension/hd.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: 2D to 1D</p>
</center>


Let us look at the image shown above. It has two dimensions x1 and x2, which measure an object in Km (x1) and Miles (x2). Now, if you were to use both these dimensions in machine learning, they will convey similar information and introduce a lot of noise in the system, so you are better of just using one dimension. Here we have converted the dimension of data from 2D (from x1 and x2) to 1D (PC1), which has made the data relatively easier to explain.

### Principal Component Analysis

Principal Components Analysis is a process to identify components that can explain the maximum amount of variance of the features concerning the target variable, if we include all features as components then we get the variance of 1. 

PCA transforms all the interrelated variables into an uncorrelated variables. Each uncorrelated variable is a Principal Component and each component is a linear combination of the original variable. Each uncorrelated component holds the feature information, which is explained as a variance. Since each principal component is a combination of the original variable, some principal components explain more variance than others.

The variance explained by one principal component is uncorrelated with other principal components. It means that with each component, we are learning or explaining a new feature. 

Now raises a question, how many components are required to explain the maximum variance?. We don’t have any textbook method for calculating the number of components for the given number of features. But we can maintain a variance threshold that needs to be explained by the variance of the components.

*Consider we have set a threshold variance of 0.8, and if have ten components with a variance as follows 0.3, 0.25, 0.15, 0.1, 0.08, 0.08, 0.07, 0.07. then we can notice 0.3 is a component with maximum variance and is called First Principal Component. Now since the threshold is at 0.8, we can add up components until it reaches a variance of 0.8.*

*By adding the first three components, we have variance explained at 0.7. By including the 4th component, we reach a variance of 0.8. So we can include four components instead of ten components. Thus, reducing the dimension from 10 to 4.*

### Missing Value Ratio

In many datasets, we have columns with many missing values. We can perform feature selection based on the missing value ratio i.e. we can set a threshold for columns, where if the column's 50% values are missing then we can drop the column. Higher the threshold, the more aggressive the drop-in features.

### Low Variance Filter

It is similar to PCA conceptually i.e. if a column carries very little information or has variance lower than a threshold value then we can drop the feature i.e., variance value acts as a filter for feature selection.

Variance is range-dependent, so normalization is required before applying this technique.

### Backward Feature Elimination

At first, with the n-input features, we train a model and calculate the error e1. Next, we train a model with the n-1 features and calculated the error e2. Now, if the error e2 is increased, then the feature that was dropped is brought back. Otherwise, we iteratively keep removing features and check if the error increases or decreases.

### Forward Feature Construction

At first, we train a model with one feature and then calculate the performance. And then, we keep adding the feature one by one and calculate the performance. If the performance decreases with an increase in the feature, then we should drop the added feature. And if the performance increases with an increase in feature, we iteratively add more features into the model.

### High Correlation Filter

If the columns present in the dataset are highly correlated, then the information becomes redundant, and we drop these highly redundant variables from our features. 

There are various correlation techniques to find the redundancy between the variables. Like **correlation coefficient**, **Pearson product-moment coefficient**, **Pearson Chi-squared**. Each correlation technique depends on the variable type like categorical or numerical etc. Before doing correlation operation, perform normalization on the columns as correlation is scaled sensitive.

Both Forward Feature Construction and Backward Feature Elimination are computationally expensive tasks.
