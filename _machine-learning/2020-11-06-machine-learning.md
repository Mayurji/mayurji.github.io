---
layout: machine-learning
title: Basics of Machine Learning
description: "Key concepts of Machine Learning"
date:   2020-11-06 13:43:52 +0530
---
{% include mathjax.html %}

### BASICS OF MACHINE LEARNING

What is Machine Learning ? 

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

**TYPES OF LEARNING**

**Supervised Learning**\
**Unsupervised Learning**\
**Reinforcement Learning**


**SUPERVISED LEARNING**

   The Algorithm learns from labeled data. It identifies/recognizes the pattern from data with labels and associates those pattern to unlabeled data.
   For example: Data on house price based on various factors like area, rooms, lawn and other details, where we predict the value of the house. So our label is House Price.
   
**UNSUPERVISED LEARNING**

   The Algorithm learns from unlabeled data.
   For example: Unstructured text documents where we cluster text or paragraphs based on word binding in the sentence using clustering algorithms, to group documents to particular topics.
   
**REINFORCEMENT LEARNING**

   The Algorithm learns from performing some action and receiving rewards(+ve/-ve) for those actions. For instance: Self Driving Cars and Game Playing Agents

**MODELING A SUPERVISED ALGORITHM**

A dataset has feature ( X's ) and target ( y ). The target can be a discrete value or a continuous value based on which we can call the problem as Classification or Regression respectively.

**CLASSIFICATION** problem is classified as Binary or Multiclass classification. Either target value is 0/1 or else multiple values like Dogs, Person, Cat. It answers the yes or no question. It has categorical outcomes.

**REGRESSION** problem is like predicting a value (Numerical). For instance, predicting the house price or stock value of a product. It answers how much questions. It has numeric outcomes.

Each dataset has features and target variable. The features are parameters which affects the target variable either directly or indirectly. Now, this relationship between X and y is built by a ML algorithm. There are various supervised ML algorithm as follows :

**Naive Bayes**\
**Linear Regression**\
**Logistic Regression**\
**Support Vector Machine**\
**Decision Trees**\
**Neural Networks**\
**Ensemble Model**\
**others (Xgboost)**

**THINGS TO KEEP IN MIND**

When designing features or algorithms for learning features, our goal is to separate the **factors of variation** that explain the observed data.

**FEATURES** are components in your dataset which helps in building ML Algorithm. There are different types features like Categorical, Numerical, Ordinal etc. So before applying algorithm on a dataset, we need to convert the dataset into a format, which is consumable by algorithm. One such vital technique is handling categorical value, by converting it into an one-hot encoding, similar to mapping category to numeric value.

**FEATURE ENGINEERING**

Any problem in Machine Learning or Deep Learning requires some amount of feature engineering, one cannot simply do a ```python model.fit(x, y)``` and get SOTA results. Feature Engineering requires creativity and understanding of the domain. For instance, cost of house of size 'a' in X locality is priced higher than same size house in Y locality, even though locality is not a feature in your raw dataset.

**ONE HOT ENCODING**

There are few algorithms like decision trees, which takes categorical feature like color with values as 'red', 'blue' etc as inputs but majority of the algorithms requires feature values to be numerical. To convert these features, we can use one hot encoding. 

```python
      orange = [0, 0, 1]
         red = [1, 0, 0]    
        blue = [0, 1, 0]
```

One hot encoding increases the dimensionality of the feature vector, but transform colors as red as 1, blue as 2 etc would bring in order to values of the colors feature and would mess the algorithm's decision making.

**HANDLING MISSING VALUES**

The raw data comes in different problems with it, sometimes the features values are missed if the dataset is prepared manually. To overcome such missing value problem, one do the following,

Drop the samples, if the dataset is big enough.\
Perform Data Imputation to fillin the gaps of missing values.\
Few algorithms are robust to missing values.

**DATA IMPUTATION**

One way to impute is to find the mean of the features and replace the missing values.(Careful if outliers are present)\
Replace with value which is outside the range of features i.e. if feature x is [0, 1] then replace missing value with -1 or 2. It provides a distinct feature values for this sample alone.\
Replace with value which is in middle of the range i.e. if feature x is [-1, 1] then replace missing value with 0. It makes algorithm to get less affected by 0.
   
**SCALING** of a feature is an important task before building/applying a ML algorithm.
    
**WHY SCALING?**
   
   For instance, If an algorithm is not using feature scaling, then it can consider the value 300 meter to be greater than 5 km, which is not true and in this case, the algorithm will give wrong predictions by giving more importance to 300 meter than 3 km. So, we use Feature Scaling to normalize the values of it.

**TYPES OF SCALING**
   
Min-Max Scaling or Normalization\
Mean Normalization\
Standardization or Z-score Normalization

**MIN-MAX SCALING** helps in rescaling the feature values into the range of [0,1] or [-1,1].

<p>$$x' = {x - min(x)  \over max(x) - min(x)}$$</p>

**MEAN NORMALIZATION** 

<p>$$x' = {x - average(x) \over max(x) - min(x)}$$</p>

**STANDARDIZATION** of features makes the value of each feature in the data to have zero mean and unit variance. It is a widely used normalization technique as major algorithms like SVM, Neural Nets and Logistic Regressions follow such standardization.

<p>$$x' = {x - \mu \over \sigma}$$</p>

<p>$$\sigma$$ is Standard Deviation.</p>

**BIAS** refers to the tendency of a measurement process to over- or under-estimate the value of a population parameter. In survey sampling, for example, bias would be the tendency of a sample statistic to systematically over- or under-estimate a population parameter.

**UNDERFITTING** occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data. Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough. Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.

**VARIANCE**, in the context of Machine Learning, is a type of error that occurs due to a model's sensitivity to small fluctuations in the training set. High variance would cause an algorithm to model the noise in the training set. This is most commonly referred to as overfitting.

**OVERFITTING** refers to a model that models the training data too well. Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.

**BATCH GRADIENT DESCENT** refers how a gradient changes are applied to the weight matrix. If we have 1000 data points, then the model is trained on 1000 data points before any update is made to the weight of the model.

**STOCHASTIC GRADIENT DESCENT** refers how a gradient changes are applied to the weight matrix. If we have 1000 data points, then the model is trained on 1 data point and an update is made to the weight of the model.

**MINI-BATCH GRADIENT DESCENT** refers how a gradient changes are applied to the weight matrix. If we have 1000 data points, then we assign a value to batch_size, if batch_size is 10, then the model is trained on 10 data point and an update is made to the weight of the model. This happens iteratively taking 10 data points and updating.

**REGULARIZATION**

We have a simple dataset with two features and a target variable, we can either use simple model with 2 coefficients for two variables or else we use a complex model which will have more coefficients and will overfit the simple dataset. The complex model will not generalize for the new data as it is a overfit model. To overcome and choose a simple model, we can use regularization.

**L1 REGULARIZATION**

While calculating the error(E) we add the absolute value of the coefficients of the model. 

In simple model case, we have 2 coefficients $$w_1, w_2$$ then,

<p>$$TotalError = E + |w_1| + |w_2|$$</p>

In Complex model case, lets say have 5 coefficients $$w_1, w_2, w_3, w_4, w_5$$ then,

<p>$$TotalError = E + |w_1| + |w_2| + |w_3| + |w_4| + |w_5|$$</p>
    
So we get smaller error for simple model and will use the same for the generalization.

**L2 REGULARIZATION**

While calculating the error(E), we square the value of the coefficients of the model. 

In simple model case, we have 2 coefficients $$w_1, w_2$$ then,

<p>$$TotalError = E + (w_1)^2 + (w_2)^2$$</p>

In Complex model case, lets say have 5 coefficients $$w_1, w_2, w_3, w_4, w_5$$ then,

<p>$$TotalError = E + (w_1)^2 + (w_2)^2 + (w_3)^2 + (w_4)^2 + (w_5)^2$$</p>

So we get smaller error for simple model and will use the same for the generalization. More the number of parameters, more complex the model is.

**HOW TO SELECT REGULARIZATION?**
Based on the complexity of the data, the model tends to be complex. So the lambda value acts like a switch either to increase the complexity or not. If we keep a small value of lambda and multiple it, with the complexity part of the model i.e. "w" parameters then we get smaller error compared to the simple model with its "w" parameters. And if lambda is large, then we punish the complexity part highly and thus making the complex model with great error.

The complexity of the model is defined by the number of the parameters(w) in the equation.

### DATASET SPLIT

Before applying ML Algorithm, we should check the dataset and split it for modelling for ML. We should split our dataset into training, testing and validation set. It helps in understanding certain factors of ML model like bias and variance i.e. also termed as Underfitting and Overfitting.

### TRAINING SET

Training Set: Before Big Data came into the picture of analytics, training data made around 70-75% data of total data. But with millions of records or data, training data occupies 95% of the total data. We model our algorithm on training set.

### VALIDATION AND TESTING SET

Validation Set: A set of examples used to tune the parameters [i.e., architecture, not weights] of a classifier, for example to choose the number of hidden units in a neural network. Before Big Data era, validation set occupied 15-12.5% data, but in Big Data times, it occupies 2.5%.

Test Set: A set of examples used only to assess the performance [generalization] of a fully specified classifier. Before Big Data era, validation set occupied 15-12.5% data, but in Big Data times, it occupies 2.5%. 

The validation and testing set are also called as hold-out sets.

### PRINCIPLE COMPONENT ANALYSIS

Principal Component Analysis is a variable reduction technique. PCA believes that if there are large number of observed variables, then some of these observed variables tend to have redundancy of information, and PCA tries to capture the variance of these variables and creates lesser number of variables called as Principal Components.

### DIFFERENT MACHINE LEARNING MODELS

**Naive Bayes**\
[Blog](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c) \
[Implementation](https://github.com/udacity/NLP-Exercises/blob/master/1.5-spam-classifier/Bayesian_Inference_solution.ipynb)

**Support Vector Machine**\
**Ensemble Methods**\
**KMeans**\
**Hierarchical Clustering**\
**DBSCAN Clustering**\
**Gaussian Mixture Model**

### FACTORS FOR SELECTING ALGORITHM

Explainability\
In-memory Vs Out-Memory\
Number of Features and examples\
Categorical Vs Numerical Features\
Nonlinearity of the data\
Training speed\
Predicton speed

<center>
<img src="{{site.url}}/assets/images/ml/ml_map.png"  style="zoom: 5%  background-color:#DCDCDC;" width="1000" height="600"/><br>
<p><b>Figure 1:</b> Algorithm Selection</p> 
</center>

#### CONFUSION MATRIX

Consider a model trained two classify Cat and Dog images. And after training, we are testing the model on 100 random images of Dogs and Cats with 50 each and get an accuracy of 85%. It means that the model has misclassified 15 images. Now let's consider that out of 15 images 10 Dog images were misclassified as Cat and 5 Cat images are misclassified as Dog.

<center>
<img src="{{site.url}}/assets/images/ml/Confusion_matrix_2.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 2:</b> Confusion Matrix</p> 
</center>

From above image, we can see our True Positive is Cat's Image and True Negative is Dog's Image. And a Cat getting misclassified is called as False Negative and when a Dog is misclassified is called as False Positive. In essence, it consider Positive as Cat and Negative as Dog.

<center>
<img src="{{site.url}}/assets/images/ml/confusion_matrix.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 3:</b> Confusion Matrix</p> 
</center>

#### PRECISION AND RECALL

**Precision** can be seen as a measure of exactness or quality, whereas **recall** is a measure of completeness or quantity. In simple terms, high precision means that an algorithm returned substantially more relevant results than irrelevant ones, while high recall means that an algorithm returned most of the relevant results.

<center>
<img src="{{site.url}}/assets/images/ml/precision.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 4:</b> Precision</p> 
</center>

**Precision** informs us, how well the model predicts the positive class. It is also known as PPV (Positive Predictive Value).

   >Precision = True Positive / (True Positive + False Positive)   

**Recall** informs us, how well the model predicts the right class for a label. It is also known as Sensitivity.

<center>
<img src="{{site.url}}/assets/images/ml/recall.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 5:</b> Recall</p> 
</center>

   >Recall = True Positive / (True Positive + False Negative)

#### SENSITIVITY AND SPECIFICITY

**Sensitivity** also known as recall or true positive rate, measures the proportion of actual positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition).

> Sensitivity = True Positive / (True Positive + False Negative)

**Specificity** also known as true negative rate, measures the proportion of actual negatives that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).

<center>
<img src="{{site.url}}/assets/images/ml/specificity.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 5:</b> Specificity</p> 
</center>

> Specificity = True Negative / (True Negative + False Positive) 

### DRAWBACK OF USING ONLY PRECISION AND RECALL

Consider having built a model to classify whether a patient suffers from cancer or not from a sample of 1000000 (1 million). We know that 100 in 1 million suffer from cancer. After training, we test the model and it misclassifies the 100 cancer as 90 with no cancer (False Negative) and 10 with (True Positive). It correctly classifies Non-cancer patient as no cancer.

Lets find accuracy of such model, accuracy = (999900+10) / 1000000 = 99.99%. Wow what a great model, wrong such sensitive usecase we cannot relie on accuracy.

Lets calculate Precision, P = 10/(10+0) = 100%.

Lets calculate Recall, R = 10/(10 + 90) = 10%.

Clearly we can see that the model failed to classify the cancer patient, and these metrics gets dominated by class with large numbers which is otherwise referred as **Imbalanced class problem**. Here, the number of non cancer class are 999900 and cancer class are 100. So its vital to check if these metric really telling story we want to hear.

#### F1-SCORE AND F1-BETA SCORE

**F1 Score**

In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall).

<center>
<img src="{{site.url}}/assets/images/ml/f1-score.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 5:</b> F1-score</p> 
</center>

**F1 Beta Score**

As we have seen in drawback of using metrics like precision and recall alone would result it false understanding of the model output. Based on usecase, we can give importance to precision and recall with help of **F1 Beta score**.

Find this below snippet from wikipedia:

<center>
<img src="{{site.url}}/assets/images/ml/f1_beta_score.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 5:</b> F1 Beta Score</p> 
</center>

#### [ROC CURVE](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

ROC- Receiver Operating Characteristics, its a plot between True Positive rate (Sensitivity) vs False Positive Rate (1 - specificity). Its a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

ROC curves are frequently used to show in a graphical way the connection/trade-off between clinical sensitivity and specificity for every possible cut-off for a test or a combination of tests. In addition the area under the ROC curve gives an idea about the benefit of using the test(s) in question.

**REFERENCE** [ROC Curve- what and How ?](https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used)
