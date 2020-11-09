---
layout: machine-learning
title: Machine Learning - I
description: "What is Machine Learning and different types of learning"
date:   2020-11-06 13:43:52 +0530
---
{% include mathjax.html %}

### MACHINE LEARNING

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

<center>
<img src="{{site.url}}/assets/images/ml/ml.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="1000" height="600"/><br>
<p><b>Figure 0:</b> Machine Learning</p> 
</center>

**TYPES OF LEARNING**

* **Supervised Learning**
* **Unsupervised Learning**
* **Reinforcement Learning**

**SUPERVISED LEARNING**

   The Algorithm learns from labeled data. It identifies/recognizes the pattern from data with labels and associates those pattern to unlabeled data.
   For example: Data on house price based on various factors like area, rooms, lawn and other details, where we predict the value of the house. So our label is House Price.
   
**UNSUPERVISED LEARNING**

   The Algorithm learns from unlabeled data.
   For example: Unstructured text documents where we cluster text or paragraphs based on word binding in the sentence using clustering algorithms, to group documents to particular topics.
   
**REINFORCEMENT LEARNING**

   The Algorithm learns from performing some action and receiving rewards(+ve/-ve) for those actions. For instance: Self Driving Cars and Game Playing Agents.

<center>
<img src="{{site.url}}/assets/images/ml/ml_algo.png"  style="zoom: 5%  background-color:#DCDCDC;" width="1000" height="600"/><br>
<p><b>Figure 1:</b> Machine Learning Algorithms</p> 
</center>

**SUPERVISED ALGORITHM**

A dataset has feature ( X's ) and target ( y ). The target can be a discrete value or a continuous value based on which we can call the problem as Classification or Regression respectively.

**CLASSIFICATION** problem is classified as Binary or Multiclass classification. Either target value is 0/1 or else multiple values like Dogs, Person, Cat. It answers the yes or no question. It has categorical outcomes.

**REGRESSION** problem is like predicting a value (Numerical). For instance, predicting the house price or stock value of a product. It answers how much questions. It has numeric outcomes.

Each dataset has features and target variable. The features are parameters which affects the target variable either directly or indirectly. Now, this relationship between X and y is built by a ML algorithm. There are various supervised and unsupervised ML algorithm as follows :

* **Naive Bayes**
* **Linear Regression**
* **Logistic Regression**
* **Support Vector Machine**
* **Decision Trees**
* **Neural Networks**
* **Ensemble Model**
* **KMeans**
* **Hierarchical Clustering**
* **DBSCAN Clustering**
* **Gaussian Mixture Model**
* **others (Xgboost)**
            
<p align='center'>. . . . .</p>
