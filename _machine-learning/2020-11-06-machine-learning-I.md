---
layout: machine-learning
title: Machine Learning Series - Part I
description: "What are Machine Learning and different types of learning"
date:   2020-11-06 13:43:52 +0530
---
{% include mathjax.html %}

<center>
<img src="{{site.url}}/assets/images/ml/ml.jpg"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Machine Learning</p> 
</center>

**What is Machine Learning?**

Machine learning is an application of artificial intelligence (AI) that provides the systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can use the data to learn from it.

**Other definitions**

Machine Learning is essentially a form of applied statistics with increased emphasis on the use of computers to statistically estimate complicated functions and a decreased emphasis on providing confidence intervals around these functions.

"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

**Core Components of Machine Learning**

* Data
* Model
* Learning

**Data**

Data is at the heart of machine learning. It is believed that the larger the corpus of data, the better is the generalization of the learning model. Fewer data results in poor generalization, thus causing the model to perform poorly on unseen data.

In the real world, the data is unclean and is not consumable by the model directly. It requires a series of preprocessing steps to clean the data and make it available for the model to work on it.

**Model**

A model is a learnable function, which learns based on data. Complex data with hundreds of features requires complex functions. 

For instance, appraisals in the workplace is a complex function because various factors play a key role in your appraisal like your effort in a team, quality deliverables, years of experience, cordial relation with managers 😉 etc. We cannot simply make a linear function like increasing salary with increasing years of experience to model such a problem. 

Hence, the model learns these different factors and their interrelation and then extracts the hidden patterns from the data, these patterns are generalized over a function to make up a model.

**Learning**

Learning, a model is said to learn from data. Learning here refers to identifying or recognizing the pattern in the data. 

A model performs well if it predicts accurately on the unseen data. Unseen data is technically called test data. There are few terms related to learning like a generalization, test error, validation error which we will discuss in the part of the series.

**Common terms in Machine Learning**

**Data** is a set of observations we make about a system. For instance, for patient records, we keep note of their health habits, existing health conditions, diseases XYZ exists or not, etc.

**Predictor**, are a set of variables, which helps in describing the system. From the above patient records example, if we want to predict, if the patient has disease XYZ, then our predictor variable will include health habits (like smoking, drinker, etc), existing health conditions like blood pressure, diabetes, etc. A predictor is also referred to as an independent variable.

<center> 
<img src="{{site.url}}/assets/images/ml/mlflow.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Machine Learning Flow</p> 
</center>

**Label** is a dependent variable, it tells what the model is supposed to predict. From the patient records system, our label is the variable disease XYZ, here we are predicting if a patient has disease XYZ or not. 

Labels are also called target or dependent variables. Dependent as it depends on predictor variables.

**Model** is a machine learning algorithm, which is applied or trained on data like patient records to predict if a patient has disease XYZ. 

In the above example, whether a patient has disease XYZ or not is called a binary classification problem, where the label variable takes in Yes or No values, Binary classification models like Logistic Regression can be utilized here.

**Parameters** are tunable knobs in machine learning algorithms, there are different parameters available for each algorithm like *depth of tree* in decision tree algorithm, *penalty* in Logistic Regression, etc. 

These tunable parameters are **Hyperparameter**, which are set manually by a user who is performing machine learning. Other sets of parameters are learned by machine learning algorithms like *coefficient of X's* in Linear regression, Weights in Deep Learning Algorithms, etc.

**Dataset** is the data on which we apply machine learning algorithm after cleaning and structuring it in the form, such that it is consumable by ml algorithm. Consider we have 1000 patient records with the target as *disease XYZ* with value 0 or 1. 0 represents no disease and 1 represents disease. To perform machine learning, we split the dataset into multiple sets like a train set, validation set, and test set. 

For training, we use a train set. For identifying the hyperparameter and parameters of the model, we use a validation set. To check the final performance of the model, we use a test set.

**Training** is a process to learn the parameters of the machine learning algorithm. During this process, we keep track of training errors, as the training continues to take place, the training error reduces. Training is done on a train set.

**Evaluating** is a process to learn hyperparameters and try out or experiment with the best possible hyperparameter for the learning algorithm. Evaluation can take place simultaneously with training. During the evaluation, we keep track of validation errors. Evaluation is done on the validation set of the dataset.

**Testing** is a process to confirm whether the model is generalized for unseen data or not. If the model performs poorly, we need to get back to training with different models or tune the model’s hyperparameters or look for more data, etc.

Note: Sometimes the training error will reduce but the validation error may fail to reduce. It happens because of a bias issue in the model. More detail in the next blog.

<center>
<img src="{{site.url}}/assets/images/ml/ml_algo.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/><br>
<p>Figure 3: Machine Learning Algorithms</p> 
</center>

**Types of Machine Learning**

* **Supervised Learning**
* **Unsupervised Learning**
* **Reinforcement Learning**

**Supervised Learning**

   The Algorithm learns from the labeled data. It identifies or recognizes the pattern from data with labels and associates those patterns to unlabeled data or unseen data.

A dataset has features ( X's ) and target ( y ). The target can be a discrete value or a continuous value based on which we can call the problem Classification or Regression respectively.

*Classification*

The problem is termed as either Binary or Multiclass classification. The target variable for the classification problem is either 0/1 or else multiple values like Dogs, people, cats, etc. It has **categorical outcomes**.

*Regression* 

The problem is related to predicting a continuous value (Numerical). For instance, predicting the house price or a stock price. It has **numeric outcomes**.

Each dataset has features (X) and target (y) variables. These features are attributes that affect the target variable either directly or indirectly. Now, this relationship between X and y is built by a machine learning algorithm.

For example Data on house price is based on various factors like area, rooms, lawn, and other details, where we predict the value of the house. So our label is House Price.
   
**Unsupervised Learning**

   The Algorithm learns from unlabeled data. A dataset with no label is trained using unsupervised learning i.e. the patterns of such a dataset are learned by using clustering techniques or other self-supervised approaches.

   For example, Unstructured text documents where we cluster text or paragraphs based on word binding in the sentence using clustering algorithms, to group documents to particular topics.
   
**Reinforcement Learning**

   The Algorithm learns from performing some action and receiving rewards(+ve/-ve) for those actions. Here, the algorithms interact with an environment, so there is a feedback loop between the learning system and its experience. For instance: Self Driving Cars and Game Playing Agents.

As part of this series, I will share separate blogs on each topic. Please make sure to subscribe to the newsletter if you wish to get an update on future articles. Thanks for reading.