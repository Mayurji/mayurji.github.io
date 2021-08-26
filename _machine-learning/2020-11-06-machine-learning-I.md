---
layout: machine-learning
title: Machine Learning - I
description: "What is Machine Learning and different types of learning"
date:   2020-11-06 13:43:52 +0530
---
{% include mathjax.html %}

<center>
<img src="{{site.url}}/assets/images/ml/yannis-h-uaPaEM7MiQQ-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="75%" height=auto/><br>
<p><span>Photo by <a href="https://unsplash.com/@yanphotobook?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Yannis H</a> on <a href="https://unsplash.com/s/photos/teaching?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span></p>
</center>

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can use data to learn from it.

**Other definitions**

Machine Learning is essentially a form of applied statistics with increased emphasis on the use of computers to statistically estimate complicated functions and a decreased emphasis on providing confidence intervals around these functions.

"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

<center>
<img src="{{site.url}}/assets/images/ml/ml.jpg"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Machine Learning</p> 
</center>

**Core of Machine Learning**

* Data
* Model
* Learning

*Data* is at the heart of machine learning, quality & clean data is like heaven on earth. But never expect your data to be as clean as given in Kaggle Competition. It is believed that larger the corpus of data, better is the generalization of the learning model. Less data results in poor generalization, thus causing the model to perform poorly on unseen data.

A *model* is a function, which is learnt based on data, complex data requires complex function. For instance, appraisals in workplace is a complex function because various factors plays a keyrole in your appraisal like your effort in team, quality deliverables, years of experience, cordial relationship with managers ;) etc. One cannot simply make linear function like increasing salary with increasing years of experience. Hence, the model learns these factors and its inter-relation and extracts the hidden patterns from data, these patterns are generalized over a function.

*Learning*, a model is said to learn from data. Learning here refers to identifying or recognizing the pattern in the data. A model is said to perform well, if it predicts accurately on unseen data. Unseen data is technically called as test data. When the model is training, it is trained on train data. The terms train and test data is explained in Machine Learning blog post II.

**Common terms in Machine Learning**

*Data*, is a set of observation we make about a system. For instance, for patient records, we keep note of their health habits, exisiting health condition, diseases XYZ exists or not etc.

*Predictor*, are a set of variables, which helps in describing a system. From above patient records, if we want to predict, if the patient has disease XYZ then our predictor variable will include health habits (like smoking, drinker etc), existing health condition like blood pressure, diabeitc etc. Predictor is also referred as independent variable.


<center> 
<img src="{{site.url}}/assets/images/ml/mlflow.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Machine Learning Flow</p> 
</center>

*Label* is a variable, it tells what the system is meant for. From patient records system, our label is the variable disease XYZ, here we are predicting if a patient has disease XYZ or not. Labels are also called target or dependent variable. Dependent as it depends on predictor variables.

*Model* is a machine learning algorithm, which is applied or trained on data like patients records to predict if patient has disease XYZ. Above example of patient has disease XYZ or not is a binary classification problem, where label variable takes Yes or No values, Binary classification model like Logistic Regression can be utilized here.

*Parameters* are tunable knobs in machine learning algorithm, there are different parameters available for each algorithm like *depth of tree* in Decision Tree Algorithm, *penalty* in Logistic Regression etc. These tunable parameters are *Hyperparameter*, which are set manually by user who is performing machine learning. Other set of parameters are learnt by machine learning algorithm like *coefficient of X's* in Linear regression, Weights in Deep Learning Algorithms etc.

*Dataset* is the data on which we apply machine learning algorithm after cleaning and structuring it in the form, such that it is consumable by ml algorithm. Consider we have 1000 patient records with target as *disease XYZ* with value 0 or 1. 0 represent no disease and 1 represent disease is present. To perform machine learning, we split dataset into multiple sets like train set, validation set and test set. 

For training, we use train set. For identifying the hyperparameter and parameters of the model, we use validation set. For check final performance of the model, we use test set.

*Training* is a process to learn the parameters of the machine learning algorithm. During this process, we keep track of training error, as the training continous to takes place, the training error reduces. Training is done on train set.

*Evaluating* is a process to learn hyperparameter and try out or experiment with the best possible hyperparameter for the learning algorithm. Evaluation can take place simultaneously with training. During evaluation, we keep track of validation error. Evaluation is done on validation set of dataset.

*Testing* is a process to confirm whether the model is generalized for unseen data or not. If the model performs poorly, we need to get back to training with different model or different parameter or hyperparameter or more data etc.

Note: Sometimes the training error will reduce but the validation error may fail to reduce. It happens because of bias issue in the model. More detail in next blog.

**Types of Machine Learning**

* **Supervised Learning**
* **Unsupervised Learning**
* **Reinforcement Learning**

**Supervised Learning**

   The Algorithm learns from labeled data. It identifies/recognizes the pattern from data with labels and associates those pattern to unlabeled data.
   For example: Data on house price based on various factors like area, rooms, lawn and other details, where we predict the value of the house. So our label is House Price.
   
**Unsupervised Learning**

   The Algorithm learns from unlabeled data.
   For example: Unstructured text documents where we cluster text or paragraphs based on word binding in the sentence using clustering algorithms, to group documents to particular topics.
   
**Reinforcement Learning**

   The Algorithm learns from performing some action and receiving rewards(+ve/-ve) for those actions. Here, the algorithms interact with an environment, so there is a feedback loop between the learning system and its experience. For instance: Self Driving Cars and Game Playing Agents.

<center>
<img src="{{site.url}}/assets/images/ml/ml_algo.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/><br>
<p>Figure 3: Machine Learning Algorithms</p> 
</center>

**Supervised Learning**

A dataset has feature ( X's ) and target ( y ). The target can be a discrete value or a continuous value based on which we can call the problem as Classification or Regression respectively.

**Classification** problem is classified as Binary or Multiclass classification. Either target value is 0/1 or else multiple values like Dogs, Person, Cat. It answers the yes or no question. It has categorical outcomes.

**Regression** problem is like predicting a value (Numerical). For instance, predicting the house price or stock value of a product. It answers how much questions. It has numeric outcomes.

Each dataset has features and target variable. The features are parameters which affects the target variable either directly or indirectly. Now, this relationship between X and y is built by a ML algorithm.

**Unsupervised Learning**

A dataset with no label is trained by unsupervised learning i.e. the patterns of such dataset is learned widely by using clustering techniques.

There are tons of material, which clearly explains different algorithms, to make it easier for reference and to cross check the understanding, I will just share the reference to each blog, instead of reinventing the wheel of writing those blogs.

* [**Naive Bayes**](https://www.kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html)
* [**Linear Regression**](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a)
* [**Logistic Regression**](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)
* [**Support Vector Machine**](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
* [**Decision Trees**](https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1)
* [**Neural Networks**](http://neuralnetworksanddeeplearning.com/chap1.html)
* [**Ensemble Model**](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)
* [**KMeans**](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
* [**Hierarchical Clustering**](https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/)
* [**DBSCAN Clustering**](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html)
* [**Gaussian Mixture Model**](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)
            
<p align='center'>. . . . .</p>
