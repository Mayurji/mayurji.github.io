---
layout: machine-learning
title: A Deep Dive Into Class Imbalance
description: To avoid chaos things must be balanced.
date:   2021-03-05 17:43:52 +0530
---
{% include mathjax.html %}

In this blog post, we'll discuss class imbalance problem, what causes it, and how to overcome it. From my experience of attending interviews, interviewers ask at least one scenario-based question on **Class Imbalance** and widely being *how to handle class imbalance?*

<center>
<img src="{{site.url}}/assets/images/ml/class_imbalance.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/> 
<p>Andrew Ng: Bridging AI's Proof-of-Concept to Production Gap</p>
</center>

Class Imbalance is a classic problem one may face while working on an ml problem, in such a case, each class is not equally represented by the number of data points. Whenever there is a substantial difference in the number of samples in each class of the training data, the training model may suck with multiple characteristics as follows

**Insufficient Signal**: 

The model performs poorly because of the under-representation of the minority class making it few-short learning on examples of the minority class.

**Delusional Accuracy**: 

Consider two classes, class A with 10k data points and class B with 1k data points, even if the model's prediction on class B is all wrong i.e. all class B is predicted as class A, making the model have low loss and high accuracy. 

The model is learning relatively little about the underlying structure of the data. It leads to local minimum from which it may be difficult to extricate a given descent algorithm.

**Horrible Outcomes**: 

For certain use cases, the error should be avoided at all costs. For instance, a model fails to correctly identify a rare but aggressive form of cancer. Then the model fails to learn and performs poorly on imbalanced classes which is insufficient for building a machine learning system.

### Common Instances of Class Imbalance

* Fraud Detection (Majority of Bank transactions are valid)
* Spam Detection (Majority of Emails are not spam)
* Disease Screening (Majority of people are Healthy)

### Causes of Class Imbalance

**Sampling Bias**

It occurs when some samples of the population are systematically more likely to be selected in a **sample** than others. It is also called ascertainment **bias** in medical fields. **Sampling bias** limits the generalizability of findings because it is a threat to external validity, specifically population validity.

**Domain-Specific**

As mentioned above, in medical fields some samples of disease are rare to occur. Making **domain-specific** class imbalance.

**Labeling Errors**

A less common cause of class imbalance, it happens because of negligence while labeling.

### Solutions for Class Imbalance

**Resampling**

Simple and effective solution is to add more samples of a minority class or remove the majority of class while modeling.

**Weight Balancing**

This solution's success is based on the tasks. It requires us to model the weights in a way to make them sensitive or focus more on rare classes by adjusting the loss function. The challenge with weight balancing is by how much the weights should be adjusted.

**Ensembles**

Ensemble models are more robust to the imbalance class. Ensemble methods have shown better results than resampling solutions. Ensemble classifiers are known to increase the accuracy of single classifiers by combining several of them and have been successfully applied to imbalanced data-sets

Various ensemble-based models 

* Boosting Based Ensemble Learning
* Bagging Based Ensemble Learning
* Hybrid Combined Ensembles

If you've liked this post, please don't forget to subscribe to the newsletter.

### Reference

**CS 329S: Machine Learning Systems Design** by [Chip Huyen](https://huyenchip.com), [Michael Cooper](https://michaeljohncooper.com/)
