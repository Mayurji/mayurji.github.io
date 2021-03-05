---
layout: machine-learning
title: A Deep Dive Into Class Imbalance
description: To avoid chaos, things must be balanced.
date:   2021-03-05 17:43:52 +0530
---
{% include mathjax.html %}

<center>
<img src="{{site.url}}/assets/images/ml/chris-liverani-dBI_My696Rk-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="100%" height=auto/><br>
<p>Photo by Chris Liverani on Unsplash</p>
</center>

In this blog post, we'll discuss about class imbalance problem in machine learning, what causes it and how to overcome it. From my experience of attending interviews, interviewers ask at least one scenario based question on **Class Imbalance**, widely being *how to handle class imbalance?*

![img](https://lh6.googleusercontent.com/vv7B7vPMZaSWQur7CiItyc3nKS_41t5kp_u2EqPEqQBk6AfAfnRP-Lmr2TZKHqDXu1XIgsxfRth7bRJeFwY0WzImWO-BN2GIYR_80AnwcV2NY0qgmEr30fm5leaxEqtLHoHnAtjc)
																					 [Andrew Ng: Bridging AI's Proof-of-Concept to Production Gap](https://www.youtube.com/watch?v=tsPuVAMaADY&ab_channel=StanfordHAI)

Class Imbalance is a classic problem one may face while working on a ml problem, in such case, each class is not equally represented by the number of data points. Whenever there is substantial difference in the number of samples in each class of the training data, the training model may suck with multiple characteristics as follows

* **Insufficient Signal** : the model performs poorly because of under representation of the minority class making it as a few-short learning on examples of the minority class. 
* **Delusional Accuracy** : consider two classes,  class A with 10k data points and class B with 1k data points, even if the model's prediction on class B is all wrong i.e. all class B is predicted as class A, making the model to have low loss and high accuracy. Even though the model is learning relatively little about the underlying structure of the data. Such a point may be a local loss minimum from which it may be difficult to extricate a given descent algorithm.
* **Horrible Outcomes** : For certain use cases, the error should be avoided at all cost. For instance, if a model fails to correctly identify a particularly rare but aggressive form of cancer. The model fails to learn or performs poorly on imbalanced class which is insufficient for building machine learning system.

### Common Instances of Class Imbalance

* Fraud Detection (Majority of Bank transactions are valid)
* Spam Detection (Majority of Emails are not spam)
* Disease Screening (Majority of people are Healthy)

### Causes of Class Imbalance

* **Sampling Bias** - It occurs when some samples of the population are systematically more likely to be selected in a **sample** than others. It is also called ascertainment **bias** in medical fields. **Sampling bias** limits the generalizability of findings because it is a threat to external validity, specifically population validity.
* **Domain Specific** - As mentioned above, in medical fields some samples of disease are rare to occur. Making **domain specific** class imbalance.
* **Labeling Errors** - A less common cause of class imbalance, it happens because of negligence while labeling.

### Solutions for Class Imbalance

**Resampling** : Simple and effective solution is to add more samples of minority class or remove the majority of class while modeling.

**Weight Balancing** : This solution's success is based on tasks. It requires us to model the weights in a way to make it sensitive or focus more rare classes by adjusting loss functions. The challenge with weight balancing is by how much should one define these adjustment of updating weights.

**Ensembles** : Ensemble models are more robust to imbalance class. Ensemble methods have shown better results than resampling solution. Ensemble classifiers are known to increase the accuracy of single classifiers by combining several of them and have been successfully applied to imbalanced data-sets

There are various ensemble based model 

* Boosting Based Ensemble Learning
* Bagging Based Ensemble Learning
* Hybrid Combined Ensembles

### Reference

**CS 329S: Machine Learning Systems Design** by [Chip Huyen](https://huyenchip.com), [Michael Cooper](https://michaeljohncooper.com/)
