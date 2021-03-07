---
layout: machine-learning
title: How To Mitigate Class Imbalance
description: To avoid chaos, things must be balanced.
date:   2021-03-06 17:43:52 +0530
---
{% include mathjax.html %}

<center>
<img src="{{site.url}}/assets/images/ml/chris-liverani-dBI_My696Rk-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="100%" height=auto/><br>
<p>Photo by Chris Liverani on Unsplash</p>
</center>

In the previous blog post, I've discussed about [what and why of class imbalance](mayurji.github.io/machine-learning/class_imbalance), and I have briefly touched upon the solutions for class imbalance. Now, we'll deep dive into solving class imbalance problem with proposed solution from previous blog post.

* **Resampling**
* **Weight Balancing Loss** 
* **Ensemble Models**

### Resampling

<center>
<img src="{{site.url}}/assets/images/ml/over_under_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/>	
<p>Resampling strategies for imbalanced datasets: Kaggle</p>
</center>

The idea is to rebalance the class distribution by resampling the data space. Resampling is done by either adding more minority samples or removing majority samples. It avoids the modification of learning algorithm by trying to decrease the effect caused by data imbalance with a preprocessing step, so it is usually more versatile than the other imbalance learning methods. The algorithms learning about imbalance data is called as **Imbalance Learning**.

* Oversampling - Adding more minority samples.
* Undersampling - Removing majority samples.

#### Oversampling

In this resampling technique, the minority samples are replicated randomly to create balance class distribution. This method might cause overfitting to minority samples, since it makes exact copies of existing samples.

<center>
<img src="{{site.url}}/assets/images/ml/smote_oversampling.png"  style="zoom: 5%  background-color:#DCDCDC;" width="60%" height=auto/>	
<p>Imbalanced Data: Datacamp</p>
</center>

SMOTE (Synthetic Minority Over-sampling Technique) is a popular oversampling technique. The main idea is to create new instance of minority samples by interpolating several minority class instances that lie together. SMOTE uses k-nearest neighbors to create synthetic examples of the minority class. SMOTE can avoid the overfitting problem. A major disadvantage of SMOTE is that While generating synthetic examples, SMOTE does not take into consideration neighboring examples can be from other classes. This can increase the overlapping of classes (making wrong labeling) and can introduce additional noise.

To overcome SMOTE issues, there is modified SMOTE calles MSMOTE. The main idea of this algorithm is to divide the instances of the minority class into three groups, **safe, border and latent noise instances**, by the calculation of distances among all samples. We create boundaries for each samples in minority class to create the new instance of minority class thus helping in reducing the mislabelled samples. 

#### Undersampling

In this resampling technique, the majority class samples are randomly dropped to create a balance class distribution. The downside is we might lose some potentially important information about the majority class.

<center>
<img src="{{site.url}}/assets/images/ml/under_sampling_tomek_links.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/>	
<p>Resampling strategies for imbalanced datasets: Kaggle</p>
</center>

A popular technique for Undersampling is Tomek Links. The main idea is to **find pairs of closest samples of opposite classes and drop the class which belongs to majority samples.** While this makes the decision boundary more clear and arguably helps models learn the boundary better, it may make the model less robust (by removing some of the subtleties of the true decision boundary).

### Weight Balancing Loss

When a model is trained, the model learns the parameters for each class. For an imbalanced class, the weightage or more weights is assigned to majority class during the loss reduction process. The key idea behind weight-balancing loss is that if we give the loss resulting from the wrong prediction on a data sample higher weight, we’ll incentivize the ML model to focus more on learning that sample correctly.

Two methods for weight balancing loss

* Biasing toward rare classes
* Biasing toward difficult samples

**Biasing toward rare classes** : As mentioned above, the model assigns more weights to majority class, it can also be termed as that, it'll bias toward majority class and making wrong prediction on minority class. What if we punish the model for making wrong predictions on minority classes to correct this bias?

A general loss function for over a set of samples:

<p>
$$
L(X_i, \ \theta) =  \ \sum_i \ L(x_i ; \ \theta)
$$
</p>

A simple weighted loss function can written as follows:

The weight of the class is inversely proportional to number of samples in that class, which makes class with less samples to have more weights.
<p>
$$
L(X_i, \ \theta) =  \ \sum_i \ W_{y_i} \ L(x_i ; \ \theta) \\
W_c =  {N  \over number \ of \ samples \ of \ class \ C} \\
N - \ total \ number \ of \ samples
$$
</p>

A more sophisticated version of this loss can take in account the overlapping among existing samples, such as [Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) (Cui et al., CVPR 2019).

**Biasing toward difficult samples** : When a model is trained, the model will predict with great confidence on certain samples and may perform poorly or predict with less confidence on certain samples. The idea here is to make the model to be incentivize towards samples which are difficult to learn. What if, we can adjust the weights to high, whenever a samples has lower probability of being right. Focal Loss is a loss function, which does exactly what we have described before.

<center>
<img src="{{site.url}}/assets/images/ml/focal_loss.png"  style="zoom: 5%  background-color:#DCDCDC;" width="60%" height=auto/>	
<p>Focal Loss: arxiv.org/pdf/1708.02002.pdf</p>
</center>


### Ensemble Models

We'll discuss about two ensemble models bagging and boosting for class imbalance. First we'll understand what these models are and how it can help in class imbalance.

#### Bagging :

How it works? Instead of training a model on entire dataset at one time, we create a multiple subsample of the dataset with replacement, making different dataset, called as bootstraps. And then we train a model on each of these bootstraps. Sampling with replacement ensures each bootstrap is independent from its peers.

<center>
<img src="{{site.url}}/assets/images/ml/bagging.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/>	
<p>Bagging - Sirakorn: Wikipedia</p>
</center>


If the problem is **classification**, the final prediction is decided by the majority vote of all models. For example, if 10 classifiers vote SPAM and 6 models vote NOT SPAM, the final prediction is SPAM.

If the problem is **regression**, the final prediction is the average of all models’ predictions.

A random forest is an example of bagging. A random forest is a collection of decision trees constructed by both bagging and feature randomness, each tree can pick only from a random subset of features to use. Due to its ensembling nature, random forests correct for decision trees’ overfitting to their training set.

* Also known as Bootstrap Aggregating
* It improves stability and accuracy
* It reduces variance and avoid overfitting

**OverBagging** is a method used for class imbalance, it uses bagging and data preprocessing.  First, the cardinality of the minority class is increased by replication of original samples using Random Oversampling, while the samples in the majority class can be all considered in each bag or can be resampled to increase the diversity. This method outperforms original bagging in dealing with binary imbalanced data problems.

There are other approaches to bagging for class imbalance such as **SMOTEBagging, UnderBagging** etc.

#### Boosting :

Boosting is a family of iterative ensemble algorithms that convert weak learners to strong ones. Each learner in this ensemble is trained on the same set of samples but the samples are weighted differently among iterations. Thus, future weak learners focus more on the examples that previous weak learners misclassified. 

<center>
<img src="{{site.url}}/assets/images/ml/boosting.png"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/>	
<p>Boosting - Sirakorn: Wikipedia</p>
</center>

How it works? 

1. You start by training the first weak classifier on the original dataset.
2. Samples are reweighted based on how well the first classifier classifies them, e.g. misclassified samples are given higher weight.
3. Train the second classifier on this reweighted dataset. Your ensemble now consists of the first and the second classifiers.
4. Samples are weighted based on how well the ensemble classifies them.
5. Train the third classifier on this reweighted dataset. Add the third classifier to the ensemble.
6. Repeat for as many iterations as needed.
7. Form the final strong classifier as a weighted combination of the existing classifiers -- classifiers with smaller training errors have higher weights.

An example of a boosting algorithm is Gradient Boosting Machine which produces a prediction model typically from weak decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

**SMOTEBoost** is an oversampling method based on the SMOTE algorithm. SMOTE uses k-nearest neighbors to create synthetic examples of the minority class. SMOTEBoost then injects the SMOTE method at each boosting iteration. The advantage of this approach is that while standard boosting gives equal weights to all misclassified data, SMOTE gives more examples of the minority class at each boosting step. Thus helping the boosting algorithm to give more weight to the class, less represented.

Similarly, **RUSBoost**(Random UnderSampling Boost) achieves the same goal by  performing random undersampling (RUS) at each boosting iteration instead of SMOTE.

All the algorithms based approach for class imbalance performs well only when its included with other sampling approaches.

### Reference

**CS 329S: Machine Learning Systems Design** by [Chip Huyen](https://huyenchip.com), [Michael Cooper](https://michaeljohncooper.com/)

[**SMOTEBoost**](https://medium.com/urbint-engineering/using-smoteboost-and-rusboost-to-deal-with-class-imbalance-c18f8bf5b805)

