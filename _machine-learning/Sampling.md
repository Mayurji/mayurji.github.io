---
layout: machine-learning
title: Sampling
description: Sampling is the art of selecting the best.
date:   2021-02-20 17:43:52 +0530
---
{% include mathjax.html %}

# Sampling

<center>
<img src="{{site.url}}/assets/images/ml/patrick-tomasso-Oaqk7qqNh_c-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="100%" height=auto/><br>
<p><span>Photo by <a href="https://unsplash.com/@impatrickt?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Patrick Tomasso</a> on <a href="https://unsplash.com/s/photos/books?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span></p>
</center>

In this blog post, we'll discuss about sampling and its related components. This topic is usually not given much importance compared to other fancy statistics terms such as bayes, frequency, distribution etc. In machine learning, sampling refers to the subset of the data from the population, where the population means every possible data available for the task, which is infinite because in real-world task, we are continuously collecting data for the model to train and validate on.

The topic of sampling is quite dry and requires special effort from the user reading it. My objective from this blog is to share the sampling topic in a more visual form.

<center>
<img src="{{site.url}}/assets/images/ml/Sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Sampling</p> 
</center>

## Types of Sampling

Two broad of sampling types

* Non-Probability
* Random or Probability Sampling

### Non-probability Sampling

The subset of we select here is based on the individual or the users, who are improbable. Since it based users or an individual, it comes with **bias**. There are various examples of Non-probability sampling as follows

* Convenient Sampling: Subset selection is done based on availability. Thus it is convenient.

<center>
<img src="{{site.url}}/assets/images/ml/convenient_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Convenient Sampling</p> 
</center>

* Snowball Sampling: Existing subset helps in finding the next set of subset. It rolls from subset of sample to other.

<center>
<img src="{{site.url}}/assets/images/ml/snowball_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Snowball Sampling</p> 
</center>

* Judgement Sampling: Subset selection based on experts advice, who judges the sample for the task.

<center>
<img src="{{site.url}}/assets/images/ml/judgement_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Judgement Sampling</p> 
</center>

* Quota Sampling: Subset selection happens in a predetermined order from the available quotas without any randomization.

<center>
<img src="{{site.url}}/assets/images/ml/quota_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 5: Quota Sampling</p> 
</center>

As ML candidate, we can think that having bias in selecting the data is bad for the model to train on but it is the best option available because the data for ML is available at convenience and is not generated randomly.

### Instances of Non-probability sampling

* Writing review on Amazon, people having access to internet or will to write is captured as data.
* Training Language models, the availability of wikipedia or reddit data, which is not all the possible data on internet, selection bias.
* Patients records for a disease present.

### Random or Probability Sampling

In random sampling, the selection of each sample from the population has equal probabilities. Consider a slot machine where each reel represent a set of items or number, lets assume there are 10 items in each reel, the chances of each item in each reel is 1/10th and the selection is done randomly.

<center>
<img src="{{site.url}}/assets/images/ml/random_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 6: Random Sampling</p> 
</center>

Implementing above sampling is quite easier but consider the situation, where we want to select 10% of all samples. And if there are some rare classes which occurs only in 0.1% of population, then there is a great chance to miss out the samples from that rare class completely. Model trained on such selection process may think that the rare class is not available.

### Stratified Sampling

Consider we have 10 different classes to predict from and each class has some set of data points to represent the class. Now when we perform random sampling like selecting 10% from each class, there is no possibility we can miss out a class completely. This is called stratified sampling.

<center>
<img src="{{site.url}}/assets/images/ml/stratified_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 7: Stratified Sampling</p> 
</center>

In stratified sampling, we group the samples based on some common link and select accordingly. For instance, we can have a divide the group based on gender and select 10% from each, keeping the variation in samples intact. Each group is called as Strata.

Downside of stratified sampling is when we cannot create subgroup from the available samples or when one sample can belong to multiple subgroups. This is challenging when you have a multilabel task and a sample can be both class A and class B. For example, in an entity classification task, a name might be both a PERSON and a LOCATION.

### Weighted Sampling

In weighted sampling, each sample is given a weight, which determines the probability of it being selected. For example, if you want a sample to be selected 30% of the time, give it weight 0.3. This method allows you to embed subject matter expertise. For example, if you know that more recent data is more valuable to your model, you can give recent data more weight.

<center>
<img src="{{site.url}}/assets/images/ml/weighted_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 8: Weighted Sampling</p> 
</center>

This also helps with the case when our available data comes from a different distribution compared to true data. For example, in our data, red samples account for only 25% and blue samples account for 75%, but we know that in the real world, red and blue have equal probability to happen, so we would give red samples weights three times the weights of blue samples.

### Importance Sampling

Importance sampling is a technique used for estimating properties of particular distribution, while having only samples generated from different distribution rather than that of interested distribution. 

From Wiki: Importance sampling is a [variance reduction](https://en.wikipedia.org/wiki/Variance_reduction) technique that can be used in the [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The idea behind importance sampling is that certain values of the input [random variables](https://en.wikipedia.org/wiki/Random_variables) in a [simulation](https://en.wikipedia.org/wiki/Simulation) have more impact on the parameter being estimated than others. If these "important" values are emphasized by sampling more frequently, then the [estimator](https://en.wikipedia.org/wiki/Estimator) variance can be reduced. Hence, the basic methodology in importance sampling is to choose a distribution which "encourages" the important values.

Challenge in Importance Sampling is that finding the distribution that encourages the important values. [Importance sampling has wide application in RL](https://jonathan-hui.medium.com/rl-importance-sampling-ebfb28b4a8c6)

For deep dive on Importance Sampling: [RL — Importance Sampling ](https://jonathan-hui.medium.com/rl-importance-sampling-ebfb28b4a8c6)(Jonathan Hui)

### Reservoir sampling

Motivation between Reservoir Sampling: Suppose we see a sequence of items, one at a time. We want to keep ten  items in memory, and we want them to be selected at random from the  sequence. If we know the total number of items *n* and can access the items arbitrarily, then the solution is easy: select 10 distinct indices *i* between 1 and *n* with equal probability, and keep the *i*-th elements. The problem is that we do not always know the exact *n* in advance.

**Key here is that we want to select k items at random with each having equal probability from an unknown sample size n.**

Imagine we have to sample k tweets from an incoming stream of tweets. You don’t know how many tweets there are but you know you can’t fit them all in memory, which means you don’t know the probability at which a tweet should be selected. You want to:

- ensure that every tweet has an equal probability of being selected,
- you can stop the algorithm at any time and get the desired samples.

One solution for that is reservoir sampling. The algorithm goes like this:

1. First k elements are put in the reservoir.
2. For each incoming ith element, generate a random number j between 1 and i
3. If 1 ≤ j ≤ k: replace jth in reservoir with ith

Each incoming ith element has (k / i) probability of being in the reservoir. You can also prove that each element in the reservoir has (k / i) probability of being there.

<center>
<img src="{{site.url}}/assets/images/ml/reservoir_sampling.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 9: Reservoir Sampling</p> 
</center>

For example, we would process a stream with three elements as follows:

1. Store the first element.
2. Store the second element with a probability of 1/2. Now both elements have equal probabilities of being in the reservoir.
3. Store the third element with a probability of 1/3. The previous two elements also have a final probability of (1/2)∗(2/3)=1/3 to be chosen.

### Reference

* **CS 329S: Machine Learning Systems Design** By [Chip Huyen](https://huyenchip.com), [Michael Cooper](https://michaeljohncooper.com/)



