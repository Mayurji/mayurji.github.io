---
layout: machine-learning
title: Unsupervised Learning
description: Clustering Algorithms
date:   2021-04-05 17:43:52 +0530
---
{% include mathjax.html %}

## Unsupervised Learning

<center>
<img src="{{site.url}}/assets/images/ml/omar-flores-lQT_bOWtysE-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;" width="80%" height=auto/><br>
<p>Photo by <a href="https://unsplash.com/@omarg247?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Omar Flores</a> on <a href="https://unsplash.com/s/photos/pattern?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a></p>
</center>

In previous blog post on [supervised learning](https://mayurji.github.io/machine-learning/Supervised%20Learning), we have seen that each observed data has a label attached to it, making it easy to train a model. However, in unsupervised learning, **the algorithm finds the hidden patterns in unlabeled data.** A popular technique in unsupervised learning is **Clustering Algorithms**.
<p>
$$
x_1 , \ x_2, \ .....x_m \\
and \ there \ is \ no \ label \ y.
$$
</p>

**Clustering Algorithms** are a set of algorithms, which clusters data points based on similarity metric either measured in terms of distance or using probability to categorize the data point etc. In simple terms, we can think of clustering as a way to group the data points that are similar in nature from group of data points which are dissimilar.

**Latent variables** plays a key role in understanding and finding pattern in unlabeled data. Latent variables are not observed variables but are rather inferred. Since we have observed variables (X's), we can utilize observed variables to identify latent variables using mathematical model. Mathematical models that aim to explain observed variables in terms of latent variables are called [latent variable models](https://en.wikipedia.org/wiki/Latent_variable_model). 

### Clustering Algorithms

#### Expectation-Maximization Algorithm

It is an efficient iterative method to estimate the parameters of the latent variables through maximum-likelihood estimation. In each iteration, the algorithm performs two alternate steps, first performing an expectation (E) step, which creates a function for the expectation of the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) evaluated using the current estimate for the parameters, and a  maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the *E* step. These parameter-estimates are then used to determine the distribution of the latent variables in the next E step.


<center>
<img src="{{site.url}}/assets/images/ml/EM_Clustering_of_Old_Faithful_data.gif"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Unsupervised Learning - EM Algorithm</p> 
</center>

**E-Step:** Evaluate the posterior probability $$Q_i (z^{(i)})$$ that each data point $$ x^{(i)} $$ came from a particular cluster $$z^{(i)}$$ as follows

<p>
$$
Q_i \ (z^{(i)}) = P(z^{(i)} | x^{(i)}; \theta)
$$
</p>

**M-Step:** Use the posterior probability $$ Q_i (z^{(i)}) $$ as cluster specific weights on data points x^{(i)} to separately re-estimate each cluster model as follows 
<p>
$$
\theta_i = argmax_\theta \sum_{i} \int_{z^{(i)}}  Q_i (z^{(i)}) \ log \ {P(x^{(i)}, \ z^{(i)}; \theta \over Q_i(z^{(i)})}. dz^{(i)}
$$
</p>

<center>
<img src="{{site.url}}/assets/images/ml/em_alog.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Unsupervised Learning - EM Algorithm</p> 
</center>

**Simple Understanding**

* Generating Latent variables z using conditional probability over observed variable x.
* Estimating parameters θ at step t using z, x and previous θ. θ is initialized randomly.
* Re-estimating θ for each specific cluster group.

#### K-Means Clustering Algorithm

K-Means is the most popular clustering algorithm used in unsupervised learning. The algorithm tries to partition the n observations into K-Cluster in which each observation belongs to the cluster with the nearest mean (Cluster centers or centroid).

<center>
<img src="{{site.url}}/assets/images/ml/kmean_algo.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Unsupervised Learning - K-Means</p> 
</center>

At first, we randomly intialize the cluster centroids, $$ \mu_1, \mu_2, ..., \mu_k \ \in R^n $$, the k-means algorithm repeats the following steps until convergence

<p>
$$
c^{(i)} \ = \ {arg \ min}_j || \ x^{(i)} - \mu_j ||^2 \\
$$
$$
\mu_j \ = \ {\sum_{i=1}^{m} 1_{c^{(i)}=j}x^{(i)} \over \sum_{i=1}^m 1_{c^{(i)}=j}}
$$

$$
c^{(i)} -  cluster\ of\ data\ point\ i. \\
\mu_j - center\ of\ cluster\ j.
$$

</p>


<center>
<img src="{{site.url}}/assets/images/ml/K-means_convergence.gif"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Unsupervised Learning - K-Means Algorithm</p> 
</center>

##### Simple Understanding

1. Pick the number of clusters k.
2. Select k random points from the data points as centroids.
3. Assign each data point to its nearest cluster centroid.
4. Calculate the centroids based on newly formed cluster.
5. Repeat step 3 and 4, until stopping criteria

##### Stopping Criteria

1. Centroids of newly formed clusters do not change
2. Points remain in the same cluster
3. Maximum number of iterations are reached

### Hierarchical Clustering

In this method, the cluster are build in hierarchy. There are two ways to build hierarchy 

**Agglomerative Clustering:** It follows bottom-up hierarchy, each observation considers itself as a cluster and combine with other observation as they move up the hierarchy. The observations which are grouped together at the top of the hierarchy thus form the clusters.

**Divisive Clustering:** It follows top-down hierarchy, all observation come under one cluster and gets split as they move down the hierarchy. The observations which are combined together till the last forms a clusters.

In order to decide which clusters should be combined (for  agglomerative), or where a cluster should be split (for divisive), a measure of dissimilarity between sets of observations is required. Now, to separate/split each cluster or hierarchy from one another different metrics and linkage criterions are used. 

**Metrics** measures the distance between the pair of observation. The choice of an appropriate metric will influence the shape of the  clusters, as some elements may be relatively closer to one another under one metric than another. 

For example, in two dimensions, under the  Manhattan distance metric, the distance between the origin (0,0) and  (0.5, 0.5) is the same as the distance between the origin and (0, 1),  while under the Euclidean distance metric the latter is strictly  greater.

<center>
<img src="{{site.url}}/assets/images/ml/distance_metric.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Agglomerative Clustering - Distance Metrics</p> 
</center>

The linkage criterion determines the distance between sets of  observations as a function of the pairwise distances between observation and is applied in agglomerative clustering to identify the merge strategy. There different linkages

* Ward Linkage -  minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
* Complete Linkage - minimizes the maximum distance between observations of pairs of clusters.
* Single Linkage - minimizes the distance between the closest observations of pairs of clusters.
* Average Linkage - minimizes the average of the distances between all observations of pairs of clusters.

<center>
<img src="{{site.url}}/assets/images/ml/hierarchical_clustering.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Unsupervised Learning - Agglomerative Clustering</p> 
</center>

### Evaluation Strategy For Clustering Algorithm

Evaluation of a clustering algorithm is tricky because the algorithm has no metric validate against unlike supervised learning. Anyway, there are few metric which can tell us how well our cluster are formed either data points are lying at the edge of cluster or in between two cluster in a confused state or clusters are separate from each other with neat boundaries between them. Couple of metrics used to asses the clustering algorithms are as follows

* **Silhouette Coefficient** - By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:
<p>
  $$
  s = {b - a \over max(a, b)}
  $$
</p>

  **Simple Understanding**

  The Silhouette Coefficient is a measure of how well samples are clustered with samples that are similar to themselves. Clustering models with a high Silhouette Coefficient are said to be dense, where samples in the same cluster are similar to each other, and well separated, where samples in different clusters are not very similar to each other.

  The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.

* **Calinksi-Harabaz Index** - It is also known as Variance Ratio Criterion. It is the ratio between how all the data-points in the all clusters are dispersed within cluster and how all the clusters are dispersed between themselves.

By noting k the number of clusters, B_k and W_k the between and within-clustering dispersion matrices respectively defined as
<p>
$$
B_k = \sum_{i=1}^k n_{c^{(i)}} \ (\mu_{c^{(i)}} - \mu) (\mu_{c^{(i)}} - \mu)^T \\

W_k = \sum_{i=1}^k  \ ({x^{(i)}} - \mu_{c^{(i)}}) (x^{(i)} - \mu_{c^{(i)}})^T
$$
</p>
the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well                    separated the clusters are. It is defined as follows:
<p>
$$
s_k = { Tr(B_k) \over Tr(W_k)} * {n \ - k \over k \ - 1} \\

n - Number\ of\ Data\ points \\
k - Number\ of\ Clusters \\

$$
</p>

*B*(*k*) has *k*−1 degrees of freedom, while *W*(*k*) has *n*−*k* degrees of freedom. 

As *k* grows, if the clusters were all actually just from the same population, *B* should be proportional to *k*−1 and *W* should be proportional to *n*−*k*. 

So if we scale for those degrees of freedom, it puts them more on the same scale (apart, of course, from the effectiveness of the clustering, which is what the index attempts to measure).

#### Further reading

* [EM-Algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
* [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)
* [Posterior Probability](https://en.wikipedia.org/wiki/Posterior_probability)
* [MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
* [Likelihood Function](https://en.wikipedia.org/wiki/Likelihood_function)
* [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
* [Hierarchical Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)
* [Linkage](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster)
* CS229 Standford Class
