---
layout: machine-learning
title: Machine Learning - III
description: "Data preparation and evaluation of machine learning models"
date:   2020-11-06 17:43:52 +0530
---
{% include mathjax.html %}
## Machine Learning Concepts II

<center>
<img src="{{site.url}}/assets/images/ml/arseny-togulev-MECKPoKJYjM-unsplash.jpg"  style="zoom: 5%  background-color:#DCDCDC;"  width="100%" height=auto/><br>    
<p><span>Photo by <a href="https://unsplash.com/@tetrakiss?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Arseny Togulev</a> on <a href="https://unsplash.com/s/photos/machine-learning?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span></p>
</center>

### Creating Datasets and Evaluation Metrics

Before applying ML Algorithm, we should check the dataset and split it for modeling for ML. We should split our dataset into training, testing and validation set. It helps in understanding certain factors of ML model like bias and variance i.e. also termed as Underfitting and Overfitting.

#### Training Set

Training Set: Before Big Data came into the picture of analytics, training data made around 70-75% data of total data. But with millions of records or data, training data occupies 95% of the total data. We model our algorithm on training set.


#### Validation and Testing Set

Validation Set: A set of examples used to tune the parameters [i.e., architecture, not weights] of a classifier, for example to choose the number of hidden units in a neural network. Before Big Data era, validation set occupied 15-12.5% data, but in Big Data times, it occupies 2.5%.

Test Set: A set of examples used only to assess the performance [generalization] of a fully specified classifier. Before Big Data era, validation set occupied 15-12.5% data, but in Big Data times, it occupies 2.5%. 

The validation and testing set are also called as hold-out sets.

#### Factors for selecting an algorithm

* Explainability
* In-memory Vs Out-of-Memory
* Number of Features and examples
* Categorical Vs Numerical Features
* Nonlinearity of the data
* Training speed
* Predicton speed

<center>
<img src="{{site.url}}/assets/images/ml/ml_map.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Algorithm Selection</p> 
</center>

#### Evaluation Metrics

**Confusion Matrix**

Consider a model trained two classify Cat and Dog images. And after training, we are testing the model on 100 random images of Dogs and Cats with 50 each and get an accuracy of 85%. It means that the model has misclassified 15 images. Now let's consider that out of 15 images 10 Dog images were misclassified as Cat and 5 Cat images are misclassified as Dog.

<center>
<img src="{{site.url}}/assets/images/ml/Confusion_matrix_2.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Confusion Matrix</p> 
</center>

From above image, we can see our True Positive is Cat's Image and True Negative is Dog's Image. And a Cat getting misclassified is called as False Negative and when a Dog is misclassified is called as False Positive. In essence, it consider Positive as Cat and Negative as Dog.

<center>
<img src="{{site.url}}/assets/images/ml/confusion_matrix.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Confusion Matrix</p> 
</center>

**Precision and Recall**

**Precision** can be seen as a measure of exactness or quality, whereas **recall** is a measure of completeness or quantity. In simple terms, high precision means that an algorithm returned substantially more relevant results than irrelevant ones, while high recall means that an algorithm returned most of the relevant results.

<center>
<img src="{{site.url}}/assets/images/ml/precision.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Precision</p> 
</center>

**Precision** informs us, how well the model predicts the positive class. It is also known as PPV (Positive Predictive Value).

<p>$$Precision = {True Positive \over (True Positive + False Positive)}$$</p>

**Recall** informs us, how well the model predicts the right class for a label. It is also known as Sensitivity.

<center>
<img src="{{site.url}}/assets/images/ml/recall.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 5: Recall</p> 
</center>

<p>$$Recall = {True Positive \over (True Positive + False Negative)}$$</p>

#### Sensitivity and Specificity

**Sensitivity** also known as recall or true positive rate, measures the proportion of actual positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition).

<p>$$Sensitivity = {True Positive \over (True Positive + False Negative)}$$</p>

**Specificity** also known as true negative rate, measures the proportion of actual negatives that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).

<center>
<img src="{{site.url}}/assets/images/ml/specificity.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 6: Specificity</p> 
</center>

<p>$$Specificity = {True Negative \over (True Negative + False Positive)}$$</p>

**Drawback of using only Precision and Recall**

Consider having built a model to classify whether a patient suffers from cancer or not from a sample of 1000000 (1 million). We know that 100 in 1 million suffer from cancer. After training, we test the model and it misclassifies the 100 cancer as 90 with no cancer (False Negative) and 10 with (True Positive). It correctly classifies Non-cancer patient as no cancer.

Lets find accuracy of such model, accuracy = (999900+10) / 1000000 = 99.99%. Wow what a great model, wrong such sensitive usecase we cannot relie on accuracy.

Lets calculate Precision, P = 10/(10+0) = 100%.

Lets calculate Recall, R = 10/(10 + 90) = 10%.

Clearly we can see that the model failed to classify the cancer patient, and these metrics gets dominated by class with large numbers which is otherwise referred as **Imbalanced class problem**. Here, the number of non cancer class are 999900 and cancer class are 100. So its vital to check if these metric really telling story we want to hear.

#### F1-Score & F1-Beta Score

**F1 Score**

In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall).

<center>
<img src="{{site.url}}/assets/images/ml/f1-score.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 7: F1-score</p> 
</center>

**F1 Beta Score**

As we have seen in drawback of using metrics like precision and recall alone would result it false understanding of the model output. Based on usecase, we can give importance to precision and recall with help of **F1 Beta score**.

Find this below snippet from wikipedia:

<center>
<img src="{{site.url}}/assets/images/ml/f1_beta_score.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 8: F1 Beta Score</p> 
</center>

**[ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)**

ROC- Receiver Operating Characteristics, its a plot between True Positive rate (Sensitivity) vs False Positive Rate (1 - specificity). Its a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

<center>
<img src="{{site.url}}/assets/images/ml/roc.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 9: Area under the ROC curve</p> 
</center>

ROC curves are frequently used to show in a graphical way the connection/trade-off between clinical sensitivity and specificity for every possible cut-off for a test or a combination of tests. In addition the area under the ROC curve gives an idea about the benefit of using the test(s) in question.

<p>$$TPR = {TP \over (TP + FN)}$$</p>

<p>$$FPR = {FP \over (FP + TN)}$$</p>

It’s easy to see that if the threshold is 0, all our predictions will be positive, so both TPR and FPR will be 1 (the upper right corner). On
the other hand, if the threshold is 1, then no positive prediction will be made, both TPR and FPR will be 0 which corresponds to the lower left corner.

The higher the area under the ROC curve (AUC), the better the classifier. A classifier with an AUC higher than 0.5 is better than a random classifier. If AUC is lower than 0.5, then something is wrong with your model. A perfect classifier would have an AUC of 1. Usually, if our model behaves well, we obtain a good classifier by selecting the value of the threshold that gives TPR close to 1 while keeping FPR near 0.

#### Reference

[ROC Curve- what and How ?](https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used)
     
  <p align='center'>. . . . .</p>
  
#### Selecting Hyperparameters

While training the model, the model learns parameters W. But there are other set of parameters called Hyperparameter, which are tuned manually by user/ML developer. For instance, there are models such Decision Tree, where we tune the **depth of the tree** or Support Vector Machine where we tune penalty parameter **C** etc.

To select an optimal hyperparameter, we can use basic technique like **Grid Search**. In Grid search, we assign set of values to hyperparameters and perform modeling on each hyperparameter value, it is expensive if there many hyperparameters to tune. 

Let’s say you train an SVM and you have two hyperparameters to tune: the penalty parameter C (a positive real number) and the kernel (either “linear” or “rbf”).

If it’s the first time you are working with this dataset, you don’t know what is the possible range of values for C. The most common trick is to use a logarithmic scale. For example, for C you can try the following values: [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]. In this case you have 14 combinations of hyperparameters to try: [(0.001, “linear”), (0.01, “linear”), (0.1, “linear”), (1.0, “linear”), (10, “linear”), (100, “linear”), (1000, “linear”), (0.001, “rbf”), (0.01, “rbf”),
(0.1, “rbf”), (1.0, “rbf”), (10, “rbf”), (100, “rbf”), (1000, “rbf”)].

You use the training set and train 14 models, one for each combination of hyperparameters. Then you assess the performance of each model on the validation data using one of the metrics we discussed in the previous section (or some other metric that matters to you). Finally, you keep the model that performs the best according to the metric.

Other techniques to find better hyperparameters
* Random Search
* Bayesian Hyperparameter
* Gradient based techniques
* Evolutionary optimization techniques

#### Cross Validation

If we have less data or poor validation set for evaluating the model, then we can use cross-validation technique to find hyperparameter. Note, validation set is used for tuning model with help of better hyperparameter and reduced validation error.

To perform cross-validation, Cross-validation works like follows. First, you fix the values of the hyperparameters you want to evaluate. Then you split your training set into several subsets of the same size. Each subset is called a fold. Typically, five-fold cross-validation is used in practice. With five-fold
cross-validation, you randomly split your training data into five folds: $${F_1 , F_2 , . . . , F_5 }$$. Each $$F_k , k = 1, . . . , 5$$ contains 20% of your training data. Then you train five models as follows.

To train the first model, $$f_1$$ , you use all examples from folds $$F_2 , F_3 , F_4 , and F_5$$ as the training set and the examples from $$F_1$$ as the validation set. To train the second model, $$f_2$$ , you use the examples from folds $$F_1 , F_3, F_4 , and F_5$$ to train and the examples from $$F_2$$ as the validation set. You continue building models iteratively like this and compute the value of the metric of interest on each validation set, from $$F_1 to F_5$$. Then you average the five values of the metric to get the final value.

#### Reference

The Hundred-Page Machine Learning by Andriy Burkov
