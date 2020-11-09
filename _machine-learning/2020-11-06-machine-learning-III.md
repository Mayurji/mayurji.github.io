---
layout: machine-learning
title: Machine Learning - III
description: "Data preparation and evaluation of machine learning models"
date:   2020-11-06 17:43:52 +0530
---
{% include mathjax.html %}



**CREATING DATASET AND EVALUATION METRICS**

Before applying ML Algorithm, we should check the dataset and split it for modelling for ML. We should split our dataset into training, testing and validation set. It helps in understanding certain factors of ML model like bias and variance i.e. also termed as Underfitting and Overfitting.

**TRAINING SET**

Training Set: Before Big Data came into the picture of analytics, training data made around 70-75% data of total data. But with millions of records or data, training data occupies 95% of the total data. We model our algorithm on training set.

**VALIDATION AND TESTING SET**

Validation Set: A set of examples used to tune the parameters [i.e., architecture, not weights] of a classifier, for example to choose the number of hidden units in a neural network. Before Big Data era, validation set occupied 15-12.5% data, but in Big Data times, it occupies 2.5%.

Test Set: A set of examples used only to assess the performance [generalization] of a fully specified classifier. Before Big Data era, validation set occupied 15-12.5% data, but in Big Data times, it occupies 2.5%. 

The validation and testing set are also called as hold-out sets.

<p align='center'>. . . . .</p>

**FACTORS FOR SELECTING ALGORITHM**

* Explainability
* In-memory Vs Out-Memory
* Number of Features and examples
* Categorical Vs Numerical Features
* Nonlinearity of the data
* Training speed
* Predicton speed

<center>
<img src="{{site.url}}/assets/images/ml/ml_map.png"  style="zoom: 5%  background-color:#DCDCDC;" width="1000" height="600"/><br>
<p><b>Figure 4:</b> Algorithm Selection</p> 
</center>

**EVALUATION METRICS IN MACHINE LEARNING**

**CONFUSION MATRIX**

Consider a model trained two classify Cat and Dog images. And after training, we are testing the model on 100 random images of Dogs and Cats with 50 each and get an accuracy of 85%. It means that the model has misclassified 15 images. Now let's consider that out of 15 images 10 Dog images were misclassified as Cat and 5 Cat images are misclassified as Dog.

<center>
<img src="{{site.url}}/assets/images/ml/Confusion_matrix_2.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 5:</b> Confusion Matrix</p> 
</center>

From above image, we can see our True Positive is Cat's Image and True Negative is Dog's Image. And a Cat getting misclassified is called as False Negative and when a Dog is misclassified is called as False Positive. In essence, it consider Positive as Cat and Negative as Dog.

<center>
<img src="{{site.url}}/assets/images/ml/confusion_matrix.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 6:</b> Confusion Matrix</p> 
</center>

**PRECISION AND RECALL**

**Precision** can be seen as a measure of exactness or quality, whereas **recall** is a measure of completeness or quantity. In simple terms, high precision means that an algorithm returned substantially more relevant results than irrelevant ones, while high recall means that an algorithm returned most of the relevant results.

<center>
<img src="{{site.url}}/assets/images/ml/precision.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 7:</b> Precision</p> 
</center>

**Precision** informs us, how well the model predicts the positive class. It is also known as PPV (Positive Predictive Value).

<p>$$Precision = {True Positive \over (True Positive + False Positive)}$$</p>

**Recall** informs us, how well the model predicts the right class for a label. It is also known as Sensitivity.

<center>
<img src="{{site.url}}/assets/images/ml/recall.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 8:</b> Recall</p> 
</center>

<p>$$Recall = {True Positive \over (True Positive + False Negative)}$$</p>

**SENSITIVITY AND SPECIFICITY**

**Sensitivity** also known as recall or true positive rate, measures the proportion of actual positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition).

<p>$$Sensitivity = {True Positive \over (True Positive + False Negative)}$$</p>

**Specificity** also known as true negative rate, measures the proportion of actual negatives that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).

<center>
<img src="{{site.url}}/assets/images/ml/specificity.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 9:</b> Specificity</p> 
</center>

<p>$$Specificity = {True Negative \over (True Negative + False Positive)}$$</p>

**DRAWBACK OF USING ONLY PRECISION AND RECALL**

Consider having built a model to classify whether a patient suffers from cancer or not from a sample of 1000000 (1 million). We know that 100 in 1 million suffer from cancer. After training, we test the model and it misclassifies the 100 cancer as 90 with no cancer (False Negative) and 10 with (True Positive). It correctly classifies Non-cancer patient as no cancer.

Lets find accuracy of such model, accuracy = (999900+10) / 1000000 = 99.99%. Wow what a great model, wrong such sensitive usecase we cannot relie on accuracy.

Lets calculate Precision, P = 10/(10+0) = 100%.

Lets calculate Recall, R = 10/(10 + 90) = 10%.

Clearly we can see that the model failed to classify the cancer patient, and these metrics gets dominated by class with large numbers which is otherwise referred as **Imbalanced class problem**. Here, the number of non cancer class are 999900 and cancer class are 100. So its vital to check if these metric really telling story we want to hear.

**F1-SCORE AND F1-BETA SCORE**

**F1 Score**

In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall).

<center>
<img src="{{site.url}}/assets/images/ml/f1-score.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 10:</b> F1-score</p> 
</center>

**F1 Beta Score**

As we have seen in drawback of using metrics like precision and recall alone would result it false understanding of the model output. Based on usecase, we can give importance to precision and recall with help of **F1 Beta score**.

Find this below snippet from wikipedia:

<center>
<img src="{{site.url}}/assets/images/ml/f1_beta_score.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 11:</b> F1 Beta Score</p> 
</center>

**[ROC CURVE](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)**

ROC- Receiver Operating Characteristics, its a plot between True Positive rate (Sensitivity) vs False Positive Rate (1 - specificity). Its a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

<center>
<img src="{{site.url}}/assets/images/ml/roc.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 12:</b> Area under the ROC curve</p> 
</center>

ROC curves are frequently used to show in a graphical way the connection/trade-off between clinical sensitivity and specificity for every possible cut-off for a test or a combination of tests. In addition the area under the ROC curve gives an idea about the benefit of using the test(s) in question.

<p>$$TPR = {TP \over (TP + FN)}$$</p>

<p>$$FPR = {FP \over (FP + TN)}$$</p>

It’s easy to see that if the threshold is 0, all our predictions will be positive, so both TPR and FPR will be 1 (the upper right corner). On
the other hand, if the threshold is 1, then no positive prediction will be made, both TPR and FPR will be 0 which corresponds to the lower left corner.

The higher the area under the ROC curve (AUC), the better the classifier. A classifier with an AUC higher than 0.5 is better than a random classifier. If AUC is lower than 0.5, then something is wrong with your model. A perfect classifier would have an AUC of 1. Usually, if our model behaves well, we obtain a good classifier by selecting the value of the threshold that gives TPR close to 1 while keeping FPR near 0.

**REFERENCE** [ROC Curve- what and How ?](https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used)
     
  <p align='center'>. . . . .</p>
