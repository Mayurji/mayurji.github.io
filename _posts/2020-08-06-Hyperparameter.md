---
layout: post
title:  Hyperparameter Tuning
description: Things we control in DL
category: Blog
date:   2020-08-06 13:43:52 +0530
---
{% include mathjax.html %}

<center>
<img src="{{site.url}}/assets/images/Hyperparameter/hyperparameter.jpg" style="zoom: 5%; background-color:#DCDCDC;"  width="75%" height=auto/><br>
<p>Figure 1: Neurons</p> 
</center>

### Topics covered

  1. What is Hyperparameter
  2. Learning Rate
  3. Minibatch Size
  4. Number of Iterations
  5. Number of Hidden Units and Layers
    

### What is Hyperparameter

**Hyperparameter is a static parameter or a variable, which needs to be assigned a value before applying an algorithm on data. For instance, parameters like learning rate, epochs, etc. are set before training the models.**

**Optimization Hyperparameter**- These parameters are related to optimization processes like gradient descent (learning rate), training process, mini-batch sizes, etc.

**Model Hyperparameter**- These parameters are related to models like several hidden layers or number of neurons in each layer etc.

### Learning Rate

It is the most important of all hyperparameters. Even if we are using a pre-trained model, we should try out multiple values of learning rate. The most commonly used learning rate is **0.1, 0.01, 0.001, 0.0001, 0.00001** etc.

<center>
<img src="{{site.url}}/assets/images/Hyperparameter/learning_rate.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Learning Rate</p>
</center>

A **large value** of learning rate tends to overshoot the gradient value making it difficult to converge the weight to the global minimum.

A **small value** of learning rate makes the progress towards global minimum very slow, which can be recognized from the validation and training loss.

An **optimum value** of learning rate will lead to a global minimum, which can be viewed by constantly decreasing loss.

### Learning Rate Decay

Sometimes keeping only one learning rate may not help us in reaching the global minimum, so changing the value of the learning rate after a certain number of epochs such that convergence takes place if the gradient is stuck in a local minimum.

<center>
<img src="{{site.url}}/assets/images/Hyperparameter/learning_rate_decay.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Learning Rate Decay</p>
</center>

### Adaptive Learning Rate

Sometimes it's crucial to understand the problem and change the learning rate accordingly like increasing it or decreasing it. Algorithms like Adam Optimizer and Adagrad Optimizer.

### Minibatch Size

It is one of the most commonly tuned hyperparameters in deep learning. If we have 1000 records for training the model then we can have three different sets of minibatch sizes.

**First**

If we keep Minibatch size = 1, then the weights are updated for every record after backpropagation. It is called Stochastic Batch Gradient Descent.

<center>
<img src="{{site.url}}/assets/images/Hyperparameter/minibatch.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Minibatch</p>
</center>


**Second**

If we keep Minibatch Size = # of records in the dataset, then the weights are updated after all records passed through the forward propagation. It is called Batch gradient descent.

**Third**

If we keep Minibatch Size = value between 1 to total no. of records, then the weights are updated after all set no. of records are passed through the forward propagation. It is called Mini-batch gradient descent.

The most commonly used value for **Minibatch sizes is 32, 64, 128, 256.** Values more than 256 require more memory and computational efficiency.

### Number of Iterations

The number of iteration or epochs can decide based on the validation error, as long as the validation error keeps decreasing we can assume that our model is learning and updating the weights positively. There is a technique called early stopping which helps in determining the no. of iterations.

<center>
<img src="{{site.url}}/assets/images/Hyperparameter/iterations.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 5: Iterations</p> 
</center>

[**Validation Monitor**](https://www.tensorflow.org/get_started/monitors#early_stopping_with_validationmonitor)

```python
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      test_set.data,
      test_set.target,
      every_n_steps=50,
      metrics=validation_metrics,
      early_stopping_metric="loss",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=200)
```
      

The last parameter indicates to ValidationMonitor that it should stop the training process if the loss did not decrease in 200 steps (rounds) of training.

[**Session Run Hook**](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook)

It is an evolving part of tf.train and going forward appear to be the proper place where you’d implement early stopping.

[**StopAtStepHook**](https://www.tensorflow.org/api_docs/python/tf/train/StopAtStepHook)

A monitor to request the training stop after a certain number of steps.

[**NanTensorHook**](https://www.tensorflow.org/api_docs/python/tf/train/NanTensorHook)

It monitors losses and stops training if it encounters a NaN loss.

### Number of Hidden Units/Layers

Highly mysterious parameters are deciding the # of hidden units and layers. What we are trying to achieve in deep learning is building a complex mapping function between features and targets. **To develop a complex function, the complexity is directly proportional to the # of hidden units, greater hidden units mean more the complexity of the function.** A point to note is, if we create too complex of a model then it overfits the training data, and which could be seen from the validation error while training, we should reduce the hidden units in that case.

To conclude, keep track of validation errors while increasing the number of hidden units.

**As stated by Andrej Karpathy, a 3 layer net outperforms the 2 layer net but going beyond that rarely helps the network. While in CNN, the greater the # of layers, the greater will be the performance of the network.**

[Andrej Karpathy](https://cs231n.github.io/neural-networks-1/)

### Further Reading

[How to batch size affects the model performance](https://arxiv.org/abs/1606.02228) |
[Stackexchange](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent) |
[BGD vs SGD](https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1) |
[Visualizing Networks](http://jalammar.github.io) |
[Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) |
[Deep Learning Book by Ian Goodfellow](http://www.deeplearningbook.org/contents/guidelines.html) |
[Generate Good Word Embedding](https://arxiv.org/abs/1507.05523) |
[Exponential Decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay) |
[Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) |
[Adagrad Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
