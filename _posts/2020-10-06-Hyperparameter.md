---
layout: post
title:  Hyperparameter Tuning
description: Why tuning hyperparameters of your model is important and how it effects the models performance.
date:   2020-10-06 13:43:52 +0530
---

## **Hyperparameter Tuning**
#### **Discover a way to efficiently tune hyperparameters**


![Hyperparameter]({{site.url}}/assets/images/Hyperparameter/hyperparameter.jpg)

## **Topics covered**

   >**What is Hyperparameter\
    Learning Rate\
    Minibatch Size\
    Number of Iterations\
    Number of Hidden Units/Layers**\
    

## **What is Hyperparameter**

**Hyperparameter is a static parameter or a variable, which needs to be assigned a value before applying an algorithm on a data. For instance, parameters like learning rate, epochs etc. are set before training the models.**

**Optimization Hyperparameter**- These parameters are related to optimization process like gradient descent (learning rate), training process, mini batch sizes etc.

**Model Hyperparameter**- These parameters are related to models like number of hidden layers or number of neurons in each layers etc.

## **Learning Rate**

It is the most important of all hyperparameter. Even if we are using pre-trained model, we should try out multiple values of learning rate. The most commonly used learning rate is **0.1, 0.01, 0.001, 0.0001, 0.00001** etc.

![Learning Rate]({{site.url}}/assets/images/Hyperparameter/learning_rate.png)

A **large value** of learning rate tend to overshoot the gradient value making it difficult to converge the weight to global minimum.

A **small value** of learning rate make the progress towards global minimum very slow, which can recognized from the validation and training loss.

An **optimum value** of learning rate will leads to global minimum, which can be viewed by constantly decreasing loss.

### **Learning Rate Decay**

Sometimes keeping only one learning rate may not help us in reaching the global minimum, so changing the value of learning rate after a certain number of epoch such that convergence takes place if the gradient is stuck in local minimum.

![Learning Rate Decay]({{site.url}}/assets/images/Hyperparameter/learning_rate_decay.png)

### Adaptive Learning Rate

Sometimes its crucial to understand the problem and change the learning rate accordingly like increasing it or decreasing it. Algorithms like Adam Optimizer and Adagrad Optimizer.

[Exponential Decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)

[Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

[Adagrad Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)


## **Minibatch Size**

It is one of the commonly tuned parameter in deep learning. If we have 1000 records for traning the model then we can have three different set of minibatch size.

**First**

If we keep Minibatch size = 1, then the weights are updated for every record after backpropagation. It is called as Stochastic Batch Gradient Descent.

![Minibatch]({{site.url}}/assets/images/Hyperparameter/minibatch.png)

**Second**

If we keep Minibatch Size = # of records in dataset, then the weights are updated after all records passed through the forward propagation. It is called Batch gradient descent.

**Third**

If we keep Minibatch Size = value between 1 to total no. of records, then the weights are updated after all set no. of records are passed through the forward propagation. It is called Mini-batch gradient descent.

Most commonly used value for **Minibatch sizes are 32, 64, 128, 256.** Values more than 256 requires more memory and computational efficiency.

[How batch size affects the model performance](https://arxiv.org/abs/1606.02228)

[Stackexchange](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent)

[BGD vs SGD](https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1)


## **Number of Iterations**

The number of iteration or epoch can decided based on the validation error, as long as validation error keeps decreasing we can assume that our model is learning and updating the weights positively. There is a technique called as early stopping which helps in determining the no. of iterations.

![Iterations]({{site.url}}/assets/images/Hyperparameter/iterations.png)

[Validation Monitor](https://www.tensorflow.org/get_started/monitors#early_stopping_with_validationmonitor)

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

It is an evolving part of tf.train, and going forward appear to be the proper place where you’d implement early stopping.

[**StopAtStepHook**](https://www.tensorflow.org/api_docs/python/tf/train/StopAtStepHook)

A monitor to request the training stop after a certain number of steps.

[**NanTensorHook**](https://www.tensorflow.org/api_docs/python/tf/train/NanTensorHook)

A monitor that monitor’s loss and stops training if it encounters a NaN loss.

## **Number of Hidden Units / Layers**

Highly mysterious parameters is deciding the # of hidden units and layers. What we are trying to achieve in deep learning is building a complex mapping function between features and target. **To develop complex function, the complexity is directly proportional to the # of hidden units, greater hidden units means more the complexity of the function.** A point to note is, if we create too complex of a model then it overfits the training data and which could be seen from the validation error while training, we should reduce the hidden units in that case.

To conclude, keep track of validation error while increasing the number of hidden units.

**As stated by Andrej Karpathy, a 3 layer net outperforms the 2 layer net but going beyond that rarely helps the network. While in CNN, greater the # of layers, greater will be the performance of the network.**

[Andrej Karpathy](https://cs231n.github.io/neural-networks-1/)
[Deep Learning Book](http://www.deeplearningbook.org/contents/ml.html)


## References

[Visualizing Networks](http://jalammar.github.io)\
[Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533)\
[Deep Learning Book by Ian Goodfellow](http://www.deeplearningbook.org/contents/guidelines.html)\
[Generate Good Word Embedding](https://arxiv.org/abs/1507.05523)
