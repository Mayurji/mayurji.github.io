---
layout: deep-learning
title: Deep Learning
description: The idea behind neural networks and why it works for complex problems.
date:   2020-10-07 13:43:52 +0530
---
{% include mathjax.html %}

**Neural Network**

A neural network is a simple system of function developed based on the complexity of the data. Neural Network models are similar to machine learning models in the context of learning a function by learning the parameters. 

For instance, in Linear regression, the model learns the coefficient $$w$$ in equation $$y=wx+c$$ by iteratively running over data. Similarly, in Neural Networks, the models learn the matrix of parameters $$w$$ for complex unstructured datasets such as images,
audio, text, etc.

<center>
<img src="{{site.url}}/assets/images/resnet/mlvsdl.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto /><br>
<p>Figure 1: Deep Learning vs Machine Learning</p> 
</center>

**Idea Behind Neural Network**

The idea behind the neural network was inspired by the human brain, researchers believed that a neuron in the brain is fired, when it crosses a threshold of sensitivity, the brain is a logical inference machine because the neurons in our brain are binary. 

Neurons compute a weighted sum of inputs and compare that sum to a threshold. The neuron is activated if the weighted sum is greater than the threshold, or else the neuron remains inactive if the weighted sum is lesser than the threshold.

**Difference between ML and DL**

The main difference between an ml model and a dl model is the feature extraction technique. 

We perform feature engineering in a machine learning model manually and use our creativity and domain expertise to build features for the model to learn from. But a deep learning model uses multiple layers of neurons(parameters) to learns extensive features from the data during training.

The complexity of a neural network increases with the complexity of the data and the task at hand. A neural network is preferred less over the machine learning models because of the explainability and interpretability. 

While using neural networks, there is a trade-off between explainability and performance in terms of accuracy. Currently, there is active research being done to unravel the black box called neural nets.

**General Equation of Neural Network**

<p>$$y=f_N(x)$$</p>

Now, the neural net function $$f_N$$ can be very complex based on the task and data. For instance, $$f_N$$ can inherit 
multiple functions in it as follows,

<p>$$y=f_3(f_2(f_1(x)))$$</p>

We can call the above function a three-layered neural network because the input passes through three sets of functions. Now, What is a layer? A layer in a neural network is a set of neurons/nodes/units connecting one layer to other layers with parameters getting learned in between them during training.
Layers combine the transformation of multiple features with the help of activation function. The activation function in a neural network
is a simple function, which introduces a non-linearity to the neural network's weights to learn better features and understand complex structures present in data like images, texts, etc. Common activation function includes Relu, tanh, Sigmoid, etc.

<p align=center>. . . . .</p>

Find an annotated research paper that explains the role of an individual unit in the neural network, [**Role of Individual Units in Deep Neural Networks.**](https://github.com/Mayurji/Deep-Learning-Papers/tree/master/Investigate%20DNN)

If you've liked this post, please don't forget to subscribe to the newsletter.
