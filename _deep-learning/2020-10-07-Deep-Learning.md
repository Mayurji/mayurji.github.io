---
layout: deep-learning
title: Deep Learning
description: "The idea behind neural networks and why it works for complex problems!"
date:   2020-10-07 13:43:52 +0530
---
{% include mathjax.html %}

**NEURAL NETWORK**

Neural Network models are similar to machine learning models in context of learning a function. 
For instance, In Linear regression, the model learns the parameter $$w$$ in equation $$y=wx+c$$.
In Neural Networks, the models learns parameters w for complex unstructured datasets such images,
audio, text etc.

<center>
<img src="{{site.url}}/assets/images/resnet/mlvsdl.png" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p><b>Figure 1:</b> Deep Learning vs Machine Learning</p> 
</center>

The idea behind neural network is inspired from brain, researchers believed that neurons in brain are fired 
when it cross a threshold of sensitivity, the brain is basically a logical inference machine because neurons 
are binary. Neurons compute a weighted sum of inputs and compare that sum to its threshold. It turns on if 
it’s above the threshold and turns off if it’s below, which is a simplified view of how neural networks work.

The key difference between ml model vs a dl model is the feature extraction technique. We perform feature
engineering in machine learning model manually and use our creativity to build better features for the model
to learn but a deep learning model uses multiple layers of neurons(parameters) to learns extensive features from
the data during training.

The complexity of a neural network increases with complexity in the data and the task in hand. Neural
network are preferred less over machine learning model because of the explainability and interpretability
provided by simple ml algorithms. While using neural networks, there is a trade-off between explainability 
and performance in terms of accuracy. Currently, there is lot of active research being done to unravel the
black box called neural nets.

General Equation of Neural Network

<p>$$y=f_N(x)$$</p>

Now, the function $$f_NN$$ can get very complex based on task and data. For instance, $$f_N$$ can inherit 
multiple functions in it as follows,

<p>$$y=f_3(f_2(f_1(x)))$$</p>

We can call the above equation as three layered network, because passes through three sets of functions. A layer in neural network
are set of neurons or parameters connecting one layer to another layer with parameters getting learned in between them.
It basically combines transformation of multiple features with help activation units. Activation units in neural network
are simple function which introduces non-linearity to the neural network to learn better features and understand complex 
datasets like images, texts etc.

Find this annotated research paper which explains role of individual unit in neural network, [Role of Individual Units in Deep Neural Networks](https://github.com/Mayurji/Deep-Learning-Papers/tree/master/Investigate%20DNN)

<p align=center>. . . . .</p>
