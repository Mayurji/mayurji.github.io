---
layout: deep-learning
title: Five Pointer After 30 Days Of ML With PyTorch
description: PyTorch Pointers
date:   2021-05-07 16:43:52 +0530
---

### After 30DaysOfML with Pytorch

Recently, I've completed 30 Days of ML with Pytorch, where I explored all the major machine learning algorithms and basics of deep learning with concepts like activations, optimizers, loss functions etc.

However, the idea behind starting 30 Days of ML with Pytorch is to learn the implementation of machine learning algorithm along with strengthening my grasp on PyTorch Library. So moving forward with this blog, I will share few important functions in pytorch which remains part of every ML and DL algorithms.

* **Initializing Weights**

  How we initialize weights plays a great role in convergence of the model and a common but inefficient way to initialize weight is to declare weights as zeros, but other efficient way includes using **uniform** or **normal** or **xaviers** technique to initialize weights.

  ```python
  import torch
  from torch import nn
  
  #1. Simple and Inefficient
  weight_vector = torch.zeros(1, n_features)
  """
  tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
  """
  
  #2. Better than zero initialization
  weight_vector = torch.FloatTensor(1, n_features).uniform_(-1,1) #normal
  """
  tensor([[ 0.8193, -0.7194,  0.5021, -0.0658, -0.6250,  0.5181, -0.0742,  0.4049,
            0.9357, -0.8531,  0.4650,  0.8140, -0.2959]])
  
  """
  
  #3 Using nn Module
  weight_vector = torch.zeros(1, 13)
  nn.init.uniform_(weight_vector, a=0, b=1)
  
  """
  tensor([[0.1853, 0.5137, 0.9067, 0.8625, 0.2130, 0.4607, 0.8734, 0.5269, 0.2133,
           0.1658, 0.7785, 0.4878, 0.1473]])
  """
  
  #4 Using Xavier Uniform for neural networks
  weight_vector = torch.zeros(1, 13)
  nn.init.xavier_uniform_(weight_vector) #try out kaiming
  """
  tensor([[ 0.2606, -0.3418, -0.1334,  0.6496, -0.4526, -0.5594, -0.1503, -0.4093,
            0.4468,  0.4897, -0.2890,  0.0265, -0.6017]])
  """
  ```

* **Creating Mask And Removing based on threshold**

  It is applied for creating **dropout** functionality in neural network for regularization.

  ```python
  y_ = torch.FloatTensor(1, 13).uniform_(-1,1)
  """
  tensor([[ 0.3296,  0.9327, -0.7962, -0.8314,  0.9889, -0.5515,  0.0733, -0.6384,
            0.3541,  0.3678,  0.6770, -0.5216, -0.1539]])
  """
  mask = y_ < 0.5
  """
  tensor([[ True, False,  True,  True, False,  True,  True,  True,  True,  True,
           False,  True,  True]])
  """
  mask * y
  """
  tensor([[ 0.3296,  0.0000, -0.7962, -0.8314,  0.0000, -0.5515,  0.0733, -0.6384,
            0.3541,  0.3678,  0.0000, -0.5216, -0.1539]])
  """
  ```

* **Filling Values Matrix**

  When specific set of value needs to be assigned as weight with equal weight to each feature based on sample size. Used in Boosting algorithm like **Adaboost**.

  ```python
  n_samples = 100
  n_features = 13
  weight = torch.zeros(1,n_features).fill_(1/n_samples)
  
  """
  tensor([[0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100,
           0.0100]])
  """
  ```

* **Calculating Distance**

  While performing **clustering, dimensionality** **reduction** and many other ml algorithms, we use various distance metrics to find  distance between vectors and matrices. In PyTorch, we can perform these  operations using **cdist**.

  ```python
  w1 = torch.FloatTensor(1, 10).uniform_(0, 1)
  w2 = torch.FloatTensor(1, 10).uniform_(0, 1)
  torch.cdist(w1, w2) # default p=2, L2 Norm or Euclidean Distance
  """
  tensor([[1.1071]])
  """
  torch.cdist(w1, w2, p=1) # Manhattan Distance or L1-Norm
  """
  tensor([[2.7885]])
  """
  torch.cdist(w1, w2, p=0) # Hamming Distance
  """
  tensor([[10.]])
  """
  ```

  **Replacing Diagonal with Vector**

  Not sure, how often this is used in building ML algorithms, but this is  quite tedious because, to set diagonal elements with one specific values is simple but it becomes difficult when we need to replace the diagonal elements of a matrix with a vector.

  ```python
  w = torch.FloatTensor(3,3).uniform_(0, 1)
  w
  """
  tensor([[0.0815, 0.3858, 0.8555],
          [0.8008, 0.4362, 0.4959],
          [0.7629, 0.2106, 0.6766]])
  """
  diagonal = torch.FloatTensor(3,3).uniform_(0,1)
  diagonal
  """
  tensor([[0.6585, 0.3995, 0.3589],
          [0.0166, 0.5944, 0.3455],
          [0.1984, 0.6490, 0.5522]])
  """
  w.as_strided([w.shape[0]], [w.shape[0]+1]).copy_(torch.diag(diagonal))
  """
  tensor([0.6585, 0.5944, 0.5522])
  """
  w
  """
  tensor([[0.6585, 0.3858, 0.8555],
          [0.8008, 0.5944, 0.4959],
          [0.7629, 0.2106, 0.5522]])
  """
  ```

  

​		  If anyone is interested in checking out the machine learning algorithm using PyTorch and try out how all the functions play     

​		  out in ML, then  check out of [github repository](https://github.com/Mayurji/MLWithPytorch) for 30 Days of ML with PyTorch.

