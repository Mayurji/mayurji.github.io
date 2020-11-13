---
layout: deep-learning
title: Deep Learning with Pytorch - I
description: "Learn how to use pytorch for deep learning"
date:   2020-10-07 16:43:52 +0530
---
{% include mathjax.html %}

### Deep Learning Using Pytorch

In Deep Learning, It is preferred to use a pre-trained model as initialization for your new model rather than 
creating a model from scratch. Pre-training provides a major boost to your model, by retaining essential features, 
in initial layers of the pre-trained model. And also, the training time cost and hardware required is drastically
brought down because pretrained model are SOTA models which are trained on huge dataset and expensive hardware.

So, we'll learn how to use a pretrained model in Pytorch and then we'll look into the basics of pytorch.

#### How to load and predict using Pre-trained Model - Resnet34

Pytorch provides three sets of libraries i.e. torchvision, torchaudio, torchtext for Vision tasks, Audio tasks and 
Text tasks respectively.

**Importing Libraries**
```python
import torch
import numpy as np
import os
import cv2
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
```
**Loading Pretrained Model**

Using models module from Torchvision, we can load many pretrained models which exists in pytorch. I am loading a resnet34 
model with pretrained=True, which means I will be using weights of the model on which it was trained on. For Instance, 
resnet34 was trained on Imagenet dataset. On executing the below line, the models module will download the model from 
Pytorch if it doesn't exists in your system.

```python
resnet = models.resnet34(pretrained=True)
```
pretrained=True returns a pre-trained model.

**Creating Transforms**

Transforms is a cool feature in torchvision, because we can apply a list of transforms/augmentation on an image by just simply 
adding it as parameter in transforms module. We can also customize other transforms if required, if its not included in 
torchvision.transforms.

```python
preprocess= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.2, 0.2, 0.2])
    ])
```

**Loading your Single Image**

  * Using Python Image Library (PIL) for loading an single image.
  * Applying the transformation declared above.
  * Checking out the shape of the image.
  * Note the shape of the image, it should apply the transforms.centercrop() and resize the image.

```python
img = Image.open('../Images/traffic.jpeg')
img_p = preprocess(img)
print(img_p.shape)
#torch.Size([3, 224, 224])
```

Pytorch uses first dimension of matrix to represent the batch size. So the pretrained model requires batch size as first dimension, 
so we reshape the image dimension. In pytorch, we use unsqueeze to add an dimension to existing matrix.

```python
batch_t = torch.unsqueeze(img_p, 0)
batch_t.shape
#torch.Size([1, 3, 224, 224])
```
Pytorch, we can use a model in two modes, train mode and an eval mode. In train mode, the model learns model parameters
and we perform batch normalization and dropout layers to avoid overfitting. In eval mode, pytorch automatically disables the
batch norm and dropout layer.

Since we are predicting using pretrained model, we use model under eval mode. We are initializing the resnet34 model and predicting
on one image under batch_t variable. The out variable contains our predicted output over 1000 classes. Since resnet was trained on 
Imagenet, which has 1000 classes.

```python
resnet.eval()
out = resnet(batch_t)
print(out.shape)
#torch.Size([1, 1000])
```
**Loading Images Classes from txt file**

Below, we are loading the class names of the classes in Imagenet. We can download the classnames of the dataset from web.
```python
with open('imagenet_class.txt') as f:
    classes = [line.strip().split(",")[1].strip() for line in f.readlines()]
```
**Finding Index of the max probability class**

The variable out is a vector with 1000 elements with set of values providing weights to each class. Higher weight of the class
results as predicted class of the image. Using max and dimension=1, we are fetching the index of the vector, where the weight is
maximum among 1000 classes.
```python
_, index = torch.max(out, 1)
index
#tensor([920])
```
**Prediction Confidence**

Softmax function is used in Multiclass classification, it squeezes the value/weight as mentioned above between 0 and 1. 
So the all 1000 weights are squeezed between 0 to 1 and all summing up to 1. We further convert the class index into 
label of the class and present it as confidence percentage.

```python
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
classes[index[0]], percentage[index[0]].item()
#('traffic_light', 99.99995422363281)
```

**Top 5 predictions**

Similar to above code, only showing top five predictions of an image.
```python
_, indices = torch.sort(out, descending=True)
[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
'''
[('traffic_light', 99.99995422363281),
 ('street_sign', 2.8018390366923995e-05),
 ('pole', 9.717282409837935e-06),
 ('loudspeaker', 2.9554805678344565e-06),
 ('binoculars', 1.4750306718269712e-06)]
'''
```

**Reference** [Deep Learning With Pytorch by Eli Stevens and Luca Antiga](https://www.manning.com/books/deep-learning-with-pytorch)
