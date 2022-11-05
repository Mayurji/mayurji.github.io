---
layout: post
title:  Knowledge Distillation, aka. Teacher-Student Model
description: Knowledge Transfer from teacher to student
category: Blog
date:   2022-10-22 13:43:52 +0530
---
{% include mathjax.html %}

With the release of large models in the last few years, from GPT-3 to Megatron, I keep pondering how to experiment and use these models for a specific use case. These models are trained on massive corpuses of data (100+ GBs) with billions of parameters. Training or performing inference using such a model requires heavy computing and cost.

### Knowledge Distillation
To overcome the above challenges, there are techniques developed to transfer the knowledge learnt by a larger model (teacher) and instil it into a smaller model (student).

Here the knowledge refers to parameters learnt during model training. This whole concept is called “Knowledge Distillation.” Knowledge distillation in machine learning refers to transferring knowledge from a teacher to a student model.

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/1-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Knowledge Distillation</p> 
</center>

We can understand this teacher-student model as a ***teacher who supervises students to learn and perform in an exam. Though teachers have broad knowledge of the subject, they can narrow it down to the topics required by the student to excel in their exam.***

Now, with the understanding of what is knowledge distillation, we can talk about the different knowledge aspects in a teacher model that can be distilled to get a student model.

- Feature based knowledge distillation
- Response based Knowledge distillation
- Relation based knowledge distillation

### Feature Based Knowledge Distillation

In order to make the parameters (Ws) in the teacher model discriminate between the features of the different objects or classes, the parameters (Ws) in the training process are learned against a loss function.

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/2-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Feature Distillation</p> 
</center>

For each class, there is a set of parameters (features) that are activated, which help in predicting that object or class. These same sets of feature activations are now learned using distillation loss when training a student model. The difference between a teacher’s and a student’s model’s activation of certain features is reduced with the aid of the loss function.

### Response Based Knowledge Distillation

In feature distillation, the student model is optimized for feature activation against distillation loss. In response-based distillation, the student model makes use of the teacher model’s logits to enhance its own logits.

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/3-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Response Distillation</p> 
</center>

Here, the distillation loss aims to capture and reduce the variation in logits between the student and teacher models. The teacher model’s predictions are mimicked as the student model gains experience through training iterations.

### Relation Based Knowledge Distillation

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/4-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Relation Distillation</p> 
</center>

In feature and response-based knowledge distillation, the outputs from feature activation and logits are used as a leverage to build the student model. However, in this case, we use the interrelationship of the layers as input with a hidden layer or a hidden layer with output as the source of learning for the student model.

These relationships now consist of layers that can be thought of as probability distributions, similarity matrices, or feature embeddings. As a result, the student learns from the teacher’s model how to construct its embedding or distribution.

### How to train a student model from teacher model?

Training a student model also called as Distillation Schemes, refers to how a teacher model can distil the knowledge to a student model, whether a student model can be trained along with teacher model or not. 

Distillation scheme is divided into three main categories as follows

- Offline Distillation
- Online Distillation
- Self-Distillation

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/5-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Distillation Schemes</p> 
</center>

### Offline Distillation

Training a student model in offline mode consists of two steps

1. Training a large teacher model on a set of training samples.
2. Extracting knowledge from pre-trained teacher model in the form of logits or intermediate features.

Logits and intermediate features are used as guide to train a student model. Usually the first step is not considered as knowledge distillation step as it assumed to be pre-defined. Offline Distillation mainly focuses on transfer of knowledge from specific parts of the teacher model like sharing probability distribution of data in the feature space or loss function for matching features. The main advantage of offline distillation is that it is easy to implement, but the only drawback is the initial training time required for the high-capacity teacher model.

### Online Distillation

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/6-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Online Distillation</p> 
</center>

Though offline distillation is simple to implement, the issue remains in finding a large-capacity high performance teacher model, which is a prerequisite for offline distillation. Thus, to resolve this issue, comes the online distillation. In Online Distillation, the student and teacher model are updated simultaneously and the whole knowledge distillation is end-to-end trainable. Various works have been developed over the years like deep mutual learning, multiple neural network work in a collaborative way. 

Researchers employed a online distillation to train large-scale distributed neural network, and proposed a variant of online distillation called co-distillation. Co-distillation in parallel trains multiple models with the same architectures and any one model is trained by transferring the knowledge from the other models. However, existing online methods (e.g., mutual learning) usually fails to address the high-capacity teacher in online settings.

### Self-Distillation

In self-distillation, the same network is used for both the teacher and student model. In 2019, researchers proposed a self-distillation method, in which knowledge from the deeper sections of the network is distilled into its shallow sections. 

<center>
<img src="{{site.url}}/assets/images/knowledgeDistillation/7-kd.png"  style="zoom: 5%  background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Self-Distillation: (i) A ResNet has been divided into four sections according to their depth. (ii) Additional bottleneck and fully connected layers are set after each section, which constitutes multiple classifiers. (iii) All of the classifiers can be utilized independently, with different accuracy and response time. (iv) Each classifier is trained under three kinds of supervision as depicted. (v) Parts under the dash line can be removed in inference.</p> 
</center>

Similar to self-distillation, self-attention was proposed for lane detection. The network utilizes the attention maps of its own layers as distillation targets for its lower layers. In other approach, knowledge from earlier epochs of the teacher model can be transferred to its later epochs to train the student model.

To be continued. Til then

<iframe src="https://giphy.com/embed/H48YKEw3fXrcvIF2xE" width="480" height="460" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/wearebottle-study-information-absorbing-H48YKEw3fXrcvIF2xE">via GIPHY</a></p>
