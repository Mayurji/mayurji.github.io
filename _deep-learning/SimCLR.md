## SimCLR — An Answer To Unlabelled Data

![SimCLR](https://giphy.com/gifs/simclr-8SZhj0qY3XlHcr1150?utm_source=iframe&utm_medium=embed&utm_campaign=Embeds&utm_term=https%3A%2F%2Fcdn.embedly.com%2F)
Figure 1. Google AI Blog

### What is Self-Supervised Learning?

It is a method of machine learning where the model learns from the supervisory signal of the data unlike supervised learning where separate labels are specified for each observation. It is also known as Representation Learning.

Note, the model’s learned representation is used for downstream tasks like BERT, where language models are used for text classification tasks. Here, we can use Linear classifiers along with a learned self-supervised model for prediction.

![Contrastive Learning](https://miro.medium.com/max/700/0*dEbG01Fg4oZ7dInY)
Figure 2. Google AI Blog

### Why Self-Supervised Learning?

Supervised learning requires a large amount of labelled dataset to train a model. For instance, current deep learning architectures for image classification are trained on labelled ImageNet dataset, and it took years to build the dataset.

In addition, there is a vast amount of unlabelled data on the internet which the self-supervised learning can tap into, to learn the rich and beautiful representations of the data.

### What is Contrastive Learning?

Contrastive Learning is a self-supervised learning approach where the model aims to learn the representation by forcing similar observations to be closer and dissimilar observations to be far apart.

Therefore, it tries to increase the contrast between dissimilar elements and reduce the contrast between the similar elements.

**SimCLR:** a simple framework for contrastive learning of visual representations.

![Model Flow](https://miro.medium.com/max/700/1*UAigJwzS02cmyCDXrfobBA.png)
Figure 3. Amitness Blog

Major components of SimCLR

   Using a wide range of data augmentation techniques helps in creating effective representation.
    
    - Random Crop, Resize, Gaussian Blur, Color Distortion etc.

   Building a learnable nonlinear model between representation and contrastive loss improves the quality of the learned representation. It is also known as a projection head.
   
    - Feedforward Network with Relu activation, g(hi).
    
  ![SimCLR Paper](https://miro.medium.com/max/654/0*QT2tMItUrL9nLVhG)
  Figure 4. Arxiv Paper
  
      Contrastive cross entropy loss along with temperature parameters benefits in representation learning.(TBD temperature)
    Greater batch size and longer training benefits the model in learning effective representation.

   In the above image, f(.) is the base encoder network that extracts the representation from augmented images. Authors use the ResNet (f) architecture to learn the image representation and projection head g(.) maps the representation to the contrastive loss function.
   
   Given a positive pair of augmented images xi and xj , the contrastive prediction task tries to predict whether xj is similar to xi. Positive pair means that xi and xj belong to the same image while negative pair refers to xi and xj belonging to different images.
   
### NCE Loss

   Contrastive loss used in SimCLR is NT-Xent Loss (Normalized Temperature Scaled Cross Entropy Loss).
   
   ![Loss Function](https://miro.medium.com/max/554/0*Yy-F783_QLlVkybQ)
   Figure 5. NCE Loss
   
𝜏 — Temperature factor

When a min-batch is selected, let’s say two, we have a total of four images after augmentation.


