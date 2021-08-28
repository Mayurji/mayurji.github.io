---
layout: machine-learning
title: Keepsake - A Version Control For Machine Learning
description:  Machines - Deploy, Track, Repair, and Repeat
date:   2021-03-07 13:43:52 +0530
---
{% include mathjax.html %}

When ML was getting mainstream like 4 to 5 years back in all the industry everyone were talking about how to build great models either by classical machine learning or DNN, etc. But today with tools like AutoML and Model Search, building models has taken a little backseat. Now, the major challenge is not about building a great model but about keeping track of all the built models.

In this blog post, we'll discuss **Keepsake**. In simple terms, we can think of keepsake as the **version control** tool for machine learning.

**From Keepsake Official Documents**

*Everyone uses version control for their software and it's much less common in Machine Learning. This causes all sorts of problems: people are manually keeping track of things in spreadsheets, model weights are scattered on S3, and nothing is reproducible. It's hard enough getting your own model from a  month ago running, let alone somebody else's.*

*So why isn’t everyone using Git? **Git doesn’t work well with machine learning.** It can’t handle large files, it can’t handle key/value metadata like metrics, and it can’t record information automatically from inside a  training script. There are some solutions for these things, but they  feel like band-aids.*

**Installing Keepsake**

```python
pip install -U keepsake
```

## How to use keepsake

Do everything you do normally while building a model. But just add two lines and one small *yaml* file. 

* **Initialize an Experiment using keepsake**
* **Creating Checkpoint for the Experiment**

```python

        ### Let this be main.py ###
        import argparse
        import keepsake
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle
        import torch
        from torch import nn
        from torch.autograd import Variable


        def train(learning_rate, num_epochs):
            # highlight-start
            # Create an "experiment". This represents a run of your training script.
            # It saves the training code at the given path and any hyperparameters.
            experiment = keepsake.init(
                path=".",
                # highlight-start
                params={"learning_rate": learning_rate, "num_epochs": num_epochs},
            )
            # highlight-end

            print("Downloading data set...")
            iris = load_iris()
            train_features, val_features, train_labels, val_labels = train_test_split(
                iris.data,
                iris.target,
                train_size=0.8,
                test_size=0.2,
                random_state=0,
                stratify=iris.target,
            )
            train_features = torch.FloatTensor(train_features)
            val_features = torch.FloatTensor(val_features)
            train_labels = torch.LongTensor(train_labels)
            val_labels = torch.LongTensor(val_labels)

            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(4, 15), nn.ReLU(), nn.Linear(15, 3),)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_features)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    output = model(val_features)
                    acc = (output.argmax(1) == val_labels).float().sum() / len(val_labels)

                print(
                    "Epoch {}, train loss: {:.3f}, validation accuracy: {:.3f}".format(
                        epoch, loss.item(), acc
                    )
                )
                torch.save(model, "model.pth")
                # highlight-start
                # Create a checkpoint within the experiment.
                # This saves the metrics at that point, and makes a copy of the file
                # or directory given, which could weights and any other artifacts.
                experiment.checkpoint(
                    path="model.pth",
                    step=epoch,
                    metrics={"loss": loss.item(), "accuracy": acc},
                    primary_metric=("loss", "minimize"),
                )
                # highlight-end


        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--learning_rate", type=float, default=0.01)
            parser.add_argument("--num_epochs", type=int, default=100)
            args = parser.parse_args()
            train(args.learning_rate, args.num_epochs)

```
* First line, we initialize **keepsake.init()**, which creates an experiment, each experiment is the one run of our training script. It stores hyperparameter and makes a copy of our training code and stores it into the path mentioned in the *init()*.

* Second line is **experiment.checkpoint()**, It creates a checkpoint for the experiment we are running. It tracks and saves all metrics we want to the path as mentioned in the **checkpoint function**.

Each experiment contains multiple checkpoints. You typically save your model periodically during training, because the best result isn't necessarily the most recent one. A checkpoint is created just after you save your model, so Keepsake can keep track of versions of your saved model.

### Storing Experiments

To store the experiments, we need to tell keepsake where to dump for each experiment.

**For Local**: With the below line, we will create a hidden folder with the name **.keepsake** and store the experiments, checkpoints, and related metadata inside it. Store the below line in a .yaml file and in the same path as your training script.

```yaml
repository: "file://.keepsake"
```

**For Cloud**: With the below line, we will pass all the experiments, checkpoints, and related metadata inside an S3 bucket in your AWS setup. Here keepsake-trial is the bucket name.

```yaml
repository: "s3://keepsake-trial"
```

### Running and Checkpoint Experiment

**Executing main.py**

<div>
<center>
<img src="{{site.url}}/assets/images/ml/run_experiment.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Run Experiments</p>
</center>
</div>

**Checkpoints are stored and can be seen in the last step**

<div>
<center>
<img src="{{site.url}}/assets/images/ml/creating_checkpoint.png " style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Creating Checkpoints</p>
</center>
</div>

**Setting Up AWS For Keepsake**

* I am assuming people to know how to create an AWS account.
* Use the Free tier provided by **AWS**.
* Create a bucket in S3, keep the required options default if you don't know what to give while creating an S3 bucket.
* Go to your profile name on the top right corner and go to **My Security Credentials** for the access key and secret key, store it in some text file if required.

<div>
<center>
<img src="{{site.url}}/assets/images/ml/access_and_secret_key.png " style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Access Key and Secret Key in AWS</p>
</center>
</div>

* Then Install AWS CLI for the system. Installing [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html). 
* Once the installation is done, Go to the terminal and type
  `aws configure`
* I've done it on Ubuntu, I am not sure if the command changes in other OS. 

<div>
<center>
<img src="{{site.url}}/assets/images/ml/aws_configure.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: AWS configure</p>
</center>
</div>

* Fill in the details as mentioned in the above image.
* **JSON is the default option for the last question, and for the region, check out what is mentioned in the S3 bucket.**

Note: While trying to save the experiment in S3, keep the yaml with S3 and bucket name as the option.

### Cool Results from Keepsake

**Find out all experiments with *ls* command**

Each experiment has **experiment id** and **checkpoint id**

<div>
<center>
<img src="{{site.url}}/assets/images/ml/ls_command.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 5: All Experiments</p>
</center>
</div>

**Find out details of each experiment with *show* command**

<div>
<center>
<img src="{{site.url}}/assets/images/ml/show_command.png" style="zoom: 5%; background-color:#DCDCDC;"  width="60%" height=auto/><br>
<p>Figure 6: Show Experiment</p>
</center>
</div>

<div>
<center>
<img src="{{site.url}}/assets/images/ml/keepsake_checkpoint.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 7: Show Checkpoint</p>
</center>
</div>

**Find out the difference between multiple experiments or checkpoints with *diff* command**

<div>
<center>
<img src="{{site.url}}/assets/images/ml/diff_checkpoint.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 8: Difference between Checkpoints</p>
</center>
</div>

**How files are created and stored in AWS Bucket**

<div>
<center>
<img src="{{site.url}}/assets/images/ml/aws_s3_bucket.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 9: Experiment Stored in AWS Bucket</p>
</center>
</div>
