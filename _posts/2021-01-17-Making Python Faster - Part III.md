---
layout: post
title: Speeding up with GPUs
description: If not libraries, use hardware.
category: Blog
date: 2021-01-17 13:43:52 +0530
---

### GPUs and PyTorch

In this blog post, we’ll discuss GPU and why it is becoming a must-have tool in your arsenal for massive data computation, especially in Deep Learning, where we require maximum parallelism with matrix and vector computation.

GPU is a popular tool for performing heavy arithmetic workloads. It was designed for gaming, generating graphics as it solves parallelizable problems. On comparing GPU with a CPU, GPU has a lower computing rate, i.e., it has a lower clock speed. Even though it sounds counterintuitive, GPU gains its parallel processing power from the number of cores available. The majority of the CPUs have 12 or fewer cores, but modern-day GPU comes with thousands of cores for parallel processing.

<div>

<center>

<img src="{{site.url}}/assets/images/PythonFaster/fasterPython(1).png" style="zoom: 5%; background-color:#DCDCDC;" width="80%" height=auto/><br>

<p>Figure 1: Making Python Faster - Tools</p>

</center>

</div>

However, it is tough to program for GPU devices, and it acts as a bottleneck to attain maximum performance from it. However, the needs of modern deep learning algorithms have been pushing new interfaces into GPUs that are easy and intuitive to use. The two front-runners in terms of easy-to-use GPU mathematics libraries are TensorFlow and PyTorch.

>Last year, Nvidia launched its new set of RTX cards, among which RTX 3090 has **10,496** CUDA cores, combined with a boost clock of 1.70GHz and 24GB GDDR6X memory. Nvidia is labeling the RTX 3090 as a Big Ferocious GPU (BFGPU). 

### Pytorch

PyTorch is a widely used deep learning library with an easy-to-use interface for GPU acceleration. The PyTorch API is like NumPy API with added advantage coming from auto-grad API, which helps in calculating the derivative of functions. Under the hood, Pytorch creates computational graph tensors, and any operation on these tensors creates a dynamic definition of a program. On execution, the program gets compiled to GPU code in the background.

Since it is dynamic, changes to the Python code automatically get reflected in changes in the GPU code without an explicit compilation step needed. This hugely aids debugging and interactivity, as compared to static graph libraries like TensorFlow.

We’ll implement the [***diffusion equation***]() using Pytorch.

```python

import torch

from torch import (roll, zeros) #1

grid_shape = (640, 640)

def laplacian(grid):

    return (

        roll(grid, +1, 0)

        + roll(grid, -1, 0)

        + roll(grid, +1, 1)

        + roll(grid, -1, 1)

        - 4 * grid

    )

def evolve(grid, dt, D=1):

    return grid + dt * D * laplacian(grid)

def run_experiment(num_iterations):

    grid = zeros(grid_shape)

    block_low = int(grid_shape[0] * 0.4)

    block_high = int(grid_shape[0] * 0.5)

    grid[block_low:block_high, block_low:block_high] = 0.005

    grid = grid.cuda() #2

    for i in range(num_iterations):

        grid = evolve(grid, 0.1)

    return grid

```

1. We import `torch` instead of `numpy`.
2. We move the ***grid*** data to GPU, where the actual manipulation happens with the help `torch`. 

<center>

<img src=”{{site.url}}/assets/images/PythonFaster/gpu vs numpy.png" style="zoom: 5%; background-color:#DCDCDC;" width="80%" height=auto /><br>

<p>Figure 2: GPU vs NumPy</p>

</center>

This speedup is a result of how parallelizable the diffusion problem is. As we said before, the GPU we are using has 4,362 independent computation cores (GPU RTX 2080 TI). It seems that once the diffusion problem is parallelized, none of the GPU core is utilized completely. Next, we can profile GPU and see how efficiently GPU is utilized.

<center>

<img src="{{site.url}}/assets/images/PythonFaster/gpu-profile.png" style="zoom: 5%; background-color:#DCDCDC;" width="80%" height=auto /><br>

<p>Figure 3: Profiling GPU</p>

</center>

Using `nvidia-smi` command, we can inspect the resource utilization of GPU. Check out the *power usage* and *GPU utilization*. GPU utilization, here at 95%, is a slightly mislabeled field. It tells us what percentage of the last second was spent on running at least one kernel. So it isn’t telling us what percentage of the GPU’s total computational power we’re using, but rather how much time was spent not being idle. This is a very useful measurement to look at when debugging memory transfer issues and making sure that the CPU is providing the GPU with enough work.

Power usage is a good proxy for judging how much of the GPU’s compute power is being used. As a rule of thumb, the more power the GPU is drawing, the more computing it is currently doing. If the GPU is waiting for data from the CPU or using only half of the available cores, power use will be reduced from the maximum.

> *gpustat* is another useful tool for providing a better view of GPU utilization.

To find slowdown in PyTorch, we can run the code with`python -m torch.utils.bottleneck` , it will show us both CPU and GPU runtime stats and helps in identifying the potential optimizations in the code.

### Hardware Specific Optimization

One specific hardware bottleneck is the time taken for data transfer between system memory and GPU memory. When we use `tensor.to(DEVICE)`, we are triggering a transfer of data that may take some time depending on the speed of the GPU’s bus and the amount of data transferred.

When we transfer data from GPU to system memory, the code will slow down because it will pause the code in the background while the transfer is happening. There is this constant to and fro motion of data transfer which causes a major overhead in GPU parallelization. One of the biggest things slowing down PyTorch code when it comes to deep learning is copying training data from the host into the GPU. Often the training data is simply too big to fit on the GPU, and doing these constant data transfers is an unavoidable penalty.

> **There are ways to alleviate the overhead from this inevitable data transfer when the problem is going from CPU to GPU. First, the memory region can be marked as pinned. This can be done by calling the `Tensor.pin_memory()` method, which returns a copy of the CPU tensor that is copied to a “page locked” region of memory. This page-locked region can be copied to the GPU much more quickly, and it can be copied asynchronously so as to not disturb any computations being done by the GPU. While training a deep learning model, data loading is generally done with the `DataLoader` class, which conveniently has a `pin_memory` parameter that can automatically do this for all your training data.**

When the code spends most of its time doing data transfers, then we’ll see a lower power draw, with less GPU utilization (as reported by `nvidia-smi`), and most of the time spent in the `to` function (as reported in `bottleneck`). Ideally, we will be using the maximum amount of power the GPU can support and have 100% utilization. It is possible to even when large amounts of data transfer are required—even when training deep learning models with a large number of images!

### When to use GPUs

When the operation relates to matrix manipulations (like multiplication, addition, and Fourier transforms) then GPUs are a fantastic tool. It is particularly true if the calculation can happen on the GPU uninterrupted for a while before being copied back into system memory.

In addition, because of the limited memory of the GPU, it is not a suitable tool for tasks that require exceedingly large amounts of data, many conditional manipulations of the data, or dynamic data. 

The general recipe for evaluating whether to use the GPU comprises the following steps:

1. Ensure that the memory use of the problem will fit within the GPU (we explore profiling memory usage).

2. Evaluate whether the algorithm requires a lot of branching conditions versus vectorized operations. For instance, NumPy methods perform vectorizer operations well. 

3. Evaluate how much data should be transferred between GPU and CPU. Some questions to ask here are, How much computation can I do before I need to plot/save results?, are there times when my code will have to copy the data to run in a library, for which I know isn’t GPU-compatible?

4. Make sure PyTorch supports the operations you’d like to do. PyTorch implements a large portion of the NumPy API, so it should not be an issue. Mostly, the API is even the same, so we don’t need to change our code at all. 

However, there are times when PyTorch doesn't support an operation (such as dealing with complex numbers) or else the API is slightly different (for example, with generating random numbers).

### References

[High-Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)

