---
layout: post
title: Making Python Faster - Part III
description: If not libraries then use hardwares!
category: Blog
date:   2021-01-17 13:43:52 +0530
---
### GPUs and Pytorch

In this blog post, we'll discuss about GPU and why it is becoming a must have tool in your arsenal for massive data computation especially in the field of Deep Learning, where we require max. parallelism with matrix and vector computation.

GPU is a immensely popular tool for performing heavy arithmetic workloads. It was originally designed for gaming, generating graphics as it solves easy parallelizable problems. When compared with CPU, GPU computing rate is slower i.e. it has lower clock speed. Even though, it sounds counterintuitive, GPU gains its parallel processing power from the number of cores available. The number of cores in CPUs are at most 12 or less, but modern-day GPU comes with thousands of cores available for parallel processing.

<div>
<center>
<img src="{{site.url}}/assets/images/PythonFaster/fasterPython(1).png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Making Python Faster - Tools</p>
</center>
</div>

Anyway, it is quite tough to program for these devices, which makes it difficult to attain the maximum performance from it. However, the needs of modern deep learning algorithms have been pushing new interfaces into GPUs that are easy and intuitive to use. The two front-runners in terms of easy-to-use GPU mathematics libraries are TensorFlow and PyTorch.

>Last year, Nvidia launched its new set of RTX cards among which, RTX 3090 has **10,496** CUDA cores, combined with a boost clock of 1.70GHz and  24GB of GDDR6X memory. Nvidia is labeling the RTX 3090 a “Big Ferocious  GPU” (BFGPU). 

### Pytorch

Pytorch is a widely used deep learning library with great easy-to-use interface for GPU acceleration. The pytorch API is similar to numpy API with added advantage coming from auto-grad API, which helps in calculating the derivative of functions. Under the hood, Pytorch creates a computational graph tensors i.e. any operation on these tensors creates a dynamic definition of a program that gets compiled to GPU code in the background when it is executed.

Since it is dynamic, changes to the Python code automatically get reflected in changes in the GPU code without an explicit compilation step needed. This hugely aids debugging and interactivity, as compared to static graph libraries like TensorFlow.

We'll implement the [***diffusion equation***]() using Pytorch.

```python
import torch
from torch import (roll, zeros)  #1

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

    grid = grid.cuda()  #2
    for i in range(num_iterations):
        grid = evolve(grid, 0.1)
    return grid
```

#1 is where we import `torch` instead of `numpy`, and #2 we move the ***grid*** data to GPU for where actual manipulation will happen with the help `torch`. 

<center>
<img src="{{site.url}}/assets/images/PythonFaster/gpu vs numpy.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto /><br>
<p>Figure 2: GPU vs NumPy</p>
</center>

This speedup is a result of how parallelizable the diffusion problem is. As we said before, the GPU we are using has 4,362 independent computation cores (GPU RTX 2080 TI). It seems that once the diffusion problem is parallelized, none of these GPU cores are being fully utilized. Next, we can profile GPU and see how efficiently GPU is utilized.

<center>
<img src="{{site.url}}/assets/images/PythonFaster/gpu-profile.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto /><br>
<p>Figure 3: Profiling GPU</p>
</center>

Using `nvidia-smi` command, we can inspect the resource the utilization of the GPU. Check out the *power usage* and *GPU utilization*. GPU utilization, here at 95%, is a slightly mislabeled field. It tells us what percentage of the last second has been spent running at least one kernel. So it isn’t telling us what percentage of the GPU’s total computational power we’re using but rather how much time was spent *not* being idle. This is a very useful measurement to look at when debugging memory transfer issues and making sure that the CPU is providing the GPU with enough work.

Power usage, on the other hand, is a good proxy for judging how much of the GPU’s compute power is being used. As a rule of thumb, the more power the GPU is drawing, the more compute it is currently doing. If the GPU is waiting for data from the CPU or using only half of the available cores, power use will be reduced from the maximum.

> *gpustat* is another useful tool for providing better view of GPU utilization.

To find slowdown in pytorch, we can run the code with`python -m torch.utils.bottleneck` , it will show us both CPU and GPU runtime stats and helps in identifying the potential optimizations in the code.

### Hardware Specific Optimization

One specific hardware bottleneck is the time taken for data transfer between system memory and GPU memory. When we use `tensor.to(DEVICE)`, we are triggering a transfer of data that may take some time depending on the speed of the GPU’s bus and the amount of data being transferred.

When we transfer data from GPU to system memory, the code will slow down, because it will pause the code in background while the transfer is happening. There is this constant to and fro motion of data transfer which causes major overhead in GPU parallelization. In fact, one of the biggest things slowing down PyTorch code when it comes to deep learning is copying training data from the host into the GPU. Often the training data is simply too big to fit on the GPU, and doing these constant data transfers is an unavoidable penalty.

> **There are ways to alleviate the overhead from this inevitable data transfer when the problem is going from CPU to GPU. First, the memory region can be marked as `pinned`. This can be done by calling the `Tensor.pin_memory()` method, which returns a copy of the CPU tensor that is copied to a “page locked” region of memory. This page-locked region can be copied to the GPU much more quickly, and it can be copied asynchronously so as to not disturb any computations being done by the GPU. While training a deep learning model, data loading is generally done with the `DataLoader` class, which conveniently has a `pin_memory` parameter that can automatically do this for all your training data.**

When the code is spending most of its time doing data transfers, we'll see a low power draw, a smaller GPU utilization (as reported by `nvidia-smi`), and most of the time being spent in the `to` function (as reported by `bottleneck`). Ideally, we will be using the maximum amount of power the GPU can support and have 100% utilization. This is possible even when large amounts of data transfer are required—even when training deep learning models with a large number of images!

### When to use GPUs

If the task requires us to calculate mainly linear algebra and matrix manipulations (like multiplication, addition, and Fourier transforms), then GPUs are a fantastic tool. This is particularly true if the calculation can happen on the GPU uninterrupted for a period of time before being copied back into system memory.

In addition, because of the limited memory of the GPU, it is not a good tool for tasks that require exceedingly large amounts of data, many conditional manipulations of the data, or changing data. Most GPUs made for computational tasks have around 12 GB of memory, which puts a significant limitation on “large amounts of data.” However, as technology improves, the size of GPU memory increases, so hopefully this limitation becomes less drastic in the future.

The general recipe for evaluating whether to use the GPU consists of the following steps:

1. Ensure that the memory use of the problem will fit within the GPU (we explore profiling memory use).
2. Evaluate whether the algorithm requires a lot of branching conditions versus vectorized operations. As a rule of thumb, `numpy` functions generally vectorize very well, so if the algorithm can be written in terms of `numpy` calls, then we can be sure that the code probably will vectorize well! 
3. Evaluate how much data needs to be moved between the GPU and the CPU. Some questions to ask here are “How much computation can I do before I need to plot/save results?” and “Are there times my code will have to copy the data to run in a library I know isn’t GPU-compatible?”
4. Make sure PyTorch supports the operations you’d like to do! PyTorch implements a large portion of the `numpy` API, so this shouldn’t be an issue. For the most part, the API is even the same, so we don’t need to change our code at all. However, in some cases either PyTorch doesn’t support an operation (such as dealing with complex numbers) or the API is slightly different (for example, with generating random numbers).

### References

[High Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)

