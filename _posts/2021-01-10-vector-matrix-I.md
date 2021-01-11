---
layout: post
title: Vector Computation in Python
description: Do it Parallelly!
category: Blog
date:   2021-01-10 13:43:52 +0530
---
To understand and work better with computationally extensive task requires us to understand the working of python under the hood. In this post, we'll discuss and understand, how python interacts at the system level, to identify the bottlenecks and how to overcome it. Unsurprisingly, vector computation plays a key role in making the system run faster and we'll see, how different python codes affects the CPU performance and how can we effectively reduce the cost of performance. And at the end, we'll see Numpy, numexpr tools for faster computation.

<center>
<img src="{{site.url}}/assets/images/dicts_sets/front-ds.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" width="1000" height="600"/><br>
<p>Figure 1: Data Structures</p>
</center>

### How python moves in system

In python, a variable declaration leads to creating a reference to the variable through pointers. It helps python create variable without knowing the datatype before hand, since python only holds the address irrespective of where it points to. The performance degradation happens when we want to access a stored element. 

*For example, doing mat [5][2] requires us to first do a list lookup for index 5 on the list mat. This will return a pointer to where the data at that location is stored. Then we need to do another list lookup on this returned object, for the element at index 2. Once we have this reference, we have the location where the actual data is stored.*

*Here, multiple lookups happen for accessing the actual data, meaning the data is presented in fragmented order and making more memory transfer overhead thus forcing CPU to wait for this transfer to happen. If we have infinitely fast bandwidth between the RAM and CPU, then we would not need CPU caches to store anything, just the instruction comes in and does its actions and leaves, but sorry to say, we don't have infinitely fast bandwidth yet.*

### Sample Code

Diffusion Equation refers to calculating the change in a system under certain circumstances over a period of time. For instance, consider we are trying to mix a dye with water, then we calculate, how long will it take to diffuse these two separate matter completely. The system remains in multiple states before it reaches its final destination. Initial state is purely water 0 and on the other hand, we have dye which is 1. And in between states is where value between 0 to 1 exists.

### Diffusion Equation Code

```python
grid_shape = (640, 640)

def evolve(grid, dt, D=1.0):
    xmax, ymax = grid_shape
    new_grid = [[0.0] * ymax for x in range(xmax)]
    for i in range(xmax):
        for j in range(ymax):
            grid_xx = (
                grid[(i + 1) % xmax][j] + grid[(i - 1) % xmax][j] - 2.0 * grid[i][j]
            )
            grid_yy = (
                grid[i][(j + 1) % ymax] + grid[i][(j - 1) % ymax] - 2.0 * grid[i][j]
            )
            new_grid[i][j] = grid[i][j] + D * (grid_xx + grid_yy) * dt
    return new_grid

def run_experiment(num_iterations):
    # Setting up initial conditions 
    xmax, ymax = grid_shape
    grid = [[0.0] * ymax for x in range(xmax)]

    # These initial conditions are simulating a drop of dye in the middle of our
    # simulated region
    block_low = int(grid_shape[0] * 0.4)
    block_high = int(grid_shape[0] * 0.5)
    for i in range(block_low, block_high):
        for j in range(block_low, block_high):
            grid[i][j] = 0.005

    # Evolve the initial conditions
    start = time.time()
    for i in range(num_iterations):
        grid = evolve(grid, 0.1)
    return time.time() - start
```
* run_experiment is executed only once.
* evolve method is executed based on number of iterations required.
* evolve method tracks the evolution of the system in place.
* grid is current system and new_grid is the updated system after each iteration.
* grid is a lists inside list, i.e. similar to matrices.
* change in system happens at dt=0.1
* initialisation of the system is different at middle of the water than at other region of the system.

In previous blog post, [python profiling](https://mayurji.github.io/blog/2021/01/02/profiling) where I mentioned about line_profiler library, which measures the time taken by every line in a function, check it out for more details. 

Now, we can run a line_profiler on top of this code to check the time taken for each line to execute.

<center>
<img src="{{site.url}}/assets/images/dicts_sets/diffusion_lr.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 2: Line Profile on Diffusion Equation</p>
</center>

We can notice in *per hit* column, the time taken for line 15 to execute is far more than other lines. If we make 500 calls to evolve function, the line 15 will still do the same assignment of 0.0 throughout the lists of list. So we can declare such assignment or dummy declaration of grid in parent function (run_experiment) to avoid such overhead of memory allocation each time we call evolve function.

```python
def evolve(grid, dt, out, D=1.0):
    xmax, ymax = grid_shape
    for i in range(xmax):
        for j in range(ymax):
            grid_xx = (
                grid[(i + 1) % xmax][j] + grid[(i - 1) % xmax][j] - 2.0 * grid[i][j]
            )
            grid_yy = (
                grid[i][(j + 1) % ymax] + grid[i][(j - 1) % ymax] - 2.0 * grid[i][j]
            )
            out[i][j] = grid[i][j] + D * (grid_xx + grid_yy) * dt


def run_experiment(num_iterations):
    # Setting up initial conditions
    xmax, ymax = grid_shape
    next_grid = [[0.0] * ymax for x in range(xmax)]
    grid = [[0.0] * ymax for x in range(xmax)]

    block_low = int(grid_shape[0] * 0.4)
    block_high = int(grid_shape[0] * 0.5)
    for i in range(block_low, block_high):
        for j in range(block_low, block_high):
            grid[i][j] = 0.005
            start = time.time()
    for i in range(num_iterations):
        # evolve modifies grid and next_grid in-place
        evolve(grid, 0.1, next_grid)
        grid, next_grid = next_grid, grid
    return time.time() - start
```
<center>
<img src="{{site.url}}/assets/images/dicts_sets/diffusion_lr_2.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 3: Line Profile on optimized Diffusion Equation</p>
</center>

After creating the next_grid in run_experiment function as mentioned above, the modified version of the speed increases by 31.25% which makes it clear that memory allocation is an expensive task.

### Perf

It is a tool to find the performance metrics of the code under scrutiny. We'll apply this tool to check the performance of the sample code above.

<center>
<img src="{{site.url}}/assets/images/dicts_sets/perfs.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 4: Perf's Insights</p>
</center>

*task-clock:* how many clock cycles did our task took. If 1 second is the total run time of the program with 2 CPUs, then our task-cycle is 2000ms.\
*instruction:* the number of CPU instructions our code as issued.\
*cycles:* number of cycles to run all the instruction.\
*context-switching(cs):* program halting during kernel i/o operation or moving to another cpu core.\
*context-migration:* program halted and resumed on different cpu than it was on before, in order to have same level of utilization.\
*fault:* its an interrupt given to program when the memory on which its running is filled, its a lazy allocation system.\
*cache-references:* when data is moved across system, it traverse through l1/l2/l3 cache. So anyway, when we refer data that is present in cache, its called cache-references.\
*cache-misses:* when data is not present in cache and had to be loaded from RAM then its called cache-misses.\
*branch:* when execution flow changes, like if and else statements. CPU predicts which branch will execute next and loads the instruction.\
*branch-miss:* wrong prediction of branch from CPU leads to branch-miss.\

Reading through an array in order will give many cache-references but not many cache-misses since if we read element i, element i + 1 will already be in cache. If, however, we read randomly through an array or otherwise don’t layout our data in memory well, every read will require access to data that couldn’t possibly already be in cache.

### Why pure python sucks at Vectorization

In second paragraph, I've mentioned that python keeps reference to actual data through pointers and these pointers are randomly allocated in the memory, which is nothing but fragmentation of data or a poor layout of data on memory. Cache-references happen when the current element and potential next element is present in cache, since the data is fragmented everywhere in memory, we cannot have cache-references and performing vectorization requires immediate next set of relevant data for computation to be in cache. 

Thus, vectorization of computation can happen only when all relevant data is present in order or in sequence. Since the bus can only move contiguous chunks of memory, this is possible only if the grid data is stored sequentially in RAM. Since a list stores pointers to data instead of the actual data, the actual values in the grid are scattered throughout memory and cannot be copied all at once. It makes memory allocation overhead, which keeps the CPU idle while the data is transferred. To overcome this transfer cost, comes Numpy.

### Numpy - The Savior

Numpy arrays store the data in contiguous chunks of memory and support vectorized operation on its data. As a result, all the arithmetic operation happen on chunks of memory rather than on individual element. Find a list of comparison between array, list and Numpy array.

```python
from array import array
import numpy


def norm_square_list(vector):
    """
    >>> vector = list(range(1_000_000))
    >>> %timeit norm_square_list(vector)
    85.5 ms ± 1.65 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    norm = 0
    for v in vector:
        norm += v * v
    return norm


def norm_square_list_comprehension(vector):
    """
    >>> vector = list(range(1_000_000))
    >>> %timeit norm_square_list_comprehension(vector)
    80.3 ms ± 1.37 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    return sum([v * v for v in vector])


def norm_square_array(vector):
    """
    >>> vector_array = array('l', range(1_000_000))
    >>> %timeit norm_square_array(vector_array)
    101 ms ± 4.69 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    norm = 0
    for v in vector:
        norm += v * v
    return norm


def norm_square_numpy(vector):
    """
    >>> vector_np = numpy.arange(1_000_000)
    >>> %timeit norm_square_numpy(vector_np)
    3.22 ms ± 136 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    return numpy.sum(vector * vector)  


def norm_square_numpy_dot(vector):
    """
    >>> vector_np = numpy.arange(1_000_000)
    >>> %timeit norm_square_numpy_dot(vector_np)
    960 µs ± 41.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    """
    return numpy.dot(vector, vector)
```

Numpy based sum of square (norm_square_numpy), it performs iteration over two loops, one for vector multiplication and then summing over the squared vectors. It is similar to norm_square_list_comprehension. A more much efficient practice is to use *numpy.dot* to do vector norms. 

The reason why Numpy is faster, is because it is built on top of native C code, a finely tuned & specially built object for dealing with array of numbers. It takes any vectorization advantage that the CPU is enabled with. *numpy.dot* outperforms all other variants of numpy and pure python codes by a great margin because, it doesn't store the intermediate value vector * vector operation in memory.

<center>
<img src="{{site.url}}/assets/images/dicts_sets/numpy_dot.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" width="1000" height="600"/><br>
<p>Figure 5: Numpy's Dot vs Rest</p>
</center>

### Numpy - Diffusion Equation

In Diffusion equation, we perform constant change in grid index as we iterate over xmax and ymax. With the help of numpy, we can vectorize this calculation and use *roll* function from numpy to perform reindexing in grid.

```python
>>> import numpy as np
>>> np.roll([1,2,3,4], 1)
array([4, 1, 2, 3])

>>> np.roll([[1,2,3],[4,5,6]], 1, axis=1)
array([[3, 1, 2],
       [6, 4, 5]])
```

The roll function creates a new numpy array, which can be thought of as both good and bad. The downside is that we are taking time to allocate new space, which then needs to be filled with the appropriate data. On the other hand, once we have created this new rolled array, we will be able to vectorize operations on it quite quickly without suffering from cache misses from the CPU cache. This can substantially affect the speed of the actual calculation we must do on the grid.

```python
from numpy import (zeros, roll)

grid_shape = (640, 640)


def laplacian(grid):
    return (
        roll(grid, +1, 0) +
        roll(grid, -1, 0) +
        roll(grid, +1, 1) +
        roll(grid, -1, 1) -
        4 * grid
    )


def evolve(grid, dt, D=1):
    return grid + dt * D * laplacian(grid)


def run_experiment(num_iterations):
    grid = zeros(grid_shape)

    block_low = int(grid_shape[0] * 0.4)
    block_high = int(grid_shape[0] * 0.5)
    grid[block_low:block_high, block_low:block_high] = 0.005

    start = time.time()
    for i in range(num_iterations):
        grid = evolve(grid, 0.1)
    return time.time() - start
```

Let's do a perf on this code

<center>
<img src="{{site.url}}/assets/images/dicts_sets/perf_numpy_diffusion.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 6: Perf's Numpy Code</p>
</center>

There is a 63.3x speedup in numpy code over pure python implementation. The total number of cycles and instruction is reduced, informing us about the vectorized operation rather than individual operation of an element. Important factor to note is the cache-misses, in pure-python we saw a 53.259% cache-misses and it is brought down to 20.8%, meaning the CPU was not idle and the relevant data was available in cache for CPU to manipulate.

*Note, the speedup by 63x is not majorly contributed by vectorization but by memory locality and reduced memory fragmentation. This realization that memory issues are the dominant factor in slowing down our code doesn’t come as too much of a shock. Computers are very well designed to do exactly the calculations we are requesting them to do with this problem—multiplying and adding numbers together. The bottleneck is in getting those numbers to the CPU fast enough to see it do the calculations as fast as it can.*

### In-place operation

Memory allocation is further improved with the help of In-place operation. We preallocate some memory at the beginning of the code and perform in-place operation to manipulate the vectors or array using *\*=* and *\+=*. We use *id* keyword to find the memory being referenced. In the below code, we have two arrays, a1 and a2 preallocated with size (10,10). When we do in-place operation, the memory reference of a1 remains same (1==2). When we do the assignment operation after addition, we allocate new memory references for the added arrays(1 != 3).

```python
>>> import numpy as np
>>> a1 = np.random.random((10,10))
>>> a2 = np.random.random((10,10))
>>> id(a1) 
140199765947424  #--> 1
>>> a1 += a2
>>> id(a1) 
140199765947424  #--> 2
>>> a1 = a1 + a2
>>> id(a1)
140199765969792  #--> 3
```
*Runtime differences between in-place and out-of-place operation*

We can clearly see, the overhead caused by the assignment operation. Anyway, this overhead is visible only when we have 100x100 elements i.e. when size is greater than CPU cache. For 2x5x5, we see the assignment is faster than in-place because the cache is able to hold 2x5x5.

```python
>>> import numpy as np

>>> %%timeit array1, array2 = np.random.random((2, 100, 100)) 
... array1 = array1 + array2
6.45 µs ± 53.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

>>> %%timeit array1, array2 = np.random.random((2, 100, 100))  
... array1 += array2
5.06 µs ± 78.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

>>> %%timeit array1, array2 = np.random.random((2, 5, 5))  
... array1 = array1 + array2
518 ns ± 4.88 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

>>> %%timeit array1, array2 = np.random.random((2, 5, 5))  
... array1 += array2
1.18 µs ± 6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

We'll not discuss, in-place operation for diffusion equation, as we can identify from above the changes brought by in-place operation.'

### numexpr

One downfall of `numpy`’s optimization of vector operations is that it occurs on only one operation at a time.  That is to say, when we are doing the operation `A * B + C` with `numpy` vectors, first the entire `A * B` operation completes, and the data is stored in a temporary vector; then this new vector is added with `C`.

numexpr is a module that can take entire vector expression and compile it into a optimized code to minimize the cache-misses and the temporary memory used. It utilizes multiple cores and takes advantage of specialized instruction that a CPU support for better speedup. It support OpenMP, which parallels out operations across multiple cores on your machine.

```python
from numexpr import evaluate

def evolve(grid, dt, next_grid, D=1):
    laplacian(grid, next_grid)
    evaluate("next_grid * D * dt + grid", out=next_grid)
```

It is very easy to change code to use `numexpr`: all that’s required is to rewrite the expressions as strings with references to local variables.  The expressions are compiled behind the scenes (and cached so that calls to the same expression don’t incur the same cost of compilation) and run using optimized code.

In the above case, we chose to use the `out` parameter of the `evaluate` function so that `numexpr` doesn’t allocate a new vector to which to return the result of the calculation.

***The key aspect of numexpr is the consideration of CPU caches. It moves data around so that the various CPU caches have the correct data in order to minimize cache misses.***

<center>
<img src="{{site.url}}/assets/images/dicts_sets/numexpr.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 7: Numpexpr</p>
</center>

Much of the extra machinery we are bringing into our program with `numexpr` deals with cache considerations.  When our grid size is small and all the data we need for our calculations fits in the cache, this extra machinery simply adds more instructions that don’t help performance.  In addition, compiling the vector operation that we encoded as a string adds a large overhead.  When the total runtime of the program is small, this overhead can be quite noticeable. However, as we increase the grid size, we should expect to see `numexpr` utilize our cache better than native `numpy` does.  In addition, `numexpr` utilizes multiple cores to do its calculation and tries to saturate each of the cores’ caches.  When the size of the grid is small, the extra overhead of managing the multiple cores overwhelms any possible increase in speed.

In conclusion, there are various tools and tips for better utilization of memory and RAM. Follow a process to tackle issues of performance by profiling and solve each issue one by one and keep track how the readability and maintainble of the code is affected after each changes is. Always write a unit test, if the code complexity increases, you'll never regret spending sometime on it.


