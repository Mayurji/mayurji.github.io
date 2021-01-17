---
layout: post
title: Making Python Faster - Part I
description: Compile it down!
category: Blog
date:   2021-01-09 13:43:52 +0530
---
{% include mathjax.html %}

To make code run faster, a number of things can be done like reducing the number of preprocessing steps or compile the code down to machine code or use a machine which has high clock speed. 

Python offers many options to perform efficient compiling like pure C compilers, Cython and LLVM-based compiling via Numba or a replacement virtual machine PyPy, which has Just-In-Time Compiler. 

* Cython, the most commonly used tool for compiling to C, covering both `numpy` and normal Python code (requires some knowledge of C)
* Numba, a new compiler specialized for `numpy` code
* PyPy, a stable just-in-time compiler for non-`numpy` code that is a replacement for the normal Python executable

Sometimes even compiled code will not bring in greater gains, for instance, if the code requires to call out different external libraries like string operation with regex, database calls, or programs with I/O operations, etc. Python code that tends to run faster after compiling is mathematical, and it has lots of loops that repeat the same operations many times. Inside these loops, you’re probably making lots of temporary objects.

<center>
<img src="{{site.url}}/assets/images/PythonFaster/compiler.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" width="1000" height="600"/><br>
<p>Figure 1: Compilers</p>
</center>

### JIT and AOT

Just-in-time compiler, compiles just the right parts of the code at the time of use. If a code is small but called frequently, then JIT runs it slowly while it compiles. While AOT (ahead of time) compiler, it compiles all libraries required for execution ahead of time using Cython , which makes it faster in some cases as mentioned earlier.

The current state of affairs shows us that compiling ahead of time buys us the best speedups, but often this requires the most manual effort. Just-in-time compiling offers some impressive speedups with very little manual intervention, but it can also run into the problem just described. You’ll have to consider these trade-offs when choosing the right technology for your problem.

### Python Overhead

Python is a dynamically typed language -- a variable can refer to an object of any type and any line of code can change the type of the object that is referred to. It makes it difficult for machine code to understand such changes of variable type. 

Consider a variable v,  at first, it is declared as floating point number and next it is declared as complex number,

```python
v = -1.0
print(type(v), abs(v))
<class 'float'> 1.0
v = 1-1j
print(type(v), abs(v))
<class 'complex'> 1.4142135623730951
```
When we perform an ***abs*** on v as floating number, we find the absolute value of -1 as 1. And performing ***abs*** on complex number leads to different formulation as follows 
<p>
$$ abs (c) = \sqrt {c.real^2 + c.imaginary^2} $$
</p>
Before calling `abs` on a variable, Python first has to look up the type of the variable and then decide which version of a function to call—this overhead adds up when you make a lot of repeated calls. 

#### Sample Code

We'll check out sample code (Julia set) for our compilation, which I've already mentioned in one of the [previous blog](https://mayurji.github.io/blog/2021/01/02/profiling).  Its a CPU bound problem and helps in understanding the potential bottleneck in the code.

```python
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output
```

### Cython

[Cython](http://cython.org) is a compiler that converts type-annotated Python into a compiled extension module. The type annotations are C-like. This extension can be imported as a regular Python module using `import`. Getting started is simple, but a learning curve must be climbed with each additional level of complexity and optimization. Cython can be used via a *setup.py* script to compile a module. It can also be used interactively in IPython via a “magic” command. Typically, the types are annotated by the developer, although some automated annotation is possible.

#### Compiling Pure Python from Cython

Three files are required for writing compiled version of the sample code. 

* Calling Python code. (.py file)
* Function to be compiled in a new ***.pyx*** file.
* setup.py, contains instruction.

```python
# filename1: cythonfn.pyx
def calculate_z(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output
```

```python
# filename2:  julia.py
import cythonfn  # as defined in setup.py
...
def calc_pure_python(desired_width, max_iterations):
    # ...
    start_time = time.time()
    output = cythonfn.calculate_z(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(f"Took {secs:0.2f} seconds")
```

```python
# filename3: setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("cythonfn.pyx",
                            compiler_directives={"language_level": "3"})) # 3 represent to force python3.x support 
```

For the Julia example, we’ll use the following:

- *julia.py* to build the input lists and call the calculation function
- *cythonfn.pyx*, which contains the CPU-bound function that we can annotate
- *setup.py*, which contains the build instructions

setup.py is called to compile the .pyx file using Cython into compiled module. On Unix-like systems, the compiled module will probably be a *.so* file; on Windows it should be a *.pyd* (DLL-like Python library). 

setup.py converts the ***.pyx*** to ***.so*** file.

```python
$ python setup.py build_ext --inplace
Compiling cythonfn.pyx because it changed.
[1/1] Cythonizing cythonfn.pyx
running build_ext
building 'cythonfn' extension
gcc -pthread -B /home/ian/miniconda3/envs/high_performance_python_book_2e/...
gcc -pthread -shared -B /home/ian/miniconda3/envs/high_performance_python_...
```

The `--inplace` argument tells Cython to build the compiled module into the current directory rather than into a separate *build* directory. After the build has completed, we’ll have the intermediate *cythonfn.c*, which is rather hard to read, along with *cythonfn[…].so*.

Now when the *julia.py* code is run, the compiled module is imported, and the Julia set is calculated, it takes 4.7 seconds, rather than the more usual 8.3 seconds. This is a useful improvement for very little effort.

To avoid using setup.py completely, we can use ***pyximport*** library. In ***julia.py*** file, we can import the library and to do an install statement as mentioned,  in the code below. It will create the compiled version of ***.pyx*** file mentioned below the install statement.

```python
import pyximport
pyximport.install(language_level=3)
import cythonfn
# followed by the usual code
```

### Understanding Cython Annotation

Lets view the intermediate `cythonfn.c` file, we can type `cython -a cythonfn.pyx` , and generate `cythonfn.html` file. We cannot optimize any block of code blindly, that's why we can check if line acts as bottleneck. Once the `html` file is generated, we can view it in a browser.

<center>
<img src="{{site.url}}/assets/images/PythonFaster/shrinked_cython.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 2: Identifying the Bottleneck in Code</p>
</center>

#### Cython Annotation

Each line can be expanded with a double-click to show the generated C code. More yellow means “more calls into the Python virtual machine,” while more white means “more non-Python C code.” The goal is to remove as many of the yellow lines as possible and end up with as much white as possible.

<center>
<img src="{{site.url}}/assets/images/PythonFaster/expanded_cython.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 3: Under The Hood: Cython</p>
</center>

Although “more yellow lines” means more calls into the virtual machine, this won’t necessarily cause your code to run slower. Each call into the virtual machine has a cost, but the cost of those calls will be significant only if the calls occur inside large loops. Calls outside large loops (for example, the line used to create `output` at the start of the function) are not expensive relative to the cost of the inner calculation loop. 

In above example, the lines with the most calls back into the Python virtual machine (the “most yellow”) are lines 4 and 8. From previous [profiling blog](https://mayurji.github.io/blog/2021/01/02/profiling), we know that line 8 is likely to be called over 30 million times, so that’s a great candidate to focus on.

To improve the execution time, the inner loops needs to be declared with objects type rather than leaving it for inference while running. These loops can then make fewer of the relatively expensive calls back into the Python virtual machine, saving us time. 

In general, the lines that probably cost the most CPU time are those:

- Inside tight inner loops
- Dereferencing `list`, `array`, or `np.array` items
- Performing mathematical operations

#### Adding Type Annotations

We can see from the above images, that almost all the lines are yellow, meaning they are accessing python virtual machine. The code utilizes the high level python objects. To reduce the overhead of type reference, we can declare C based type reference in the code.

* int for a signed integer
* unsigned int for an integer that can only be positive
* double complex for double precision complex numbers

```python
# Adding primitive C types to start making our compiled function run faster by doing more work in C and less via the Python virtual machine

def calculate_z(int maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    cdef unsigned int i, n
    cdef double complex z, c
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output
```
<center>
<img src="{{site.url}}/assets/images/PythonFaster/cython_annotations.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;"/><br>
<p>Figure 4: Cython Annotations</p>
</center>

*The `cdef` keyword lets us declare variables inside the function body. These changes are made in .pyx file*

The benefits of giving annotation to variable is visible from the above images, critically,  we can see that line no. 11 and 12—**two of the most frequently called lines—have now turned from yellow to white, indicating that they no longer call back to the Python virtual machine.** We can anticipate a great speedup compared to the previous version.

After compiling, this version takes 0.49 seconds to complete. With only a few changes to the function, we are running at 15 times the speed of the original Python version.

In previous example, we saw that ***abs*** on complex number leads to square root of the sum of the squares of the real and imaginary components. In above example, we can perform few changes such that instead of calculating the square root, we can square on both sides, and avoid square root operation as follows
<p>
$$ \sqrt{c.real^2 + c.imaginary^2} < \sqrt{4}  \to {c.real^2 + c.imaginary^2} < {4} $$
</p>
*square root is an expensive operation!*

```python
def calculate_z(int maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    cdef unsigned int i, n
    cdef double complex z, c
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and (z.real * z.real + z.imag * z.imag) < 4:
            z = z * z + c
            n += 1
        output[i] = n
    return output
```

The `while` statement may look like we are doing more work than usual but its not immediately visible, how much of a speed gain it brings. Since we know from previous blog post on profiling, that this line is iterated over 30 million times.

This change has a dramatic effect—by reducing the number of Python calls in the innermost loop, we greatly reduce the calculation time of the function. This new version completes in just 0.19 seconds, an amazing 40× speedup over the original version. As ever, take a guide from what you see, but *measure* to test all of your changes!

*Mathematical calculations are faster in general when reduced to low level functions!*

Under the hood, there is something called as bound checking for each dereference in the list. The goal of the bounds checking is to ensure that the program does not access data outside the allocated array—in C it is easy to accidentally access memory outside the bounds of an array, and this will give unexpected results (and probably a segmentation fault!). We can disable bound checking if required.

Cython has a set of flags that can be expressed in various ways. The easiest is to add them as single-line comments at the start of the *.pyx* file. It is also possible to use a decorator or compile-time flag to change these settings. To disable bounds checking, we add a directive for Cython inside a comment at the start of the *.pyx* file:

```python
#cython: boundscheck=False
def calculate_z(int maxiter, zs, cs):
```

​																																..................

*From **Wikipedia***

*In computer programming, **bounds checking** is any method of detecting whether a variable is within some **bounds** before it is used. It is usually used to ensure that a number fits into a given type (range **checking**), or that a variable being used as an array index is within the **bounds** of the array (index **checking**).*

​																																..................

