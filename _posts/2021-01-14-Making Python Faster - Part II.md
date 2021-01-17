## Making Python Faster - Part II

In previous blog on [vector computation in python](https://mayurji.github.io/blog/2021/01/10/vector-matrix-I), we've come across the overhead caused by list for each dereference, as the objects they reference can occur anywhere in memory. While  `arrays` are stored in contiguous block of memory, which enables faster addressing. We'll address this overhead by using numpy object along with Cython annotation.

### Cython and NumPy

In Making Python Faster - Part I blog, we used list as the object type to iterate over complex variable zs. Now, we'll use numpy array and Cython annotations to declare zs variable. 

```python
# cythonfn.pyx
import numpy as np
cimport numpy as np

def calculate_z(int maxiter, double complex[:] zs, double complex[:] cs):
    """Calculate output list using Julia update rule"""
    cdef unsigned int i, n
    cdef double complex z, c
    cdef int[:] output = np.empty(len(zs), dtype=np.int32)
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

In calculate_z function, the argument zs is a double precision complex object using the buffer protocol and this annotation is called as `memory annotation`. Note, we are also declaring a output variable as 1D numpy array as *np.empty*, the call to np.empty allocates a block of memory with no default initialization. So, there won't be any reassignment with default value, when the *output* variable is updated.

**Cython Docs**

*Typed memoryviews allow efficient access to memory buffers, such as those underlying NumPy arrays, without incurring any Python overhead. Memoryviews are similar to the current NumPy array buffer support (`np.ndarray[np.float64_t, ndim=2]`), but they have more features and cleaner syntax.*

### Cython and OpenMP

OpenMP is a Open Multi-Processing API, it supports parallel execution and memory sharing for C, C++ and Fortran. With Cython, OpenMP can be added using `prange` (parallel range) and adding the `-fopenmp` compiler directive to *setup.py*. 

While running `prange` loop, we disable python GIL (Global Interpreter Lock). The purpose of GIL is to protect python objects, preventing multiple thread or processes from accessing the same memory simultaneously, which leads to corruption. By disabling GIL, we need to ensure that we don't corrupt our memory.

```python
# cythonfn.pyx
from cython.parallel import prange
import numpy as np
cimport numpy as np

def calculate_z(int maxiter, double complex[:] zs, double complex[:] cs):
    """Calculate output list using Julia update rule"""
    cdef unsigned int i, length
    cdef double complex z, c
    cdef int[:] output = np.empty(len(zs), dtype=np.int32)
    length = len(zs)
    with nogil:
        for i in prange(length, schedule="guided"):
            z = zs[i]
            c = cs[i]
            output[i] = 0
            while output[i] < maxiter and (z.real * z.real + z.imag * z.imag) < 4:
                z = z * z + c
                output[i] += 1
    return output
```

In the above, we use `nogil` keyword, to disable the GIL and execute `prange` loop which enables an OpenMP parallel for loop to independently calculate each i.

It is recommended that after disabling GIL, we don't operate on python object such as lists etc. We must operate on primitive datatypes which supports memoryview interface. If we use python object, we may face issues related to associated memory management that GIL deliberately avoids. To run `cythonfn.pyx` , we need to modify `setup.py`. We should inform the C compiler to use -fopenmp as an argument during compilation to enable OpenMP and to link with the OpenMP Libraries.

```python
#setup.py
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

ext_modules = [Extension("cythonfn",
                         ["cythonfn.pyx"],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'])]

from Cython.Build import cythonize
setup(ext_modules=cythonize(ext_modules,
                            compiler_directives={"language_level": "3"},),
      include_dirs=[np.get_include()])
```

In `prange` loop, we can use different scheduling scheme, a `static` schedule means we evenly distribute the workload across available CPUs. Some regions of code are expensive and will take longer to execute. Some regions are manipulated faster and the thread will remain idle after execution.

Both the `dynamic` and `guided` schedule options attempt to mitigate this problem by allocating work in smaller chunks dynamically at runtime, so that the CPUs are more evenly distributed when the workload’s calculation time is variable. The correct choice will vary depending on the nature of your workload.

By introducing OpenMP and using `schedule="guided"`, we drop our execution time to approximately 0.05 seconds—the `guided` schedule will dynamically assign work, so fewer threads will wait for new work.

We also could have disabled the bounds checking for this example by using `#cython: boundscheck=False`, but it wouldn’t improve our runtime.



## Numba

Numba is a just-in-time compiler specialized for `numpy` code. It compiles the code using LLVM compiler at runtime. It provides a decorator, which points to function that needs to be taken care by Numba and it aims to run on all standard numpy code.



```python
from numba import jit

@jit
def z_serial_purepython(maxiter, zs, cs, output):
```

Above is a function from Julia code defined previously, once numba is imported, LLVM starts to complie this function behind the scenes during execution.

We have discussed about Cython and Numpy, where we provide annotations to compile the python code into C code and reduce the time taken to less than 1 second. We can achieve the similar performance by using just the `@jit` decorator on top of a function, without any annotation.

To improve the performance further, we can use the combo of OpenMP and Numba. With `prange` from OpenMP, we can drastically reduce the time taken for execution from **0.47 seconds to 0.06 seconds**. Anyway, while declaring the decorator, we should pass the argument `nopython` and `parallel`, as mentioned below

```python
@jit(nopython=False, parallel=True)
def calculate_z(maxiter, zs, cs, output):
    """Calculate output list using Julia update rule"""
    for i in prange(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and (z.real*z.real + z.imag*z.imag) < 4:
            z = z * z + c
            n += 1
        output[i] = n
```

The `nopython` specifier means that if Numba cannot compile all of the code, it will fail. Without this, Numba can silently fall back on a Python mode that is slower; your code will run correctly, but you won’t see any speedups. Adding `parallel` enables support for `prange`.

*Numba is a library provided by Continuum Analytics and it works pretty well without hassle when worked with Ananconda Distribution, otherwise, installing Numba can be time consuming*.

> There are similar tools like CPython like PyPy. It works best with python code without using any numpy code. Unlike Cython, where we provide annotations, in PyPy most speedup is done with little or no work on our part. Check [PyPy](["PyPy for Successful Web and Data Processing Systems (2014)"](https://calibre-internal.invalid/OEBPS/ch12.html#lessons-from-field-marko)) for more details.

For pure Python code, PyPy is an obvious first choice. For `numpy` code, Numba is a great first choice.

### Packages to lookout for

The [PyData compilers page](http://compilers.pydata.org) lists a set of high performance and compiler tools.

[Pythran](https://oreil.ly/Zi4r5) is an AOT compiler aimed at scientists who are using `numpy`. Using few annotations, it will compile Python numeric code to a faster binary—it produces speedups that are very similar to `Cython` but for much less work. Among other features, it always releases the GIL and can use both SIMD instructions and OpenMP. Like Numba, it doesn’t support classes. If you have tight, locally bound loops in Numpy, Pythran is certainly worth evaluating. The associated FluidPython project aims to make Pythran even easier to write and provides JIT capability.

[Transonic](https://oreil.ly/tT4Sf) attempts to unify Cython, Pythran, and Numba, and potentially other compilers, behind one interface to enable quick evaluation of multiple compilers without having to rewrite code.

[ShedSkin](https://oreil.ly/BePH-) is an AOT compiler aimed at nonscientific, pure Python code. It has no `numpy` support, but if your code is pure Python, ShedSkin produces speedups similar to those seen by PyPy (without using `numpy`). It supports Python 2.7 with some Python 3.*x* support.

[PyCUDA](https://oreil.ly/Lg4H3) and [PyOpenCL](https://oreil.ly/8e3OA) offer CUDA and OpenCL bindings into Python for direct access to GPUs. Both libraries are mature and support Python 3.4+.

[Nuitka](https://oreil.ly/dLPEw) is a Python compiler that aims to be an alternative to the usual CPython interpreter, with the option of creating compiled executables. It supports all of Python 3.7, though in our testing it didn’t produce any noticeable speed gains for our plain Python numerical tests.