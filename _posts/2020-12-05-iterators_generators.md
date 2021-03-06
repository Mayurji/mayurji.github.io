---
layout: post
title:  Iterators and Generators in Python
description: Saving memory is an essential aspect of performant programming!
category: Blog
date:   2020-12-05 13:43:52 +0530
---

In this post, We'll discuss the python's ***Iterators and Generators*** objects and decode why generators are memory efficient and why iterators are used over generators irrespective of memory usage.

<center>
<img src="{{site.url}}/assets/images/dicts_sets/front-ds.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Data Structures</p>
</center>

### Iterators

In python, *for* loop requries an object we are looping through to support iterator function, python provides a built-in function *__iter__* to convert lists, set, dictionary, tuples into an iterator such that we can iterate over keys or items in that object. While building an iterator like list iterator, a number of functions are built like *__iter__*, *__next__*, keeping track of the states and raising exception *StopIteration* when no values are present to iterate over.

### Dismantling For loops

```python
# The Python loop
for i in object:
    do_work(i)

# Is equivalent to
object_iterator = iter(object)
while True:
    try:
        i = next(object_iterator)
    except StopIteration:
        break
    else:
        do_work(i)
```

<center>
<img src="{{site.url}}/assets/images/iterators/iterators.png" style="zoom: 5%; background-color:#DCDCDC;" width="80%" height=auto/><br>
<p>Figure 2: Iterators</p>
</center>

In the above image, once the iteration starts, the for loop iterates over all the elements one by one, processes it and then store the elements in list inside the *for loop* as shown in green circle. The processed element are made accessible after all the elements are iterated over, the unpause here refers to accessing or a call to the *__next__* function of the list.

*List iterator based fibonacci series*

```python
def fibonacci_list(num_items):
    numbers = []
    a, b = 0, 1
    while len(numbers) < num_items:
        numbers.append(a)
        a, b = b, a+b
    return numbers
```
### Generators

Generator is a function that returns an object(iterator) which we can iterate over (one value at a time). A normal function terminates with return statement, it evaluation comes out of the function, while a generator function *yields* the results of evaluation while pausing the execution, starts again with next value of iteration, during the pause state, the control is passed to the caller. Function like *__iter__*, *__next__* are implemented automatically. 

*Generator based fibonacci series*

```python
def fibonacci_gen(num_items):
    a, b = 0, 1
    while num_items:
    	yield a  
        a, b = b, a+b
        num_items -= 1
```

In Generator, we don't store the element in array for further evaluation/usage, unlike in lists, where we can reference the list anywhere, without performing iteration over all elements again.

<center>
<img src="{{site.url}}/assets/images/iterators/generator.png" style="zoom: 5%; background-color:#DCDCDC;" width="80%" height=auto/><br>
<p>Figure 3: Generator</p>
</center>

In the above image, once the generator starts, the for loop iterates over all the elements one by one and processes it and then make it accessible outside the function and then unpause by moving to next element in generator, thus avoiding memory cost.

### Memory Efficiency 

A major benefit of using a Generator is the memory saved during the iteration, because we don't store the elements anywhere after processing. For instance, consider an iteration over a million numbers, if we store the numbers in a list, it occupies hundreds of megabytes for storing it, while on the other hand, generator there is no concept of storing the items, we perform lazy evaluation when the generator is called. ***Lazy Evaluation, we don't priorly identify or evaluate all the element, we evaluate element when required.***

```python
def test_fibonacci_list():
    """
    >>> %timeit test_fibonacci_list()
    332 ms ± 13.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> %memit test_fibonacci_list()
    peak memory: 492.82 MiB, increment: 441.75 MiB
    """
    for i in fibonacci_list(100_000):
        pass


def test_fibonacci_gen():
    """
    >>> %timeit test_fibonacci_gen()
    126 ms ± 905 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    >>> %memit test_fibonacci_gen()
    peak memory: 51.13 MiB, increment: 0.00 MiB
    """
    for i in fibonacci_gen(100_000):
        pass
```

Right now, we may believe we should utilize generators everywhere to save the memory, but what if the results generated by iterator (either be list or generator) are to be referenced in future. For referencing in future, we should iterate over generator again to get the processed results again, but instead we can utilize all the processed elements stored in a list. 

Thus, it comes to down what you want optimize, ***memory consumption or CPU optimization***. Extra memory helps in storing the processed result and if the memory is a constrain, then recalculate the processed element again using generator.

*It is important to note that many of python's built-in function like range, map, zip, filter, reveresed or enumerate utilize generator in the backend, they all perform the calculations required and doesn't store the result.*

### Generator Example

**In Data Analysis**

*Consider working on a NLP based project, where we iterate over millions of text samples from a csv or a text file for text processing. If we use a generator, we might end up doing processing faster, but, without storing the processed text. We know that text processing is a computionally expensive task, and it is better to avoid generator on such occasion.*

*Anyway, while working on such huge files, we can apply filter function on each row of the text samples and interestingly, the filter function, internally runs on generator based iteration, just fetching the records with conditon (as mentioned below in example, finding numbers divisible by 3).*

**In Reducing Memory**

*Creating a list of numbers that are divisible by 3. The memory consumption is high and can be avoided by using generator.*

*List Iterator*

```python
divisible_by_three = len([n for n in fibonacci_gen(100000) if n % 3 == 0])
```

*Generator Function*

```python
divisible_by_three = sum(1 for n in fibonacci_gen(100000) if n % 3 == 0)
```

In conclusion, both the iterators and generator are applied based on requirement, as mentioned earlier the trade-off between the CPU optimization and Memory consumption. It's upto the individual, how to utilize these python objects efficiently.

### Reference

* [High Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)
