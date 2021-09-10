---
layout: post
title:  Lists and Tuples
description: Mutable VS Immutable
category: Blog
date:   2020-10-02 13:43:52 +0530
---
### List and Tuples

An efficient program depends on two things, the first is the input, and the second is the data structure to process that input.

In this blog post, we’ll try to understand the data structure, List & Tuple, and the inputs it can efficiently process compared to other data structures like dict, set, etc.

List and tuples come under a class of data structure called *array*. An array is a collection of elements, and the ordering of these elements is as important as the element itself. Because to retrieve an element from a list given its position or an index takes constant time complexity O(1).

<center>
<img src="{{site.url}}/assets/images/dicts_sets/front-ds.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 1: Data Structure</p>
</center>

The list is a dynamic array. We can modify and resize the data structure as required.

A tuple is a static array whose elements are fixed and immutable. A tuple is cached at Python runtime, which means that the program doesn’t need to talk to the kernel to reserve memory every time we want to use one.

In a computer, a memory is a series of numbered buckets. Each bucket is capable of holding a number. Python stores data in these buckets by reference, which means the number itself points to or refers to the data we care about.

<center>
<img src="{{site.url}}/assets/images/lists_tuples/system_memory.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 2: Example of system memory layout for an array of size 6</p>
</center>

When we create a list or a tuple, a computer needs to allocate a block of system memory. An integer pointer references each section in the block. To look up any specific element in a list, we should know the bucket number and the element we want.

For instance, consider we have an array starting at bucket number S, now to find the 5th element in that array, we can directly search for bucket number S + 5. If an element's bucket number is not available, then we must search through all the elements of that array. The time complexity increases with an increase in the size of the array. This search is also called Linear Search. Its worst-case performance is O(n), n is the total number of elements in the list. We can efficiently search through all the elements using binary search provided the array is sorted.

For searching and sorting, python has built-in objects like __eq__, __lt__ for comparison, and the lists in python have a built-in sorting algorithm that uses Tim Sort its best-case performance is O(n), and worst-case performance is O(n log n).

Once the sorting is done, we can perform a binary search, whose average complexity is O(log n). It achieves this by first looking at the middle of the list and comparing this value with the desired value. If this midpoint’s value is less than our desired value, we consider the right half of the list, and we continue halving the list like this until the value is found, or until the value is known not to occur in the sorted list. As a result, we do not need to read all values in the list, as was necessary for the linear search; instead, we read only a small subset of them.

Note 

Check out the *bisect* module from Python standard library, which adds elements to the list, along with maintaining the sorting order.

### LISTS

The list is a dynamic array. It supports dynamic changes because of the resize operation available to it.

Consider a list A of size N if a new item is added to list A, then python under the hood creates a new list, which is large enough to hold both the N elements and the new element. So instead of allocating N + 1 space for the new list, M sized list is assigned, where M >> N. The old list of N elements is copied to the new list of size M, and the old list is deleted. It is recommended that the number of such allocations is less. To avoid such allocation frequently, python creates a new list with extra space because memory copies are expensive to maintain if the list starts growing.

*List allocation equation*

```python
M = (N >> 3) + (3 if N < 9 else 6)
```

<center>
<img src="{{site.url}}/assets/images/lists_tuples/overallocation_in_lists.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 3: Overallocation in Lists</p>
</center>

*Graph showing how many extra elements are being allocated to a list of a particular size. For example, if you create a list with 8,000 elements using appends, Python will allocate space for about 8,600 elements, over-allocating 600 elements!*

<center>
<img src="{{site.url}}/assets/images/lists_tuples/append_vs_comprehension.png" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Figure 4: Memory and time consequences of appends versus list comprehensions</p>
</center>

We use 2.7× the memory by building the list with appends versus a list comprehension. The extra space allocated to append-based insertion is a lot more compared to list apprehension.

### TUPLES

Tuples are fixed and immutable. It means once a tuple is created, it cannot be modified or resized.

**Immutable property**

```python
>>> t = (1, 2, 3, 4)
>>> t[0] = 5
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

**Concatenating property**

```python
>>> t1 = (1, 2, 3, 4)
>>> t2 = (5, 6, 7, 8)
>>> t1 + t2
(1, 2, 3, 4, 5, 6, 7, 8)
```

Now, if we consider the tuples' concatenation operation with the list’s append operation, then it's interesting to see that the time taken for tuples concatenation is O(n), while for the list is O(1). Because the list appends the elements, as long as there is extra space in the list. For tuples, every time a new element is concatenated to an existing tuple, it creates a new tuple on a different memory location, causing the concatenation to take O(n) time (as there is no in-place append-like operation in the tuple).

A tuple is a lightweight object because it takes only the memory it requires, unlike a list. It is recommended to use tuple if the data is static.

Another benefit of using tuples is resource caching. Python is garbage collected it means that if a variable isn't used anymore, then it frees its memory, giving it back to the OS for allocating that memory to other applications or variables. For tuple, if a tuples space is not used anymore, then it reserves the memory of it and if in future the memory of that size is required, python instead of reaching out to OS for system memory allocates the reserved memory. It avoids system call for a block of memory.

### Instantiation in List and Tuple

```python
>>> %timeit l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
95 ns ± 1.87 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

>>> %timeit t = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
12.5 ns ± 0.199 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)
```
Both the lists and tuples have their pros and cons. It is important to understand their properties like overallocation in the list and immutability & resource caching in the tuple.

If you've liked this post, please don't forget to subscribe to the newsletter.

### Reference

[High-Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)
