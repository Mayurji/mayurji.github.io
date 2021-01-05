---
layout: post
title:  Lists and Tuples in Python
description: Finding the right data structure is an art!
category: Blog
image_url: "/assets/images/lists_tuples/lists_tuples.png"
date:   2020-10-02 13:43:52 +0530
---
Writing efficient programs involves understanding a couple of things, first, what is the input to the program and second is selecting the best data structure to process that input.

In this blog post, we’ll try to understand the data structure, List & Tuple and the inputs it can efficiently process compared to other data structure like dict, set etc.

List and tuples comes under a class of data structure called *array*. Array is a collection of elements, and ordering or positioning of these element is as important as element itself. Because to retrieve an element, given a position or index, it takes O(1) time complexity, to find a element.

<center>
<img src="{{site.url}}/assets/images/dicts_sets/dicts_sets.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" width="1000" height="600"/><br>
<p>Figure 1: Data Structure</p>
</center>

* List is a dynamic array, where we can modify and resize the data we are storing in it.

* Tuple is a static array, whose elements are fixed and immutable. Tuples are cached by the Python runtime, which means that we don’t need to talk to the kernel to reserve memory every time we want to use one.

In a computer system, the memory is a series of numbered buckets, each capable of holding a number. Python stores data in these buckets by reference, which means the number itself simply points to, or refers to, the data we actually care about.

<center>
<img src="{{site.url}}/assets/images/lists_tuples/system_memory.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p>Figure 2: Example of system memory layout for an array of size 6</p>
</center>

When we create a list or a tuple, we need to allocate a block of system memory, every section of that block is referenced using an integer pointer. In order to look up any specific element in a list, we should know the bucket number and the element we want.

For instance, consider we have an array starting at bucket number, S, to find the 5th element in that array, we can directly search for bucket number S + 5, similarly for all the i element in an array. But if the bucket number is not available for the given array, then we need to perform a search of element throughout the array, the time complexity increases with increase in size of the array. This search is also called Linear Search. It’s worst case performance is O(n), n is total number of element in the list. Other efficient searching algorithms can be applied for searching a element in an array such as binary search algorithm provided the list is sorted.

For searching and sorting, python has built in objects like __eq__, __lt__ for comparison and the lists in python have a built in sorting algorithm that uses Tim Sort, its best case performance is O(n) and worst case performance is O(n log n).

Once the sorting is done, we can perform binary search, whose average complexity is O(log n). It achieves this by first looking at the middle of the list and comparing this value with the desired value. If this midpoint’s value is less than our desired value, we consider the right half of the list, and we continue halving the list like this until the value is found, or until the value is known not to occur in the sorted list. As a result, we do not need to read all values in the list, as was necessary for the linear search; instead, we read only a small subset of them.

Note: Checkout *bisect* module from Python standard library, which adds elements to the list, along with maintaining the sorting order.

### LISTS

List is a dynamic array, its support dynamic changes because of the resize operation available to it.

Consider a list A of size N, if a new item is appended to list A, then python creates a new list, which is large enough to hold N element and the new element. So instead of allocation N + 1 items, M items are allocated, M > N. The old list is copied to new list and old list is deleted or destroyed. It is recommended that this number of allocation should be reduced by requesting extra space while allocation is done the first time. Since memory copies are expensive to maintain, if the list starts growing.

*List allocation equation*

```python
M = (N >> 3) + (3 if N < 9 else 6)
```

<center>
<img src="{{site.url}}/assets/images/lists_tuples/overallocation_in_lists.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p>Figure 3: Overallocation in Lists</p>
</center>

*Graph showing how many extra elements are being allocated to a list of a particular size. For example, if you create a list with 8,000 elements using appends, Python will allocate space for about 8,600 elements, overallocating 600 elements!*

<center>
<img src="{{site.url}}/assets/images/lists_tuples/append_vs_comprehension.png" class="post-body" style="zoom: 5%; background-color:#DCDCDC;" /><br>
<p>Figure 4: Memory and time consequences of appends versus list comprehensions</p>
</center>

We use 2.7× the memory by building the list with appends versus a list comprehension. The extra space allocated to append based insertion is lot more compared to list apprehension.

### TUPLES

Tuples are fixed and immutable. This means that once a tuple is created, unlike a list, it cannot be modified or resized.

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

Now, if we consider the concatenation operation with the list’s append operation, then its interesting to see that the time taken for tuples concatenation is O(n), while for the list is O(1). Because the list appends the elements, as long as there is extra space in the list. For tuples, every time a new element is concatenated to existing tuple, it creates a new tuple on a different memory location, causing the concatenation to take O(n) time (as there is no in-place append-like operation).

Tuples are considered lightweight, because it takes only the memory that is required for the data unlike list. It is recommended to use tuple, if the data is static.

Another benefit of using Tuples is resource caching. Python is garbage collected, it means that if a variable isn’t used anymore, then it frees its memory, giving it back to OS for allocating that memory to other application or variable. For tuple, if a tuples space is not used anymore, then it reserves the memory of it and if in future the memory of that size is required, python instead of reaching out to OS for system memory, it allocates the reserved memory. It avoids system call for block of memory.

### Instantiation in List and Tuple

```python
>>> %timeit l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
95 ns ± 1.87 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

>>> %timeit t = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
12.5 ns ± 0.199 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)
```
Both the lists and tuples have there pros and cons, but its important to keep in mind about their properties like overallocation in list and immutability & resource caching in tuple while using it as a data structure.

I hope, you liked reading the article !

### Reference

[High Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)
