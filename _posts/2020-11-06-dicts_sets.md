Python - Dictionary vs Sets

*Understanding data structure helps in performant programming!*

In this blog post, we'll discuss the python data structures, *dict* and *set*.

Sets and Dictionary are the data structures used, when the data has no intrinsic order to it. But each data has unique object to reference it. For *dict*, the reference object is called *key* and referenced data is called *value*, the widely used reference object is of type string, but any hashable type is valid. While *set* is a unique collections of *keys* alone. 

### Hashable Type

Hashable type are those objects, which implements *__hash__* function. The hash function returns a integer value for the object (string, integer, float) passed to it. The integer value helps for quick lookup in the dict.

In the previous blog post, [Python - Lists vs Tuples](https://mayur-ds.medium.com/python-lists-and-tuples-760d45ebeaa8), I've mentioned the best case for a lookup in a list or tuple is O(log n) based on Binary Search. In dict and set, a element lookup takes a constant time of O(1), since the search is based on arbitary index. The speed in *dict* and *set* is achieved by using an open address hash table as its underlying data structure.


### Where to utilize dict and set

### Dict

Consider a phonebook with list of names and phone number. To find a phone number of a person, the general strategy for lookup in *a list or a tuple* data structure takes following step

*  Iterate over the names of the person
*  Find the match of the name by comparison with other names
*  Fetch the corresponding phone number, once the name match is found.

While in *dict*, we can store the person name as the key and phone number as value. It takes O(1) to find the phone number. In dict, we can fetch the phone number by doing a lookup on person name, which is unique. It doesn't require iteration through all names.

### Set

Consider a requirement for finding the total number of unique names, if the data is stored in lists or tuples, it requires multiple for loops, which makes the time complexity to be O(n^2). 

```python
def list_unique_names(phonebook):
    unique_names = []
    for name, phonenumber in phonebook: 
        first_name, last_name = name.split(" ", 1)
        for unique in unique_names: 
            if unique == first_name:
                break
        else:
            unique_names.append(first_name)
    return len(unique_names)
```

* We must go over all the items in the phone book, and thus this loop costs O(n).

* then, we must check the current name against all the unique names we have already seen. If it is a new unique name, we add it to our list of unique names. We then continue through the list, performing this step for every item in the phone book.

```python
def set_unique_names(phonebook):
	unique_names = set()
    for name, phonenumber in phonebook: 
        first_name, last_name = name.split(" ", 1)
        unique_names.add(first_name) 
    return len(unique_names)
```

* For the set method, instead of iterating over all unique names we have already seen, we can simply add the current name to the set of unique names. Because sets guarantee the uniqueness of the keys they contain, if we try to add an item that is already in the set, that item simply won’t be added. Furthermore, this operation costs O(1).

To achieve this speed, the dict and set uses the hash tables. Hash table is filled using hash function, which cleverly turns the arbitiary key into an index for fetching the value stored on that key.

### Inserting & Resizing in Dict

For creating a dict or any other data structure, we need to allocate a chunk of system memory to it. And to insert in dict, the insertion position or index depends on the data. While inserting, the key is hashed and masked to turn into an effective integer that fits the memory size allocated to it.

So if we have allocated 8 blocks of memory and our hash value is 28975, we consider the bucket at index 28975 & 0b111 = 7. If, however, the dictionary has grown to require 512 blocks of memory, the mask becomes 0b111111111 (and in this case, we would consider the bucket at index 28975 & 0b11111111). Now, if this bucket is available, then we can store the key & value into the memory block. If the memory block is not available, then the dict finds new memory block for insertion.

To find the new index, a mechanism called *probing* is used. Python’s probing mechanism adds a contribution from the higher-order bits of the original hash (recall that for a table of length 8 we considered only the last three bits of the hash for the initial index, through the use of a mask value of mask = 0b111 = bin(8 - 1)). Using these higher-order bits gives each hash a different sequence of next possible hashes, which helps to avoid future collisions.

*When a dict or set is initialized, the default size assigned is 8, i.e. meaning a hash table of size 8 is created, and when more items are added to the dict/set, python checks if two-third of the size is filled, if yes, then it increases the size of the dict/set by 3x. The resize happens everytime the dict is two-third filled. The possible resizes of dict are as follows*

```python
8; 18; 39; 81; 165; 333; 669; 1,341; 2,685; 5,373; 10,749; 21,501; 43,005; …
```

### Dictionary & Namespaces

Whenever a variable, function or module is invoked in python, a hierarchy of objects lookups happens, maintenance of such hierarchy is done by Namespace Management. It stores variable names in an hierarchy of its execution or calling. The lookups in namespace management heavily depends on dictionary.

When an object is invoked in python, the hierarchy of lookups starts, first, it checks into *local() array*, which isn't dictionary and if it doesn't exists there, then the hierarchy is moved to *global() array*, if not found, then the object is searched in *__builtin__* objects. It is important to note that while *local()* and *global()* is explicit dictionaries, *__builtin__* is a module object and a lookup in builtin objects is same as dictionary lookup in *local() map*.

### Example Namespace Lookup

```python
import math
from math import sin

def test1(x):
    """
    >>> %timeit test1(123_456)
    162 µs ± 3.82 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    """
    res = 1
    for _ in range(1000):
        res += math.sin(x)
    return res

def test2(x):
    """
    >>> %timeit test2(123_456)
    124 µs ± 6.77 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    """
    res = 1
    for _ in range(1000):
        res += sin(x)
    return res

def test3(x, sin=math.sin):
    """
    >>> %timeit test3(123_456)
    105 µs ± 3.35 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    """
    res = 1
    for _ in range(1000):
        res += sin(x)
    return res
```

Bytecode generated using ***dis*** module

IMAGE(bytecode)

In test1, *sin function* is   called explicitly by looking at math library. From bytecode generated, we can see, there are two dictionary lookup happens, one is finding the math module and then finding the sin function inside it.

In test2, *sin function* is imported from math library, which makes it available to global namespace, instead of  finding the math module and then finding the sin function inside it, we need to search sin function in global namespace. It thus helps in reducing the time taken.

*This is yet another reason to be explicit about what functions you are importing from a module. This practice not only makes code more readable, because the reader knows exactly what functionality is required from external sources, but it also simplifies changing the implementation of specific functions and generally speeds up code!*

In test3, *sin function* is defined in the function definition by default as keyword argument, and as mentioned earlier, the first lookup is done on local() array, which is not a dictionary lookup and local() array is a small array which has a very fast lookup. The execution time of test3 is fastest among all other tests.

With this in mind, a more readable solution would be to set a local variable with the global reference before the loop is started. We’ll still have to do the global lookup once whenever the function is called, but all the calls to that function in the loop will be made faster. This speaks to the fact that even minute slowdowns in code can be amplified if that code is being run millions of times. Even though a dictionary lookup itself may take only several hundred nanoseconds, if we are looping millions of times over this lookup, those nanoseconds can quickly add up.

### Downsides of using Dictionary and Sets

* Memory footprint is high
* Complex hashing function leads to slower lookup.

Reference

    * [Dictionary’s Internals](https://www.freecodecamp.org/news/exploring-python-internals-the-dictionary-a32c14e73efa/)
    * [High Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)