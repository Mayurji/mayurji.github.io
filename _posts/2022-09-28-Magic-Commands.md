---
layout: post
title:  How to use Jupyter Notebook effectively with Magic commands?
description: Tricks you can use in Jupyter Notebook.
category: Blog
date:   2022-09-28 13:43:52 +0530
---
{% include mathjax.html %}    

<center>
<img src="{{site.url}}/assets/images/ml/patrick-tomasso-Oaqk7qqNh_c-unsplash.jpg" style="zoom: 5%; background-color:#DCDCDC;"  width="80%" height=auto/><br>
<p>Unsplash</p>
</center>

Data scientists and data analysts frequently utilise Jupyter Notebooks as a tool to begin their analyses. And there are a few important Jupyter notebook tricks or commands that are hardly ever used by the community.

In this post, we’ll execute commands which will make us a better user of Jupyter notebooks. Only pre-requisite to use magic commands is to install Ipython-kernel


```python

    pip install ipykernel

```

### Time taken to execute a block of code? %%time

```python

    %%time
    count = 0
    for i in range(10000):
        count += i
    
    CPU times: user 1.39 ms, sys: 0 ns, total: 1.39 ms 
    Wall time: 1.4 ms

```

### Time taken to execute each line of code? %time

```python

    %time print("a")
    %time print("b")
    a 
    CPU times: user 35 µs, sys: 0 ns, total: 35 µs 
    Wall time: 37.9 µs

    b 
    CPU times: user 8 µs, sys: 0 ns, total: 8 µs 
    Wall time: 9.06 µs

```

### How to push code of your Jupyter cell into Python file? %%writefile
```python

    %%writefile addition.py
    def add(a, b):
        return a+b
    add(5, 6)

```

In the above code, the function add or the lines below “%%writefile addition.py” is pushed into addition.py file. It will overwrite existing file or create a new file.

### How to load the content of python file into Jupyter cell? %load

```python

    %load addition.py

```
On running above code, the contents of the file will be loaded into jupyter cell as follows

```python

    # %load addition.py
    def add(a, b):
        return a+b
    add(5, 6)

```

### How to run a python file in Jupyter cell? %run

```python

    %run addition.py
    11

```

### How to check the environment variables? %env

```python

    %env
    {'ELECTRON_RUN_AS_NODE': '1',  'USER': 'mayur',  'LANGUAGE': 'en_IN:en',  'TEXTDOMAIN': 'im-config',  'XDG_SEAT': 'seat0',  'XDG_SESSION_TYPE': 'x11',  'SSH_AGENT_PID': '1711',  'SHLVL': '1',
    ...
    'PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING': '1'}

```

### How to set environment variable? %set_env

```python

    %set_env NEW_PATH=/usr/bin/python1

```
You can verify the newly created environment variable using %env.

### How to find all the initialized variables in Jupyter? %who

```python

    %who
    count i

```

### How to all the variables based on datatype? %who datatype

```python

    %who int
    count i

```

### How to find all the available magic commands? %lsmagic

```python

    %lsmagic
    {"line":{"automagic":"AutoMagics","autocall":"AutoMagics","alias_magic":"BasicMagics","lsmagic":"BasicMagics","magic":"BasicMagics","page":"BasicMagics","pprint":"BasicMagics","colors":"BasicMagics","xmode":"BasicMagics","quickref":"BasicMagics","doctest_mode":"BasicMagics","gui":"BasicMagics",..............
    "python3":"Other","pypy":"Other","SVG":"Other","HTML":"Other","file":"Other"}}

```

### How to write html code in Jupyter cell? %%html

```python

    %%html
    <html>
       <body>
           <strong>This is %%HTML magic commands</strong>
       </body>
    </html>

    This is %%HTML magic commands

```

Find the above post as jupyter notebook here: [github](https://github.com/Mayurji/LearningPython/blob/main/miscellaneous/magic_commands.ipynb)

Find the above post as youtube video here: [YouTube](https://youtu.be/MvZxrIpFim8)


