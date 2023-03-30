---
layout: post
title:  Building An One-stop-shop For Python Modules
description: Playing With Python Modules
category: Blog
date:   2023-03-24 13:43:52 +0530
---
{% include mathjax.html %}

Python provides several built-in modules that can be used to handle data in various formats such as JSON, XML, CSV, and more. Here's a brief introduction to some of these modules:

## JSON

JSON (JavaScript Object Notation) is a lightweight data interchange format that is easy for humans to read and write and easy for machines to parse and generate. Python's built-in json module provides functions for encoding Python objects into JSON strings and decoding JSON strings into Python objects.

```python
    
    import json

    stock = {'Tesla': [30, 46, 78, 37, 98], 
             'Google': [90, 121, 124, 210, 200]}

```
Here, we convert the python dictionary object write as a json file and read the json file.

```python

    'Convert a Python dictionary object into JSON'

    with open('stock.json', 'w') as file:
        json.dump(stock, file)

    'Loading a JSON file'

    with open('stock.json', 'r') as file:
        json.load(file)

```
Next, we convert the python dictionary object into JSON string and then load it as JSON object.

```python

    'Encoding Python Dict into JSON String'

    json_str = json.dumps(stock)
    print(json_str)

    #output:
    "{'Tesla': [30, 46, 78, 37, 98], 'Google': [90, 121, 124, 210, 200]}"

    'Decoding JSON String into JSON Object'
    json.loads(json_str)

    #output:
    {
        'Tesla': [30, 46, 78, 37, 98], 
        'Google': [90, 121, 124, 210, 200]
    }

```

## XML

XML (Extensible Markup Language) is a markup language that is used to store and transport data. Python's built-in xml module provides functions for parsing XML data and generating XML documents.

```python

    'Reading a XML File'
    !cat food.xml

    #Output:

    <?xml version="1.0" encoding="UTF-8"?>
    <metadata>
    <food>
        <item name="breakfast">Idly</item>
        <price>$2.5</price>
        <description>
    Two idly's with chutney
    </description>
        <calories>553</calories>
    </food>
    </metadata>

```
We can parse the XML file using xml module as follows

```python

    import xml.etree.ElementTree as ET

    parsed = ET.parse('food.xml')

    'Get the root element of the XML file'

    root = parsed.getroot()
    print(root)

    #output:
    <Element 'metadata' at 0x7f8d8e8096b0>

```
Next, we iterate through all XML tags as follows

```python

    'Iterate through Parsed XML file'

    for ele in parsed.iter():
        print(ele)

    #output:
    <Element 'metadata' at 0x7f8d8e8096b0>
    <Element 'food' at 0x7f8d8e809950>
    <Element 'item' at 0x7f8d8e809cb0>
    <Element 'price' at 0x7f8d8e809d70>
    <Element 'description' at 0x7f8d8e809d10>
    <Element 'calories' at 0x7f8d8e809e30>

```
Next, we iterate through all XML tags and extract text from it, as follows

```python

    'Iterate through Parsed XML file to get the text'

    for ele in parsed.iter():
        print(ele.tag, ":", ele.text)

    #output:
    metadata : 

    food : 
        
    item : Idly
    price : $2.5
    description : 
    Two idly's with chutney
    
    calories : 553

```

Next, we'll find the text based on tag name as follow:

```python

    root[0].find('item').text

    #output:
    'Idly'

```

CSV: CSV (Comma Separated Values) is a common format for storing and exchanging data in a simple tabular format. Python's built-in csv module provides functions for reading and writing CSV files.

Pickle: Python's built-in pickle module provides a way to serialize and deserialize Python objects. It can be used to store Python objects to a file or transmit them over a network.

These modules are widely used in data processing, web development, and other areas of software development where data needs to be stored, exchanged, or manipulated in different formats.




