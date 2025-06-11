I'll explain the `functools` module in Python, which provides useful tools for working with functions and callable objects. Let's look at some of its key features with examples.

1. **`partial` - Creating Partial Functions**
```python
from functools import partial

# Original function
def multiply(x, y):
    return x * y

# Create a new function with one argument fixed
double = partial(multiply, 2)  # fixes x=2
triple = partial(multiply, 3)  # fixes x=3

print(double(5))  # Output: 10
print(triple(5))  # Output: 15
```
`partial` lets you "freeze" some arguments of a function, creating a new function with fewer parameters.

2. **`reduce` - Cumulative Operations**
```python
from functools import reduce

# Add numbers cumulatively
numbers = [1, 2, 3, 4, 5]
sum_result = reduce(lambda x, y: x + y, numbers)
print(sum_result)  # Output: 15

# Find maximum
max_result = reduce(lambda x, y: x if x > y else y, numbers)
print(max_result)  # Output: 5
```
`reduce` applies a function of two arguments cumulatively to sequence items, reducing it to a single value.

3. **`lru_cache` - Memoization Decorator**
```python
from functools import lru_cache

@lru_cache(maxsize=32)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First call computes, subsequent calls use cache
print(fibonacci(10))  # Output: 55
print(fibonacci.cache_info())  # Shows cache stats
```
`lru_cache` caches function results, speeding up subsequent calls with the same arguments.

4. **`wraps` - Preserving Function Metadata**
```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Something before")
        func(*args, **kwargs)
        print("Something after")
    return wrapper

@my_decorator
def say_hello(name):
    """Say hello to someone"""
    print(f"Hello, {name}!")

say_hello("Alice")
print(say_hello.__name__)  # Output: "say_hello"
print(say_hello.__doc__)   # Output: "Say hello to someone"
```
`wraps` helps decorators preserve the original function's metadata.

5. **`cmp_to_key` - Converting Comparison Functions**
```python
from functools import cmp_to_key

def compare(a, b):
    return (a > b) - (a < b)  # Returns -1, 0, or 1

# Sort strings by length
words = ["python", "is", "awesome"]
sorted_words = sorted(words, key=cmp_to_key(lambda a, b: compare(len(a), len(b))))
print(sorted_words)  # Output: ['is', 'python', 'awesome']
```
`cmp_to_key` converts old-style comparison functions to key functions for sorting.

These are some of the most commonly used tools in `functools`. The module also includes other utilities like `singledispatch` for single-dispatch generic functions and `total_ordering` for auto-generating comparison methods.

Each of these tools helps make your Python code more functional, efficient, or maintainable depending on your use case. Let me know if you'd like deeper examples of any specific function!

In Python, `functools.cmp_to_key` is a utility that helps you use old-style comparison functions (i.e., `cmp` functions) with newer sorting mechanisms like `sorted()` or `list.sort()`, which expect a **key function**, not a **comparison function**.

---

### 🔄 Why `cmp_to_key`?

In **Python 2**, you could use a `cmp` function like this:

```python
sorted(data, cmp=my_cmp_function)
```

But in **Python 3**, the `cmp` parameter was **removed** from `sorted()` and `list.sort()`. Instead, you're expected to use a **key function**:

```python
sorted(data, key=my_key_function)
```

If you still have a `cmp`-style function, you can convert it using `functools.cmp_to_key`.

---

### 📘 Syntax

```python
from functools import cmp_to_key
key_function = cmp_to_key(cmp_function)
```

---

### 🧪 Example

Suppose you want to sort a list of strings by their length, and if lengths are equal, sort reverse-alphabetically.

#### 1. Define a `cmp` function

```python
def custom_cmp(a, b):
    if len(a) != len(b):
        return len(a) - len(b)  # shorter comes first
    return -1 if a > b else (1 if a < b else 0)  # reverse alphabetical
```

#### 2. Use `cmp_to_key`

```python
from functools import cmp_to_key

words = ['apple', 'banana', 'pear', 'grape', 'fig', 'kiwi']
sorted_words = sorted(words, key=cmp_to_key(custom_cmp))
print(sorted_words)
```

#### ✅ Output

```
['fig', 'pear', 'kiwi', 'grape', 'apple', 'banana']
```

* Sorted by length: fig (3), pear/kiwi/grape (4–5), apple/banana (5–6)
* Ties broken by reverse alphabet: 'pear' before 'kiwi'

---

### 🧠 How it works internally

`cmp_to_key` creates a **wrapper class** with rich comparison methods (`__lt__`, `__gt__`, etc.) that use your `cmp` function under the hood. So you can use your comparison logic even when Python requires key-based sorting.

---

### ✅ Use Cases

* Migrating legacy Python 2 code to Python 3.
* Sorting with complex comparison logic that doesn’t easily map to a key function.
* Custom multi-level sorts where a single key is insufficient.

---

Would you like to see how to write the same sort using a `key` function instead of `cmp_to_key`, for comparison?
