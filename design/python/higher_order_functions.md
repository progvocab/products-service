Higher-order functions in Python are functions that can either:
1. Take one or more functions as arguments
2. Return a function as a result
3. Or both

These are fundamental to functional programming in Python. Let's explore the main built-in higher-order functions with examples, plus show how to create custom ones.

### Built-in Higher-Order Functions

1. **`map()` - Applies a function to all items in an iterable**
```python
# Regular function
def square(x):
    return x * x

# Using map with a function
numbers = [1, 2, 3, 4]
squared = map(square, numbers)
print(list(squared))  # Output: [1, 4, 9, 16]

# With lambda
doubled = map(lambda x: x * 2, numbers)
print(list(doubled))  # Output: [2, 4, 6, 8]
```
`map()` takes a function and an iterable, applying the function to each element.

2. **`filter()` - Filters elements based on a predicate function**
```python
# Regular function
def is_even(n):
    return n % 2 == 0

# Using filter
numbers = [1, 2, 3, 4, 5, 6]
evens = filter(is_even, numbers)
print(list(evens))  # Output: [2, 4, 6]

# With lambda
odds = filter(lambda x: x % 2 != 0, numbers)
print(list(odds))  # Output: [1, 3, 5]
```
`filter()` keeps only the elements where the function returns `True`.

3. **`reduce()` - Cumulative operation (from functools)**
```python
from functools import reduce

# Function to add two numbers
def add(x, y):
    return x + y

numbers = [1, 2, 3, 4]
total = reduce(add, numbers)
print(total)  # Output: 10

# With lambda and initial value
product = reduce(lambda x, y: x * y, numbers, 1)
print(product)  # Output: 24
```
`reduce()` applies a function cumulatively to reduce the iterable to a single value.

4. **`sorted()` - Sorts using a key function**
```python
# Sort by length
words = ["python", "is", "cool"]
sorted_by_length = sorted(words, key=len)
print(sorted_by_length)  # Output: ['is', 'cool', 'python']

# Sort by last letter
sorted_by_last = sorted(words, key=lambda x: x[-1])
print(sorted_by_last)  # Output: ['cool', 'python', 'is']
```
`sorted()` uses a key function to determine sort order.

### Custom Higher-Order Functions

1. **Function Returning a Function**
```python
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # Output: 10
print(triple(5))  # Output: 15
```
Here, `create_multiplier` returns a new function with a fixed factor.

2. **Function Taking a Function as Argument**
```python
def apply_twice(func, value):
    return func(func(value))

def add_one(x):
    return x + 1

result = apply_twice(add_one, 5)
print(result)  # Output: 7 (5 + 1 = 6, 6 + 1 = 7)

# With lambda
square_twice = apply_twice(lambda x: x * x, 3)
print(square_twice)  # Output: 81 (3 * 3 = 9, 9 * 9 = 81)
```
`apply_twice` takes a function and applies it twice to the value.

3. **Combining Both**
```python
def create_repeater(times):
    def repeater(func):
        def wrapper(x):
            result = x
            for _ in range(times):
                result = func(result)
            return result
        return wrapper
    return repeater

# Create a function that triples an operation
triple_it = create_repeater(3)

@triple_it
def increment(x):
    return x + 1

print(increment(5))  # Output: 8 (5 + 1 + 1 + 1)
```
This combines returning functions and accepting functions as arguments using a decorator.

### Key Characteristics
- **First-class functions**: Python treats functions as objects that can be passed around
- **Immutability**: These functions typically don't modify state
- **Composition**: They enable function composition for cleaner code

Higher-order functions promote code reuse, abstraction, and functional programming style. They're especially powerful when combined with lambda functions for concise, inline operations. Let me know if you'd like more examples or clarification on any of these!