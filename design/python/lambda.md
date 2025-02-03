# **Lambda Functions in Python**  

**Lambda functions**, also known as **anonymous functions**, are small, unnamed functions defined using the `lambda` keyword. They are often used for **short, throwaway functions** that are not reused elsewhere.

---

## **1. Basic Syntax of Lambda Functions**
```python
lambda arguments: expression
```
- **arguments:** Input parameters (can be multiple, separated by commas).
- **expression:** A single expression evaluated and returned.

---

## **2. Example of a Simple Lambda Function**

### **Example 1: Addition**
```python
add = lambda x, y: x + y
print(add(3, 5))
```
### **Output:**
```
8
```
- Here, `lambda x, y: x + y` is equivalent to:
```python
def add(x, y):
    return x + y
```

---

## **3. Why Use Lambda Functions?**
- **Concise Syntax:** Ideal for simple operations.
- **Inline Use:** Often used where functions are required temporarily.
- **Higher-Order Functions:** Easily combined with functions like `map()`, `filter()`, and `sorted()`.

---

## **4. Use Cases with Examples**

### **4.1. Using Lambda with `map()`**
`map()` applies a function to **all items** in an iterable.

```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)
```
### **Output:**
```
[1, 4, 9, 16, 25]
```
- Each number is squared using a lambda function.

---

### **4.2. Using Lambda with `filter()`**
`filter()` selects items from an iterable based on a condition.

```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)
```
### **Output:**
```
[2, 4, 6]
```
- Filters out even numbers.

---

### **4.3. Using Lambda with `sorted()`**
Custom sorting criteria with `sorted()`.

```python
students = [("Alice", 85), ("Bob", 75), ("Charlie", 95)]
sorted_students = sorted(students, key=lambda x: x[1])
print(sorted_students)
```
### **Output:**
```
[('Bob', 75), ('Alice', 85), ('Charlie', 95)]
```
- Sorts by scores (second element of each tuple).

---

### **4.4. Using Lambda with `reduce()`**
`reduce()` applies a function cumulatively to the items of an iterable.

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print(product)
```
### **Output:**
```
120
```
- Multiplies all numbers together.

---

## **5. Combining Lambda with Other Functions**

### **5.1. Lambda in List Comprehension**
```python
numbers = [1, 2, 3, 4]
doubled = [(lambda x: x * 2)(x) for x in numbers]
print(doubled)
```
### **Output:**
```
[2, 4, 6, 8]
```

---

### **5.2. Lambda in Dictionary Sorting**
```python
students = {"Alice": 85, "Bob": 75, "Charlie": 95}
sorted_students = sorted(students.items(), key=lambda item: item[1])
print(sorted_students)
```
### **Output:**
```
[('Bob', 75), ('Alice', 85), ('Charlie', 95)]
```

---

## **6. Limitations of Lambda Functions**
- **Single Expression:** Limited to one expression (no multi-line logic).
- **No Annotations:** Cannot include type hints or annotations.
- **Readability:** Can be less readable for complex operations.

---

## **7. When to Use Lambda Functions?**
- **Small, simple operations:** Like sorting, filtering, or mapping.
- **Higher-order functions:** Where temporary, inline functions are needed.
- **Avoid complex logic:** For readability, prefer `def` for complex operations.

---

## **8. Lambda vs Regular Function (def)**
```python
# Lambda
multiply = lambda x, y: x * y

# Regular Function
def multiply(x, y):
    return x * y

print(multiply(2, 3))  # Output: 6
```
- **Lambda:** Concise, single-expression
- **def:** More flexible, supports complex logic

---

## **Conclusion**
Lambda functions in Python are **powerful and concise tools** for writing small, unnamed functions, especially useful in functional programming constructs. Use them when you need **quick, inline functionality**. For more complex logic, stick to **regular functions** for better readability.

Let me know if you want more examples or details on **advanced usage** of lambda functions!