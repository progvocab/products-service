## Single Dimension Array
### Array of same number

```python
x=[1] * 5
# x = [1,1,1,1,1]
```

### Array of sequence

```python
x=[i for i in range( 5 ) ]
# x = [0,1,2,3,4]
```

### Array of Odd or Even Numbers
- Find odd numbers in a list
```python
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y = [ i for i in x if i % 2 ]
# y = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
```

- Find even numbers in a list 
```python
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y = [ i for i in x if i % 2 ==0]
# y = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```
## 2 Dimension Array
### Array of Single Number
```python
n=3
m=4
y = [ [0]*m for _ in range(n) ]
# y = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

### Array of Sequence 
```python
n=3
m=4
y = [ [i]*m for i in range(n) ]
# y = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
```
Let’s break down the compact Python expression:

```python
[strongconnect(v) for v in graph if v not in indices]
```

### ✅ What this is:

This is a **list comprehension** used purely for its **side effects** — in this case, to **call the `strongconnect()` function** for each unvisited vertex in the graph.

---

### 🔍 Step-by-step explanation:

Let’s rewrite it in an **expanded form** first:

```python
for v in graph:
    if v not in indices:
        strongconnect(v)
```

The list comprehension version does the **same thing** but in one line.

---

### 💡 Components:

| Part                  | Meaning                                                           |
| --------------------- | ----------------------------------------------------------------- |
| `for v in graph`      | Iterate over all vertices in the graph                            |
| `if v not in indices` | Only process vertices not yet visited (not in the `indices` dict) |
| `strongconnect(v)`    | Run the Tarjan’s DFS on this vertex                               |
| `[...]`               | Wraps it in a list, but result is ignored                         |

---

### ⚠️ Note:

Even though it’s a list comprehension, the **return list is not used**. It’s just a Pythonic way to run something **for its side effects** (like calling a function).

If you want to be more explicit and avoid confusion, you can write:

```python
for v in graph:
    if v not in indices:
        strongconnect(v)
```

But in compact or functional code styles, you might see the list comprehension version.

---

### ✅ When to use

Use list comprehensions **only when you're interested in the result** (e.g., collecting values).
For side effects (like this DFS traversal), prefer the `for` loop version for clarity — unless you're writing intentionally compact code.

Would you like me to convert the full code using more traditional and readable `for` loops instead of compact constructs?
