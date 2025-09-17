Great! Here's the counterpart to `all()` — the **`any()`** function in Python.

---

### ✅ **What is `any()`?**

The `any()` function returns `True` **if at least one element** in the iterable is truthy.

---

### 🧾 **Syntax:**

```python
any(iterable)
```

* **Returns `True`** if **any element** of the iterable is true.
* **Returns `False`** only if **all elements are falsy**.

---

### 🔍 **Examples:**

#### 1. At least one truthy value:

```python
any([0, False, None, 5])
# Output: True (because of 5)
```

#### 2. All falsy values:

```python
any([0, False, None, ''])
# Output: False
```

#### 3. With conditions:

```python
nums = [3, 5, 7, 10]
any(n % 2 == 0 for n in nums)
# Output: True (10 is even)
```

#### 4. Check if string has vowels:

```python
s = "xyz"
any(c in 'aeiou' for c in s)
# Output: False
```

---

### 🧠 **When is `any()` useful?**

* Checking if any field is filled in a form
* Validating if **at least one** element in a dataset meets a condition
* Short-circuiting checks in matrix/graph/grid problems
* Efficient filtering in large collections

---

### 🔁 Summary:

| Function | Returns `True` When...             |
| -------- | ---------------------------------- |
| `all()`  | **All** elements are truthy        |
| `any()`  | **At least one** element is truthy |

---

Would you like compact real-world examples using `all()` and `any()` in graph or data validation tasks?
