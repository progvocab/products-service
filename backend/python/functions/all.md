The `all()` function in Python is used to **check if all elements in an iterable are `True`**.

---

### ✅ **Syntax:**

```python
all(iterable)
```

* Returns `True` if **every element** in the iterable is truthy.
* Returns `False` **as soon as** it encounters a falsy value (like `0`, `False`, `None`, `''`, `[]`, etc.).

---

### 📘 **Examples:**

#### 1. All true values:

```python
all([True, 1, "hello", [1]])
# Output: True
```

#### 2. One or more false values:

```python
all([True, 0, 1])
# Output: False
```

#### 3. Used with comparisons:

```python
nums = [1, 2, 3, 4]
all(n < 5 for n in nums)
# Output: True
```

#### 4. Checking sorted array:

```python
arr = [1, 2, 3, 4, 5]
is_sorted = all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
# Output: True
```

---

### 🧠 **When is `all()` useful?**

* Validating data in a list (`all(field is not None for field in form_data)`)
* Checking if an array is sorted
* Verifying constraints on graph nodes, matrix elements, etc.
* Short-circuiting conditions elegantly without verbose loops

---

Let me know if you'd like to see the counterpart: `any()`.
