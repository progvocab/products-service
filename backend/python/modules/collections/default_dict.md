### 🧰 `collections.defaultdict` in Python

The `collections.defaultdict` is a subclass of the built-in `dict` class. It simplifies handling missing keys by automatically creating a default value when a key is accessed but not yet present.

---

### ✅ Syntax

```python
from collections import defaultdict

defaultdict(default_factory)
```

* `default_factory`: A function (like `int`, `list`, `set`, etc.) that provides the default value for missing keys.

---

### 🧪 Basic Example

```python
from collections import defaultdict

# default value is an empty list
dd = defaultdict(list)

dd['a'].append(1)
dd['b'].append(2)

print(dd)  # {'a': [1], 'b': [2]}
```

If you had used a normal `dict`, `dd['a'].append(1)` would raise a `KeyError` unless `'a'` was already initialized.

---

### 🔧 Common Use Cases

| Use Case               | `default_factory` |
| ---------------------- | ----------------- |
| Counting items         | `int`             |
| Grouping values        | `list`            |
| Creating sets of items | `set`             |

---

### 📊 1. Frequency Counter

```python
from collections import defaultdict

freq = defaultdict(int)

for char in "banana":
    freq[char] += 1

print(freq)  # {'b': 1, 'a': 3, 'n': 2}
```

---

### 📚 2. Grouping Items

```python
from collections import defaultdict

words = ["apple", "banana", "apricot", "blueberry"]

grouped = defaultdict(list)
for word in words:
    key = word[0]  # group by first letter
    grouped[key].append(word)

print(grouped)
# {'a': ['apple', 'apricot'], 'b': ['banana', 'blueberry']}
```

---

### 🧼 3. Set of Values per Key

```python
from collections import defaultdict

d = defaultdict(set)
d['a'].add(1)
d['a'].add(2)
d['b'].add(3)

print(d)  # {'a': {1, 2}, 'b': {3}}
```

---

### ⚠️ Gotchas

* If you access a key that doesn't exist, it **creates** it with the default value.

```python
dd = defaultdict(int)
print(dd['missing'])  # 0 — and 'missing' is now a key in the dict
```

---

### 💡 Custom Factory Example

```python
def default_name():
    return "Unknown"

names = defaultdict(default_name)
names['id1'] = 'Alice'
print(names['id1'])  # Alice
print(names['id2'])  # Unknown
```

---

### 🧠 Summary

| Feature               | Behavior                          |
| --------------------- | --------------------------------- |
| Auto-initializes keys | Yes                               |
| Default value type    | Controlled by `default_factory`   |
| Subclass of `dict`    | Yes                               |
| Prevents KeyError     | Yes (when accessing missing keys) |

---

Would you like examples comparing `dict.setdefault()` vs `defaultdict()`?
