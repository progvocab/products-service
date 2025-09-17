Python provides several powerful **built-in data structures** that are essential for organizing and storing data efficiently. Here's a detailed breakdown of the four most commonly used ones:

---

## 📋 1. **List**

A **list** is an ordered, mutable (changeable) collection of elements. It can contain items of **mixed types**, including other lists.

### ✅ Properties:

* Ordered
* Mutable
* Allows duplicates
* Indexed

### 🧪 Example:

```python
my_list = [1, 2, 3, "hello", [4, 5]]
my_list.append(6)
print(my_list[3])   # "hello"
```

---

## 🔸 2. **Tuple**

A **tuple** is similar to a list but **immutable**. Once created, its elements cannot be changed.

### ✅ Properties:

* Ordered
* Immutable
* Allows duplicates
* Indexed

### 🧪 Example:

```python
my_tuple = (1, 2, 3)
print(my_tuple[1])  # 2

# Tuples can be used as dictionary keys
coordinates = {(1, 2): "A point"}
```

---

## 🟢 3. **Set**

A **set** is an unordered collection of **unique** elements. Useful for membership testing and eliminating duplicates.

### ✅ Properties:

* Unordered
* Mutable
* No duplicates
* No indexing or slicing

### 🧪 Example:

```python
my_set = {1, 2, 3, 3, 4}
print(my_set)       # {1, 2, 3, 4}

my_set.add(5)
my_set.remove(2)
print(3 in my_set)  # True
```

---

## 📙 4. **Dictionary**

A **dictionary** is a collection of key-value pairs. Keys must be **unique and immutable** (e.g., strings, numbers, or tuples).

### ✅ Properties:

* Unordered (as of Python 3.6+, insertion order is preserved)
* Mutable
* Keys must be unique
* Access by key

### 🧪 Example:

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

print(person["name"])       # Alice
person["age"] = 31
person["email"] = "alice@example.com"
```

---

## 🧾 Comparison Table

| Feature    | List         | Tuple      | Set          | Dictionary        |
| ---------- | ------------ | ---------- | ------------ | ----------------- |
| Ordered    | ✅            | ✅          | ❌            | ✅ (since 3.6+)    |
| Mutable    | ✅            | ❌          | ✅            | ✅                 |
| Duplicates | ✅            | ✅          | ❌            | ❌ (in keys)       |
| Indexed    | ✅            | ✅          | ❌            | ✅ (via keys)      |
| Use Case   | General data | Fixed data | Unique items | Key-value mapping |

---

Would you like a visual diagram or examples of how to choose the right one for a specific use case?
