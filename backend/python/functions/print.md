The `print()` function in Python is one of the most commonly used functions to **display output** to the console.

---

## 🖨️ `print()` — Overview

**Syntax:**

```python
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
```

| Parameter  | Description                                     |
| ---------- | ----------------------------------------------- |
| `*objects` | One or more values to print                     |
| `sep`      | String inserted between values (default: space) |
| `end`      | String printed at end (default: newline `\n`)   |
| `file`     | Output stream (default: `sys.stdout`)           |
| `flush`    | Force flush the stream (default: `False`)       |

---

### ✅ **Basic Usage**

```python
print("Hello, world!")
```

---

### 🧩 **Multiple Arguments + `sep`**

```python
print("Shahbaz", "Aamir", sep="-")
# Output: Shahbaz-Aamir
```

```python
print("2025", "07", "20", sep="/")
# Output: 2025/07/20
```

---

### ⏎ **Customize Line End with `end`**

```python
print("Loading", end="...")
print("Done")
# Output: Loading...Done
```

---

### 🗂️ **Redirect Output using `file`**

```python
with open("output.txt", "w") as f:
    print("Writing to file", file=f)
```

---

### 🔄 **Flush Output Immediately**

Useful for logging or progress bars.

```python
import time

for i in range(3):
    print(".", end="", flush=True)
    time.sleep(1)
```

---

## 🔤 **String Formatting / Placeholders**

### 1. **Old Style (`%`)**

```python
name = "Shahbaz"
age = 30
print("Name: %s, Age: %d" % (name, age))
```

### 2. **`str.format()`**

```python
print("Name: {}, Age: {}".format("Aamir", 28))
print("Name: {0}, Age: {1}".format("Shahbaz", 30))
```

### 3. **f-strings (Python 3.6+)** ✅ Best Practice

```python
name = "Shahbaz"
age = 30
print(f"Name: {name}, Age: {age}")
```

Supports expressions too:

```python
print(f"5 + 3 = {5 + 3}")
```

---

## 💡 Bonus: Printing Lists or Dicts

```python
data = {"name": "Shahbaz", "city": "Delhi"}
print(data)
```

Use pretty-print for nested objects:

```python
import pprint
pprint.pprint(data)
```

---

## 🛠️ Common Use Cases

| Use Case            | Example                                       |
| ------------------- | --------------------------------------------- |
| Concatenate Strings | `print("Hello" + "World")`                    |
| Join List           | `print(" ".join(["This", "is", "joined"]))`   |
| Inline Logging      | `print(f"[INFO] Task done at {time.time()}")` |
| Print Progress      | `print(".", end="", flush=True)`              |

---

Would you like a printable cheat sheet for `print()` or comparison with `logging` for larger apps?
