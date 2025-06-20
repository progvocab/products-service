### 🌀 `deque` in Python (from `collections` module)

A **`deque`** (pronounced *deck*, short for **double-ended queue**) is a **flexible list-like container** that allows **fast appends and pops from both ends**.

It's part of Python’s `collections` module and is implemented as a **doubly linked list**, unlike lists which are dynamic arrays.

---

## 🔧 Why use `deque`?

* `O(1)` time complexity for:

  * `append()` and `pop()` (right end)
  * `appendleft()` and `popleft()` (left end)
* `list.insert(0, x)` and `list.pop(0)` are **O(n)** — much slower!

---

## ✅ Import Statement

```python
from collections import deque
```

---

## 📘 Creating a deque

```python
from collections import deque

d = deque()                  # empty deque
d = deque([1, 2, 3])         # deque with initial elements
print(d)  # deque([1, 2, 3])
```

---

## 🔹 Common Methods with Examples

| Method             | Description                              | Example                |
| ------------------ | ---------------------------------------- | ---------------------- |
| `append(x)`        | Add element to **right end**             | `d.append(4)`          |
| `appendleft(x)`    | Add element to **left end**              | `d.appendleft(0)`      |
| `pop()`            | Remove and return **rightmost**          | `d.pop()`              |
| `popleft()`        | Remove and return **leftmost**           | `d.popleft()`          |
| `extend(iter)`     | Extend deque at right with iterable      | `d.extend([4,5])`      |
| `extendleft(iter)` | Extend at left (reversed order!)         | `d.extendleft([0,-1])` |
| `remove(x)`        | Remove first occurrence of value         | `d.remove(2)`          |
| `rotate(n)`        | Rotate n steps to right (or left if n<0) | `d.rotate(1)`          |
| `reverse()`        | Reverse the deque in-place               | `d.reverse()`          |
| `clear()`          | Remove all elements                      | `d.clear()`            |
| `copy()` (3.5+)    | Shallow copy                             | `d2 = d.copy()`        |

---

## 🧪 Code Examples

```python
from collections import deque

d = deque([1, 2, 3])
print("Initial:", d)

d.append(4)
print("append(4):", d)

d.appendleft(0)
print("appendleft(0):", d)

d.pop()
print("pop():", d)

d.popleft()
print("popleft():", d)

d.extend([4, 5])
print("extend([4,5]):", d)

d.extendleft([-1, -2])
print("extendleft([-1, -2]):", d)  # Note: reversed order

d.remove(2)
print("remove(2):", d)

d.rotate(1)
print("rotate(1):", d)

d.rotate(-2)
print("rotate(-2):", d)

d.reverse()
print("reverse():", d)

d.clear()
print("clear():", d)
```

---

## 🧠 Deque vs List

| Feature                         | `deque`                 | `list`                 |
| ------------------------------- | ----------------------- | ---------------------- |
| Fast insert/remove at both ends | ✅                       | ❌ (slow at front)      |
| Random access (`d[2]`)          | ❌ (O(n))                | ✅ (O(1))               |
| Ideal use cases                 | Queues, Sliding Windows | Index-based operations |

---

## 🔄 Queue / Stack / Sliding Window

* **Queue (FIFO)**: Use `append()` + `popleft()`
* **Stack (LIFO)**: Use `append()` + `pop()`
* **Sliding Window**: Efficient left/right pop for window shifts

---

Let me know if you want examples of:

* **Sliding window max/min**
* **Palindrome checker**
* **BFS using deque**


Great question!

### ✅ What happens when you create a `deque` from a list of dictionaries?

It works exactly like creating a deque from any list — each **dictionary becomes an element** of the deque.

---

### 🔸 Example

```python
from collections import deque

data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
]

dq = deque(data)
print(dq)
```

**Output:**

```
deque([{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}])
```

* Now `dq[0]` is `{'id': 1, 'name': 'Alice'}`.
* `dq.popleft()` would remove the first dictionary.

---

## 🧠 Use Cases of `deque` with List of Dicts

### 1. **Queue of Tasks or Events**

Each dictionary can represent a task or event.

```python
tasks = deque([
    {"id": 101, "task": "send_email"},
    {"id": 102, "task": "backup_db"}
])

tasks.append({"id": 103, "task": "generate_report"})
current = tasks.popleft()
print(current)  # {'id': 101, 'task': 'send_email'}
```

---

### 2. **Sliding Window over Structured Data**

You can maintain a fixed-size `deque` of data points (e.g., for analytics).

```python
from collections import deque

window = deque(maxlen=3)
window.append({"user": "Alice", "action": "click"})
window.append({"user": "Bob", "action": "scroll"})
window.append({"user": "Charlie", "action": "click"})
window.append({"user": "Diana", "action": "hover"})

print(window)
# deque([{'user': 'Bob', ...}, {'user': 'Charlie'...}, {'user': 'Diana'...}])
```

---

### 3. **Caching (LRU, FIFO)**

Each dict can represent a cached item with metadata (timestamp, ID, etc.).

```python
cache = deque(maxlen=2)
cache.append({"key": "A", "value": 1})
cache.append({"key": "B", "value": 2})
cache.append({"key": "C", "value": 3})  # "A" is removed (FIFO)

print(cache)  # [{'key': 'B', ...}, {'key': 'C', ...}]
```

---

### 4. **Stream Processing / Messaging**

Process events from the front, and keep adding new ones to the back.

```python
messages = deque()
messages.append({"msg": "Hello", "sender": "User1"})
messages.append({"msg": "Bye", "sender": "User2"})

while messages:
    msg = messages.popleft()
    print(f"Processing message from {msg['sender']}")
```

---

### ✅ Summary

| Purpose          | How deque of dicts helps                      |
| ---------------- | --------------------------------------------- |
| Task queues      | Easy FIFO processing of structured tasks      |
| Sliding window   | Track recent structured entries               |
| Cache management | Maintain LRU/FIFO cache of objects            |
| Stream handling  | Append and pop structured records efficiently |

---

Let me know if you want a real project-like example (e.g., implementing a task scheduler or LRU cache with `deque` of dicts).
