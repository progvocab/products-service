Great topic! Let's dive deep into Python’s built-in **`lru_cache`** decorator from the `functools` module.

---

## 🧠 What is `lru_cache`?

`lru_cache` stands for **Least Recently Used Cache**.

It is used to **automatically memoize** the results of function calls to speed up repeated computations, especially for **pure functions** (functions with no side effects and consistent outputs for same inputs).

---

### 🔹 Importing and Using `lru_cache`

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

* Decorates the `fib` function.
* Results of up to `128` unique inputs are cached.
* Cache avoids recomputation.

---

## 🔸 Parameters of `lru_cache`

```python
@lru_cache(maxsize=128, typed=False)
```

| Parameter | Meaning                                                                              |
| --------- | ------------------------------------------------------------------------------------ |
| `maxsize` | Max number of cached entries. `None` means **unbounded** cache.                      |
| `typed`   | If `True`, arguments of different types are cached separately (e.g., `1` and `1.0`). |

---

## 🔸 Default Behavior

* If `maxsize` is reached, **least recently used entry is evicted**.
* Eviction policy is **LRU** (Least Recently Used).

🧠 **LRU strategy**:
If you access a cached item, it becomes "recently used" and moves to the front of the internal queue.

---

## 🔎 What Happens on Cache Overflow?

When more than `maxsize` **unique** inputs are used:

* The **oldest least recently used** entry is removed.
* This happens automatically and transparently.

🔸 Example:

```python
@lru_cache(maxsize=2)
def slow_func(x):
    return x * 2

slow_func(1)  # Cached
slow_func(2)  # Cached
slow_func(3)  # Cache full, evicts 1
```

Only results for `2` and `3` are now cached.

---

## ⏳ Time-To-Live (TTL)?

❌ **`lru_cache` does NOT support TTL (expiration by time)** out of the box.

If you need a **TTL-based cache**, use:

### ✅ Alternative: `cachetools` (third-party)

```python
from cachetools import TTLCache

cache = TTLCache(maxsize=100, ttl=300)  # 300 seconds = 5 mins
```

Or use `functools.cache` + custom logic in Python 3.9+.

---

## ✅ Inspecting the Cache

`lru_cache` gives a few built-in tools:

```python
print(fib.cache_info())
# CacheInfo(hits=12, misses=6, maxsize=128, currsize=6)

fib.cache_clear()  # clears the cache
```

---

## ✅ Summary

| Feature        | `lru_cache` Behavior                       |
| -------------- | ------------------------------------------ |
| Max size       | Configurable via `maxsize`, default is 128 |
| On overflow    | Evicts least recently used entry           |
| TTL            | ❌ Not supported natively                   |
| Type-sensitive | Optional with `typed=True`                 |
| Thread-safe    | ✅ Yes                                      |
| Use cases      | Memoization of recursive/pure functions    |

---

## 🧪 Real-World Use Cases

* Recursive algorithms like Fibonacci, DFS, etc.
* Expensive I/O or database calls with repeated inputs.
* Dynamic programming optimization.

---

Would you like a custom LRU+TTL cache implemented in Python? Or a comparison with `functools.cache` in Python 3.9+?
