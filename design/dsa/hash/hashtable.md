Here’s a simple implementation of a **Hash Table (HashMap)** in Python using **lists** for buckets and **separate chaining** to handle collisions.

---

## **Hash Table Implementation (Basic)**

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.buckets = [[] for _ in range(self.size)]  # list of lists

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        index = self._hash(key)
        # Check if key exists and update it
        for i, (k, v) in enumerate(self.buckets[index]):
            if k == key:
                self.buckets[index][i] = (key, value)
                return
        # Otherwise, insert new key-value pair
        self.buckets[index].append((key, value))

    def get(self, key):
        index = self._hash(key)
        for k, v in self.buckets[index]:
            if k == key:
                return v
        return None  # key not found

    def remove(self, key):
        index = self._hash(key)
        for i, (k, _) in enumerate(self.buckets[index]):
            if k == key:
                del self.buckets[index][i]
                return True
        return False  # key not found

    def __str__(self):
        return str(self.buckets)
```

---

### **Usage Example**

```python
ht = HashTable()

ht.put("name", "Alice")
ht.put("age", 30)
ht.put("city", "New York")

print(ht.get("name"))  # Output: Alice
print(ht.get("age"))   # Output: 30

ht.remove("age")
print(ht.get("age"))   # Output: None

print(ht)  # Shows the internal bucket structure
```

---

### **Features:**

* Uses Python’s built-in `hash()` function.
* Resolves collisions via **separate chaining** (each bucket is a list).
* Supports basic operations: **put**, **get**, and **remove**.

Would you like an extended version with:

* Dynamic resizing (rehashing)?
* Support for custom object keys?
* Open addressing instead of chaining?
