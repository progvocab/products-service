### **`NavigableMap` in Java** – Complete Overview

#### **What is NavigableMap?**
`NavigableMap<K, V>` is a subinterface of `SortedMap` that provides **navigation methods** to traverse the map in both directions (ascending and descending).

It is **implemented by**:
- `TreeMap` (most common)
- `ConcurrentSkipListMap` (thread-safe variant)

---

### **Key Features of NavigableMap**
| Feature | Description |
|--------|-------------|
| Navigation | Methods like `lowerKey`, `floorKey`, `ceilingKey`, `higherKey` |
| Range Views | Methods like `subMap`, `headMap`, `tailMap` with inclusive/exclusive bounds |
| Reverse Order | `descendingMap`, `descendingKeySet` |
| Sorted | Maintains keys in sorted order (natural or custom comparator) |

---

### **TreeMap vs NavigableMap vs HashMap**

| Feature | `HashMap` | `TreeMap` (implements `NavigableMap`) | `NavigableMap` |
|--------|-----------|----------------------------------------|----------------|
| Ordering | No | Sorted (ascending) | Sorted (with navigation features) |
| Null Keys | 1 allowed | Not allowed | Depends on impl (TreeMap → Not allowed) |
| Thread-Safe | No | No | No |
| Lookup Time | O(1) average | O(log n) | O(log n) |
| Navigation Methods | No | Yes | Yes |
| Backed By | Hash Table | Red-Black Tree | Red-Black Tree / SkipList |

---

### **Code Examples**

#### 1. TreeMap with NavigableMap

```java
import java.util.*;

public class NavigableMapExample {
    public static void main(String[] args) {
        NavigableMap<Integer, String> map = new TreeMap<>();
        map.put(10, "Ten");
        map.put(20, "Twenty");
        map.put(30, "Thirty");

        System.out.println("Original Map: " + map);
        System.out.println("Lower than 20: " + map.lowerKey(20));      // 10
        System.out.println("Ceiling of 20: " + map.ceilingKey(20));    // 20
        System.out.println("Descending Map: " + map.descendingMap());
    }
}
```

#### 2. Submap Example

```java
NavigableMap<Integer, String> sub = map.subMap(10, true, 30, false);
System.out.println("SubMap (10 inclusive to 30 exclusive): " + sub);
```

---

### **Internal Working**
- Implemented using a **Self-balancing Binary Search Tree** (Red-Black Tree).
- Provides **O(log n)** performance for:
  - Insertions
  - Deletions
  - Lookups
  - Navigation methods

---

### **Use Cases**

| Use Case | Why Use NavigableMap? |
|----------|------------------------|
| Range queries | Use `subMap`, `headMap`, `tailMap` |
| Closest match lookups | `floorKey`, `ceilingKey`, `higherKey` |
| Time-series data | Sorted timestamps, easy navigation |
| Leaderboards | Sorted scores and retrieval |
| Autocomplete systems | Find closest matches by prefix |

---

Would you like a visual of Red-Black Tree insertions or comparisons with `ConcurrentSkipListMap` too?