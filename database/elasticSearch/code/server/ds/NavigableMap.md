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



### **Subclasses and Implementations of `NavigableMap`**

`NavigableMap` is an interface. It has the following **common implementations**:

| Implementation | Thread-Safe | Backed By |
|----------------|-------------|-----------|
| `TreeMap` | No | Red-Black Tree |
| `ConcurrentSkipListMap` | Yes | Skip List |

These implementations offer different guarantees for performance, concurrency, and use cases.

---

### **1. `TreeMap` (Most Common Implementation)**

- **Sorted by natural order** or custom `Comparator`
- Backed by **Red-Black Tree** (self-balancing BST)
- **Not thread-safe**

#### **Example: Range Queries with TreeMap**

```java
import java.util.*;

public class TreeMapRangeExample {
    public static void main(String[] args) {
        NavigableMap<Integer, String> map = new TreeMap<>();
        map.put(100, "A");
        map.put(200, "B");
        map.put(300, "C");
        map.put(400, "D");

        // Range: [150, 350)
        NavigableMap<Integer, String> subMap = map.subMap(150, true, 350, false);
        System.out.println("Range [150, 350): " + subMap); // Output: {200=B, 300=C}
    }
}
```

---

### **2. `ConcurrentSkipListMap` (Thread-safe version)**

- Concurrent, scalable, sorted map
- Backed by **Skip List**
- Suitable for multi-threaded environments

#### **Example: ConcurrentSkipListMap with Range Queries**

```java
import java.util.concurrent.*;

public class ConcurrentSkipListMapExample {
    public static void main(String[] args) {
        ConcurrentNavigableMap<Integer, String> map = new ConcurrentSkipListMap<>();
        map.put(10, "X");
        map.put(20, "Y");
        map.put(30, "Z");

        System.out.println("HeadMap (< 25): " + map.headMap(25)); // {10=X, 20=Y}
        System.out.println("TailMap (>= 20): " + map.tailMap(20)); // {20=Y, 30=Z}
    }
}
```

---

### **Comparison of TreeMap and ConcurrentSkipListMap**

| Feature | `TreeMap` | `ConcurrentSkipListMap` |
|--------|-----------|--------------------------|
| Thread-safe | No | Yes |
| Performance | O(log n) | O(log n) |
| Use Case | Single-threaded sorted map | Concurrent access with ordering |
| Backing Data Structure | Red-Black Tree | Skip List |

---

### **NavigableMap Range Query Methods**

| Method | Description | Example |
|--------|-------------|---------|
| `subMap(from, to)` | Returns keys in range [from, to) | `subMap(10, 30)` |
| `subMap(from, fromInclusive, to, toInclusive)` | Fine control over bounds | `subMap(10, true, 30, false)` |
| `headMap(toKey)` | All keys < toKey | `headMap(30)` |
| `tailMap(fromKey)` | All keys >= fromKey | `tailMap(20)` |

---

Would you like a comparison chart of Skip List vs Red-Black Tree too?



Here’s a **comparison chart of Skip List vs Red-Black Tree**, focusing on internal structure, performance, and typical usage:

<br>

| Feature                          | **Skip List**                                | **Red-Black Tree**                             |
|----------------------------------|-----------------------------------------------|-------------------------------------------------|
| **Structure**                   | Multiple levels of linked lists              | Binary search tree with color-coded nodes       |
| **Balancing Mechanism**         | Probabilistic (random level assignment)      | Deterministic (color rules + rotations)         |
| **Search / Insert / Delete**    | O(log n) average, O(n) worst                 | O(log n) worst-case                             |
| **Concurrency**                 | Easier to make thread-safe                  | Harder to make concurrent due to rotations      |
| **Memory Overhead**             | More (pointers per level)                   | Less (fewer pointers)                           |
| **Implementation Complexity**   | Easier and intuitive                        | More complex due to strict balancing rules      |
| **Use in Java**                 | `ConcurrentSkipListMap`                     | `TreeMap`                                       |
| **Use Case**                    | Concurrent sorted maps (multithreading)     | General purpose in-memory sorted maps           |

---

### Visual Analogy:

- **Skip List**: Like express lanes in a highway—some cars can skip to higher lanes to reach destination faster.
- **Red-Black Tree**: Like a perfectly maintained binary tree ensuring balance via strict rules.

Would you like visual diagrams for both to better understand the structure?


A **Skip List** is a probabilistic data structure that allows fast search, insertion, and deletion operations—similar to balanced trees—but using multiple layers of linked lists.

---

### **Key Concepts**

- A Skip List consists of **multiple levels of sorted linked lists**.
- Each level allows **“skipping”** over multiple nodes.
- Higher levels are **sparser** and created randomly.
- Search starts from the top level and proceeds down and right.

---

### **Operations and Complexity**

| Operation   | Average Time | Worst Time |
|-------------|--------------|-------------|
| Search      | O(log n)     | O(n)        |
| Insert      | O(log n)     | O(n)        |
| Delete      | O(log n)     | O(n)        |

---

### **Java Code Example**

Here is a simple generic `SkipList` implementation in Java:

```java
import java.util.Random;

public class SkipList<T extends Comparable<T>> {

    class Node {
        T value;
        Node[] forward;

        Node(T value, int level) {
            this.value = value;
            forward = new Node[level + 1];
        }
    }

    private static final int MAX_LEVEL = 16;
    private final Node head = new Node(null, MAX_LEVEL);
    private int level = 0;
    private final Random random = new Random();

    private int randomLevel() {
        int lvl = 0;
        while (random.nextBoolean() && lvl < MAX_LEVEL) {
            lvl++;
        }
        return lvl;
    }

    public void insert(T value) {
        Node[] update = new Node[MAX_LEVEL + 1];
        Node current = head;

        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && current.forward[i].value.compareTo(value) < 0) {
                current = current.forward[i];
            }
            update[i] = current;
        }

        current = current.forward[0];

        if (current == null || !current.value.equals(value)) {
            int lvl = randomLevel();

            if (lvl > level) {
                for (int i = level + 1; i <= lvl; i++) {
                    update[i] = head;
                }
                level = lvl;
            }

            Node newNode = new Node(value, lvl);
            for (int i = 0; i <= lvl; i++) {
                newNode.forward[i] = update[i].forward[i];
                update[i].forward[i] = newNode;
            }
        }
    }

    public boolean search(T value) {
        Node current = head;
        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && current.forward[i].value.compareTo(value) < 0) {
                current = current.forward[i];
            }
        }
        current = current.forward[0];
        return current != null && current.value.equals(value);
    }

    public void delete(T value) {
        Node[] update = new Node[MAX_LEVEL + 1];
        Node current = head;

        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && current.forward[i].value.compareTo(value) < 0) {
                current = current.forward[i];
            }
            update[i] = current;
        }

        current = current.forward[0];
        if (current != null && current.value.equals(value)) {
            for (int i = 0; i <= level; i++) {
                if (update[i].forward[i] != current) break;
                update[i].forward[i] = current.forward[i];
            }

            while (level > 0 && head.forward[level] == null) {
                level--;
            }
        }
    }
}
```

---

### **Use Cases**

- **Concurrent Maps**: `ConcurrentSkipListMap` and `ConcurrentSkipListSet` in Java use Skip List for thread-safe sorted collections.
- **In-memory sorted indexes**: Useful in cache and database engines.
- **Alternative to Balanced Trees**: Easier to implement, especially for concurrent environments.
- **Redis**: Skip Lists are used internally for sorted sets (`ZSET`).

---

Would you like a visual animation or example use in real-world APIs like Redis or Java SDK?