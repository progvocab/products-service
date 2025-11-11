Excellent — let’s dive into the **internals of `TreeMap`** in Java and see how **`get()`** and **`put()`** really work under the hood.

---

## 🌳 **TreeMap Overview**

* `TreeMap` is a **Red-Black Tree**–based implementation of the `NavigableMap` interface.
* It stores **key-value pairs in sorted order** (by natural ordering or by a provided `Comparator`).
* Every node (`Entry<K, V>`) contains:

  ```java
  class Entry<K,V> {
      K key;
      V value;
      Entry<K,V> left;
      Entry<K,V> right;
      Entry<K,V> parent;
      boolean color; // true = RED, false = BLACK
  }
  ```

---

## ⚙️ **1. put(K key, V value)** — *Insertion logic*

When you call `map.put(key, value)`:

1. **If root is null** → create a new black root node.

   ```java
   if (root == null)
       root = new Entry<>(key, value, null);
   ```

2. **Otherwise**, start from the root:

   * Compare the `key` with the current node’s key using:

     * `compare(key, node.key)` → uses `Comparator` if present, else `Comparable`.
   * Move **left** if `key < node.key`, **right** if `key > node.key`.
   * Continue until you find a null child — that’s where the new node goes.

3. **Insert the node** and color it **RED** by default.

4. **Fix tree balance**:

   * Call `fixAfterInsertion()` — this restores the **Red-Black Tree properties**.
   * Operations: color flips, rotations (left or right) to maintain balance and ordering.

5. If a **duplicate key** is found (compare = 0):

   * It **overwrites** the value of the existing node.
   * No new node is created, and no rebalancing occurs.

---

## 🔍 **2. get(Object key)** — *Search logic*

When you call `map.get(key)`:

1. Start from the **root node**.
2. Compare the given `key` with the current node’s key:

   * If `compare(key, node.key) < 0` → go **left**.
   * If `> 0` → go **right**.
   * If `== 0` → found! return `node.value`.
3. Continue recursively (or iteratively) until:

   * The key is found → return value.
   * Or you reach a null node → return `null`.

Because it’s a **balanced tree**, both `get()` and `put()` operations are **O(log n)** on average and in the worst case.

---

## 🧠 **Extra Details**

* The Red-Black Tree ensures the tree remains **approximately balanced**:

  * No path from root to leaf is more than **twice as long** as any other.
  * So even in skewed data insertions, performance remains stable.

* `TreeMap` does **not allow null keys**, because it relies on comparison logic.

---

### 🩵 In Short:

| Operation | Description                                                 | Time Complexity |
| --------- | ----------------------------------------------------------- | --------------- |
| **put()** | Binary search for position → Insert → Rebalance (if needed) | O(log n)        |
| **get()** | Binary search through tree by comparator/comparable         | O(log n)        |

---

Would you like me to show the **actual pseudocode (simplified Java source logic)** for `fixAfterInsertion()` and `getEntry()` from the OpenJDK TreeMap source? It’s quite insightful to see how rotations and color flips work.
