##  B-Trees 

 
A **B-Tree** is a **self-balancing search tree** optimized for disk storage and large datasets. It generalizes a **binary search tree (BST)** by allowing **multiple keys per node** and having a **higher branching factor** to reduce tree height. B-Trees are widely used in **databases, file systems, and indexing structures**.


For a B-Tree of **order `m`**:
1. **Each node has at most `m` children**.
2. **Each node (except root) has at least ⌈m/2⌉ children**.
3. **Each node can have `m-1` keys**.
4. **Keys are stored in sorted order**.
5. **B-Trees are balanced ,All leaves are at the same depth**.
6. **Insertion and deletion operations keep the tree balanced**.
7. Designed for **minimizing disk I/O**, making them ideal for databases and file systems.

### **B-Tree   (order = 3)**

```
        [10, 20]
       /   |   \
    [5]  [15]  [25, 30]
```
 

### **Operations**
- **Search**: O(log n)
- **Insert/Delete**: O(log n), though insertions/deletions may cause splits/merges.
 
 
###  Use Cases
- **Databases (MySQL, PostgreSQL, MongoDB, etc.)** – Indexing.
- **File Systems (NTFS, HFS+, Ext4, etc.)** – Directory structure.
- **Key-Value Stores (LevelDB, RocksDB, etc.)** – Persistent storage.
- **Optimized for Disk Storage** – Reduces **disk I/O**.
- **Balanced Structure** – Ensures **O(log N)** search, insert, delete.
- **Used in File Systems** (HFS+).
- B-trees as the default indexing method for primary and foreign keys, enhancing join operations.
- facilitates quick lookups and range-based queries


 

### **Comparision**

 

* **Binary Search (on Sorted Arrays)** is ideal for **static in-memory datasets** where data rarely changes. It offers `O(log n)` search but **slow insertion and deletion** (`O(n)`) due to shifting elements. Common in read-heavy, immutable datasets.

* **B-Trees** are built for **I/O efficiency**, optimized for **disk-based systems** like **databases and file systems**. They store both **keys and data** in internal nodes, minimizing disk reads during search, insertion, and deletion — all in `O(log n)` time.

* **B+ Trees** are an extension of B-Trees used in **modern databases (MySQL, PostgreSQL, Oracle, SQL Server, NTFS)**. They keep **data only in leaf nodes** and link all leaves sequentially, enabling **fast range queries** and **better cache locality**.

* **Red-Black Trees** are **self-balancing binary search trees** designed for **in-memory operations** where fast updates are frequent. Used in **C++ STL `map`**, **Java `TreeMap`**, and **Linux kernel structures**, offering `O(log n)` for insert, delete, and search.
 


  
##   **B+ Tree**

A **B+ Tree** is an **extension of the B-Tree**, optimized for **range queries and disk-based storage**.


1. **All actual data records are stored only in leaf nodes**.
2. Internal nodes store **only keys** (used for indexing).
3. Leaf nodes are **linked** to each other (forming a **linked list**) — enabling **fast range and sequential access**.
4. Provides better **cache and disk utilization** due to uniform leaf size.

| Feature                  | **B-Tree**                                 | **B+ Tree**                                             |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------- |
| **Data storage**         | Data stored in **internal and leaf nodes** | Data stored **only in leaf nodes**                      |
| **Internal nodes**       | Contain both **keys and data**             | Contain only **keys** (for navigation)                  |
| **Leaf linkage**         | Leaf nodes **not linked**                  | Leaf nodes are **linked** for sequential access         |
| **Search time**          | `O(log n)`                                 | `O(log n)` (slightly better in practice for disk reads) |
| **Range queries**        | Slower (must traverse multiple branches)   | Faster (via linked leaves)                              |
| **Space utilization**    | Less efficient (mixed data and index)      | More efficient (compact internal nodes)                 |
| **Used in**              | In-memory data structures                  | **Databases, file systems, indexes**                    |
| **Insertion / Deletion** | `O(log n)`                                 | `O(log n)`                                              |
| **Traversal**            | Complex                                    | Simple (follow leaf links)                              |

 

Think of:

* **B-Tree** → like a **book** where every chapter (node) contains both an **index and full pages**.
* **B+ Tree** → like a **library** where the **catalog (internal nodes)** only stores references, and **actual books (leaf nodes)** are kept separately and arranged sequentially.
 
###  **B-Tree vs. Binary Search Tree (BST)**
| Feature | **B-Tree** | **Binary Search Tree** |
|---------|-----------|-----------------|
| **Height** | **O(log N)** | O(N) (worst case) |
| **Keys per Node** | **Multiple** | **One** |
| **Used in** | **Databases, File Systems** | **Memory-based data structures** |


### **B-Tree vs Red-Black Tree**

| Feature              | **B-Tree**                              | **Red-Black Tree**                     |
|----------------------|------------------------------------------|----------------------------------------|
| Type                 | Multi-way tree                          | Binary search tree                     |
| Node Keys            | Multiple keys per node                  | One key per node                       |
| Balancing            | Strictly balanced                       | Loosely balanced                       |
| Height               | Shorter (due to more keys per node)     | Taller                                 |
| Disk Access          | Optimized for **disk access**           | Not optimized for disk                 |
| Use Case             | **Databases, file systems**             | **In-memory** data structures          |
| Insertion/Deletion   | More complex (splits/merges)            | Simpler (rotations)                    |
| Search Time          | O(log n)                                | O(log n)                               |
| Cache Friendliness   | Higher (due to fewer disk reads)        | Lower                                  |




- **Use B-Trees** when:
  - Data is stored on disk (e.g., DB indexes, file systems)
  - Large block reads are preferred

- **Use Red-Black Trees** when:
  - Data is in-memory
  - Fast insertions and deletions are required with simple logic
 
 

Implements a **B-Tree of order `t`**, where each node:
- Stores **at most `2t-1` keys**.
- Has **at most `2t` children**.
- Ensures **all leaf nodes are at the same level**.

 
```python
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t  # Minimum degree
        self.leaf = leaf  # True if leaf node
        self.keys = []  # List of keys
        self.children = []  # List of children pointers

    def traverse(self):
        """Prints the B-Tree in sorted order"""
        for i in range(len(self.keys)):
            if not self.leaf:
                self.children[i].traverse()
            print(self.keys[i], end=" ")
        if not self.leaf:
            self.children[-1].traverse()

    def search(self, key):
        """Searches for a key in the tree"""
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1

        if i < len(self.keys) and self.keys[i] == key:
            return self

        if self.leaf:
            return None

        return self.children[i].search(key)


class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, True)
        self.t = t  # Minimum degree

    def traverse(self):
        """Print the B-Tree in sorted order"""
        if self.root:
            self.root.traverse()
        print()

    def search(self, key):
        """Search a key in the B-Tree"""
        return None if not self.root else self.root.search(key)

    def insert(self, key):
        """Insert a key into the B-Tree"""
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            new_root = BTreeNode(self.t, False)
            new_root.children.append(root)
            self._split_child(new_root, 0, root)
            self.root = new_root

        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        """Insert key into a node that is not full"""
        i = len(node.keys) - 1

        if node.leaf:
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.t) - 1:
                self._split_child(node, i, node.children[i])
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i, full_child):
        """Split a full child node into two"""
        t = self.t
        new_child = BTreeNode(t, full_child.leaf)
        parent.keys.insert(i, full_child.keys[t - 1])
        parent.children.insert(i + 1, new_child)

        new_child.keys = full_child.keys[t:]
        full_child.keys = full_child.keys[:t - 1]

        if not full_child.leaf:
            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

    def delete(self, key):
        """Deletes a key from the B-Tree"""
        if not self.root:
            return

        self._delete_recursive(self.root, key)

        if len(self.root.keys) == 0:
            if self.root.leaf:
                self.root = None
            else:
                self.root = self.root.children[0]

    def _delete_recursive(self, node, key):
        """Recursively delete a key from the tree"""
        t = self.t
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and node.keys[i] == key:
            if node.leaf:
                node.keys.pop(i)
            else:
                node.keys[i] = self._get_predecessor(node.children[i])
                self._delete_recursive(node.children[i], node.keys[i])
        else:
            if node.leaf:
                return

            if len(node.children[i].keys) < t:
                self._fill(node, i)

            if i < len(node.keys) and key == node.keys[i]:
                self._delete_recursive(node.children[i], key)
            else:
                self._delete_recursive(node.children[i - 1], key)

    def _fill(self, parent, i):
        """Ensures a child has at least t keys before deletion"""
        if i != 0 and len(parent.children[i - 1].keys) >= self.t:
            self._borrow_from_prev(parent, i)
        elif i != len(parent.keys) and len(parent.children[i + 1].keys) >= self.t:
            self._borrow_from_next(parent, i)
        else:
            if i != len(parent.keys):
                self._merge(parent, i)
            else:
                self._merge(parent, i - 1)

    def _borrow_from_prev(self, parent, i):
        """Borrow a key from the previous sibling"""
        child = parent.children[i]
        sibling = parent.children[i - 1]

        child.keys.insert(0, parent.keys[i - 1])
        parent.keys[i - 1] = sibling.keys.pop()

        if not sibling.leaf:
            child.children.insert(0, sibling.children.pop())

    def _borrow_from_next(self, parent, i):
        """Borrow a key from the next sibling"""
        child = parent.children[i]
        sibling = parent.children[i + 1]

        child.keys.append(parent.keys[i])
        parent.keys[i] = sibling.keys.pop(0)

        if not sibling.leaf:
            child.children.append(sibling.children.pop(0))

    def _merge(self, parent, i):
        """Merge two children of a node"""
        child = parent.children[i]
        sibling = parent.children[i + 1]

        child.keys.append(parent.keys.pop(i))
        child.keys.extend(sibling.keys)
        child.children.extend(sibling.children)

        parent.children.pop(i + 1)

    def _get_predecessor(self, node):
        """Finds the largest key in the left subtree"""
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1]
```

---

###   Example Usage**
```python
btree = BTree(3)  # Order 3 (t=3)

# Insert keys
keys = [10, 20, 5, 6, 12, 30, 7, 17]
for key in keys:
    btree.insert(key)

print("B-Tree after insertion:")
btree.traverse()

# Search for a key
search_key = 12
found = btree.search(search_key)
print(f"\nSearch for key {search_key}: {'Found' if found else 'Not Found'}")

# Delete a key
btree.delete(6)
print("\nB-Tree after deleting 6:")
btree.traverse()
```

 



 



 


