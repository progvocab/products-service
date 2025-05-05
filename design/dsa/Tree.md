Here’s a **comprehensive categorized list of tree data structures**, grouped by their **type, purpose, and structural properties**:

---

### **1. Basic Tree Types**

| Tree Type        | Description                                         |
| ---------------- | --------------------------------------------------- |
| **General Tree** | A tree where nodes can have any number of children. |
| **Binary Tree**  | Each node has at most two children.                 |
| **Ternary Tree** | Each node has up to three children.                 |
| **N-ary Tree**   | Generalized tree with up to N children per node.    |
| **K-ary Tree**   | Fixed number of children (k) for each node.         |

---

### **2. Binary Tree Variants**

| Tree Type                | Description                                                              |
| ------------------------ | ------------------------------------------------------------------------ |
| **Full Binary Tree**     | Every node has 0 or 2 children.                                          |
| **Complete Binary Tree** | All levels filled except possibly the last; left-aligned.                |
| **Perfect Binary Tree**  | All internal nodes have 2 children and all leaves are at same level.     |
| **Balanced Binary Tree** | Height difference between left and right subtree is limited (e.g., ≤ 1). |
| **Degenerate Tree**      | Each parent has only one child; behaves like a linked list.              |

---

### **3. Binary Search Tree (BST) Variants**

| Tree Type              | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| **Binary Search Tree** | Left < Root < Right ordering rule.                             |
| **Self-Balancing BST** | Auto-balances during insert/delete to maintain optimal height. |

#### Self-Balancing BSTs:

* **AVL Tree** – Balances by rotation; strict balancing (height-balance).
* **Red-Black Tree** – Balanced by coloring nodes and rotation (used in Java's `TreeMap`, `TreeSet`).
* **Splay Tree** – Moves recently accessed elements to the root.
* **Treap** – Combines BST and heap properties.
* **Scapegoat Tree** – Balances occasionally via rebuilding subtrees.
* **Tango Tree** – Used in competitive analysis.

---

### **4. Heap Trees**

| Tree Type          | Description                                               |
| ------------------ | --------------------------------------------------------- |
| **Binary Heap**    | Complete binary tree, used in priority queues.            |
| **Min Heap**       | Root is minimum; parents ≤ children.                      |
| **Max Heap**       | Root is maximum; parents ≥ children.                      |
| **Binomial Heap**  | Supports efficient merge operations.                      |
| **Fibonacci Heap** | Supports fast decrease-key; used in Dijkstra’s algorithm. |
| **Pairing Heap**   | Simplified alternative to Fibonacci heap.                 |

---

### **5. B-Trees and Variants**

| Tree Type    | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| **B-Tree**   | Balanced tree for large data blocks (used in databases/filesystems). |
| **B+ Tree**  | Leaf nodes are linked in a list; internal nodes contain keys only.   |
| **B* Tree*\* | Variant of B+ with better space utilization.                         |

---

### **6. Trie-based Trees**

| Tree Type                      | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| **Trie (Prefix Tree)**         | For string keys, stores characters at each level.   |
| **Radix Tree / Patricia Tree** | Compressed trie with space optimization.            |
| **Suffix Tree**                | All suffixes of a string; used in pattern matching. |

---

### **7. Multi-dimensional Trees**

| Tree Type                              | Description                                          |
| -------------------------------------- | ---------------------------------------------------- |
| **Segment Tree**                       | For range queries on arrays.                         |
| **Fenwick Tree (Binary Indexed Tree)** | Efficient prefix sums.                               |
| **KD-Tree**                            | K-dimensional space partitioning.                    |
| **Quad Tree**                          | Divides 2D space into 4 quadrants; used in graphics. |
| **Octree**                             | Divides 3D space into 8 regions.                     |
| **R-Tree**                             | Spatial indexing; used in GIS systems.               |

---

### **8. Specialized Trees**

| Tree Type                    | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| **Suffix Automaton**         | Optimized suffix tree for substring queries.         |
| **Decision Tree**            | Used in machine learning models.                     |
| **Expression Tree**          | Nodes represent expressions/operators.               |
| **Parse Tree / Syntax Tree** | Represents source code grammar.                      |
| **Game Tree**                | Represents possible game moves (e.g., chess).        |
| **Interval Tree**            | Manages overlapping intervals efficiently.           |
| **Cartesian Tree**           | Combines binary tree and heap for Cartesian sorting. |

---

Would you like a diagrammatic classification or code samples for specific types?
