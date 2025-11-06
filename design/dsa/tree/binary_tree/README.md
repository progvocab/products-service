# **Binary Trees**  

##   **1. Definition**

A **Binary Tree** is a hierarchical data structure in which each node has at most **two children** — called the **left** and **right** child.

Formally:
Each node `N` can have
`N.left`, `N.right`, and a `N.value`.

---

##   **2. Types of Binary Trees**

| Type                         | Description                                                                    | Example                                              |
| ---------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **Full Binary Tree**         | Every node has 0 or 2 children.                                                | ![Full Binary Tree](https://i.imgur.com/dcP1m5r.png) |
| **Complete Binary Tree**     | All levels filled except possibly last, filled from left to right.             | Used in **Heaps**                                    |
| **Perfect Binary Tree**      | All internal nodes have 2 children **and all leaves are at the same level**.   | Height = log₂(n+1) - 1                               |
| **Balanced Binary Tree**     | Height difference between left and right subtrees is small (e.g., ≤ 1 in AVL). | Used in **AVL, Red-Black Tree**                      |
| **Degenerate (Skewed) Tree** | Each parent has only one child → behaves like a **linked list**.               | Unbalanced case                                      |
| **Binary Search Tree (BST)** | Left child < Parent < Right child.                                             | Supports O(log n) search                             |
| **Threaded Binary Tree**     | Uses “threads” (pointers) for efficient inorder traversal without recursion.   | Optimized traversal                                  |

---

##   **3. Key Properties**

| Property                             | Formula / Definition  |
| ------------------------------------ | --------------------- |
| **Maximum nodes at level `L`**       | `2^L`                 |
| **Maximum nodes in height `h` tree** | `2^(h+1) - 1`         |
| **Minimum height for `n` nodes**     | `ceil(log₂(n+1)) - 1` |
| **Height of perfect binary tree**    | `log₂(n+1) - 1`       |
| **Leaf nodes (perfect tree)**        | `(n + 1) / 2`         |
| **Internal nodes (perfect tree)**    | `(n - 1) / 2`         |

---

##   **4. Basic Operations**

| Operation                      | Description                               | Time Complexity (Balanced) |
| ------------------------------ | ----------------------------------------- | -------------------------- |
| **Insertion**                  | Add a new node (depending on order rule). | O(log n)                   |
| **Deletion**                   | Remove a node and restructure.            | O(log n)                   |
| **Search**                     | Find a value (depends on type).           | O(log n)                   |
| **Traversal**                  | Visit nodes in a specific order.          | O(n)                       |
| **Height / Depth calculation** | Compute distance from root to leaf.       | O(n)                       |

---

##   **5. Tree Traversals**

###   **Depth-First Traversals (DFS)**

1. **Preorder (Root → Left → Right)** — Used to copy trees.
2. **Inorder (Left → Root → Right)** — Produces **sorted order** in BSTs.
3. **Postorder (Left → Right → Root)** — Used to delete/free trees.

###   **Breadth-First Traversal (BFS)**

* **Level Order Traversal** — Visit nodes level by level (uses Queue).

---

##   **6. Real-World Applications**

| Application                       | Use Case                                                   |
| --------------------------------- | ---------------------------------------------------------- |
| **Expression Trees**              | Represent arithmetic expressions (used in compilers).      |
| **Binary Search Trees (BSTs)**    | Fast searching, insertion, deletion (e.g., symbol tables). |
| **Heaps (Complete Binary Trees)** | Priority queues and heap sort.                             |
| **Huffman Coding Tree**           | Used in data compression (e.g., JPEG, MP3).                |
| **Routing Tables**                | Network routing algorithms.                                |
| **Game Trees / Decision Trees**   | AI search and machine learning models.                     |

---

##   **7. Time and Space Complexity**

| Operation | Balanced Tree | Unbalanced Tree |
| --------- | ------------- | --------------- |
| Search    | O(log n)      | O(n)            |
| Insert    | O(log n)      | O(n)            |
| Delete    | O(log n)      | O(n)            |
| Traversal | O(n)          | O(n)            |
| Space     | O(n)          | O(n)            |

---

##   **8. Key Takeaways**

* Binary Tree is a **foundation** for more advanced trees:
  **BST → AVL → Red-Black → B-Tree → B+ Tree → Segment Tree → Trie**, etc.
* **Height determines efficiency** — the more balanced, the faster the operations.
* Traversals are crucial for understanding algorithms on trees (DFS, BFS).
* In **in-memory** systems → Binary / AVL / Red-Black trees are preferred.
* For **disk-based** systems → Multiway trees like **B-Tree / B+ Tree** are used.

