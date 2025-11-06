### **Searching in a Binary Search Tree (BST)**  

A **Binary Search Tree (BST)** is a tree where:  
- The **left child** contains values **smaller** than the root.  
- The **right child** contains values **greater** than the root.  

Searching in a BST is efficient, with an **average time complexity of O(log n)**.

 

## **1. Recursive Search in BST**  
- If the key is **equal** to the root, return the node.  
- If the key is **less than** the root, search in the **left subtree**.  
- If the key is **greater than** the root, search in the **right subtree**.  
  
 

## **2. Iterative Search in BST**  
- Uses a **loop instead of recursion** (saves function call overhead).  
- Similar logic as recursion, but traverses the tree iteratively.

 ## **Takeaways**
  **Recursive search** is elegant but uses extra function calls.  
  **Iterative search** avoids recursion overhead.  
  **Time complexity is O(log n) for balanced BSTs** but **O(n) for skewed trees**.  

## **3. Time Complexity**
| Case | Complexity |
|------|------------|
| **Best Case** (Root is the target) | **O(1)** |
| **Average Case** (Balanced BST) | **O(log n)** |
| **Worst Case** (Skewed BST) | **O(n)** |


 
```
        10
       /  \
      5    15
     / \   /  \
    2   7 12  18
```
- Searching for `7` → **Root → Left → Right**  
- Searching for `12` → **Root → Right → Left**  


 **Balanced BST (O(log n))** → Efficient search.  
 **Unbalanced BST (O(n))** → Degrades to linear search (like a linked list).  
 

##   **1. Unbalanced Binary Search Tree (BST)**

###   Description

* A **simple BST** where each node’s left child < parent < right child.
* No guarantee of height balance.

###   Characteristics

* Structure depends on **insertion order**.
* If data is **sorted**, the tree degenerates into a **linked list**.

###   Complexity

| Operation | Best Case | Worst Case |
| --------- | --------- | ---------- |
| Search    | O(log n)  | O(n)       |
| Insertion | O(log n)  | O(n)       |
| Deletion  | O(log n)  | O(n)       |

###   Notes

* Easy to implement.
* Performs poorly for **sorted or skewed input**.
* Used for **educational or simple lookups** when data is small.

 
##   **2. Balanced Binary Search Trees**

Balanced BSTs ensure that the **height difference** between subtrees is limited — maintaining near `O(log n)` operations.

 

###   **2.1 AVL Tree (Adelson-Velsky and Landis)**

###    Description

* Strictly balanced: for every node, the **height difference ≤ 1**.
* Uses **rotations** after insertions/deletions to maintain balance.

###   Complexity

| Operation | Time Complexity |
| --------- | --------------- |
| Search    | O(log n)        |
| Insert    | O(log n)        |
| Delete    | O(log n)        |

###   Notes

* **Excellent for read-heavy** workloads (since it’s tightly balanced).
* Slightly **more rotations** → higher insertion overhead.
* Used in **databases and search engines** for fast lookups.

 

###   **2.2 Red-Black Tree**

###   Description

* Loosely balanced binary search tree.
* Each node is **red or black**, following color rules to ensure balance.

###   Characteristics

* Height is at most **2× log₂(n+1)**.
* Fewer rotations than AVL (faster updates, slightly slower reads).

###   Complexity

| Operation | Time Complexity |
| --------- | --------------- |
| Search    | O(log n)        |
| Insert    | O(log n)        |
| Delete    | O(log n)        |

###   Notes

* **Write-heavy** workloads benefit from fewer rotations.
* Used in **C++ STL map/set**, **Java TreeMap**, **Linux kernel**.

---

###   **2.3 Splay Tree**

###    Description

* Self-adjusting tree that **moves recently accessed nodes** near the root (via rotations).

###   Characteristics

* No strict balance property.
* Frequently accessed elements become faster to reach.

###   Complexity

| Operation | Amortized Time |
| --------- | -------------- |
| Search    | O(log n)       |
| Insert    | O(log n)       |
| Delete    | O(log n)       |

###   Notes

* Great for **cache locality and access patterns**.
* Used in **network routing tables** and **data compression**.

---

###   **2.4 Treap (Tree + Heap)**

###   Description

* Randomized BST where each node has a **key (BST order)** and a **priority (heap order)**.
* Expected height = O(log n).

###   Complexity

| Operation | Expected Time |
| --------- | ------------- |
| Search    | O(log n)      |
| Insert    | O(log n)      |
| Delete    | O(log n)      |

###   Notes

* **Randomization** prevents skewed trees.
* Simpler to implement than AVL/Red-Black.
* Used in **probabilistic data structures**.

---

###   **2.5 Weight-Balanced Tree / Scapegoat Tree**

###   Description

* Maintains balance based on **node counts** (weight) instead of height.
* Rebuilds subtrees when imbalance exceeds threshold.

###   Complexity

| Operation | Amortized Time |
| --------- | -------------- |
| Search    | O(log n)       |
| Insert    | O(log n)       |
| Delete    | O(log n)       |

###   Notes

* **Good trade-off** between AVL strictness and Red-Black efficiency.
* Simpler logic — no color or height tracking.

---

##   **3. Summary Comparison Table**

| Tree Type          | Balanced?         | Height             | Search    | Insert    | Delete    | Rebalancing Method        | Use Case               |
| ------------------ | ----------------- | ------------------ | --------- | --------- | --------- | ------------------------- | ---------------------- |
| **Unbalanced BST** |  No              | O(n) worst         | O(n)      | O(n)      | O(n)      | None                      | Small datasets         |
| **AVL Tree**       |   Yes (strict)    | O(log n)           | O(log n)  | O(log n)  | O(log n)  | Rotations based on height | Read-heavy             |
| **Red-Black Tree** |   Yes (loose)     | O(log n)           | O(log n)  | O(log n)  | O(log n)  | Rotations + color fix     | Write-heavy            |
| **Splay Tree**     |   Self-adjusting | O(log n) amortized | O(log n)* | O(log n)* | O(log n)* | Rotations (splaying)      | Recently accessed data |
| **Treap**          |  Probabilistic   | O(log n) expected  | O(log n)  | O(log n)  | O(log n)  | Randomized rotation       | Random data            |
| **Scapegoat Tree** |   Amortized       | O(log n)           | O(log n)  | O(log n)  | O(log n)  | Rebuild subtrees          | Simpler balanced BST   |

---

##   Key Takeaways

* **Unbalanced BST**: Simple but risky (can degrade to O(n)).
* **AVL Tree**: Very fast search, slightly costlier insert/delete.
* **Red-Black Tree**: Great all-rounder — widely used in languages and OS kernels.
* **Splay Tree**: Optimized for access frequency.
* **Treap**: Randomly balanced, easy to implement.
* **Scapegoat/Weight-balanced**: Balances infrequently but efficiently.

