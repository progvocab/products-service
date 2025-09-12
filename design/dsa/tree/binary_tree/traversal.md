let’s cover **Depth First Search (DFS)** and **Breadth First Search (BFS)** on a **binary tree** in Python.

---

# 🔹 Binary Tree Node Definition

We’ll use a simple class to define tree nodes:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

---

# 🔹 1. Depth First Search (DFS)

DFS explores as deep as possible before backtracking.
For trees, we usually write it **recursively** or with a **stack**.

### (a) Recursive DFS Traversals

* **Preorder (Root → Left → Right)**
* **Inorder (Left → Root → Right)**
* **Postorder (Left → Right → Root)**

```python
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

---

### (b) Iterative DFS (using stack)

```python
def dfs_iterative(root):
    if not root:
        return []
    
    stack, result = [root], []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

✅ This is a **preorder traversal**.

---

# 🔹 2. Breadth First Search (BFS)

BFS explores level by level (root, then children, then grandchildren).
We use a **queue**.

```python
from collections import deque

def bfs(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
```

---

# 🔹 Example Usage

```python
# Build a sample tree:
#        1
#       / \
#      2   3
#     / \   \
#    4   5   6

root = TreeNode(1)
root.left = TreeNode(2, TreeNode(4), TreeNode(5))
root.right = TreeNode(3, None, TreeNode(6))

print("Preorder DFS:", preorder(root))     # [1, 2, 4, 5, 3, 6]
print("Inorder DFS:", inorder(root))       # [4, 2, 5, 1, 3, 6]
print("Postorder DFS:", postorder(root))   # [4, 5, 2, 6, 3, 1]
print("Iterative DFS:", dfs_iterative(root)) # [1, 2, 4, 5, 3, 6]
print("BFS:", bfs(root))                   # [1, 2, 3, 4, 5, 6]
```

---

# 🔹 Summary

| Algorithm | Data Structure       | Typical Order                |
| --------- | -------------------- | ---------------------------- |
| DFS       | Stack (or recursion) | Preorder, Inorder, Postorder |
| BFS       | Queue                | Level-order traversal        |

---

👉 Do you want me to also show how DFS and BFS differ when used for **searching a target value** in a binary tree?

---

---
Traversing a **binary tree** means visiting all its nodes in a specific order. There are **three main types** of traversal:  

## **1. Depth-First Traversal (DFS)**  
DFS explores as far as possible along a branch before backtracking. It includes:  
### **a) Inorder Traversal (Left → Root → Right)**
- Used for **retrieving sorted data** from a Binary Search Tree (BST).  

**Example:**
```go
package main

import "fmt"

type Node struct {
    data  int
    left  *Node
    right *Node
}

func inorder(root *Node) {
    if root != nil {
        inorder(root.left)
        fmt.Print(root.data, " ")
        inorder(root.right)
    }
}

func main() {
    root := &Node{10, &Node{5, nil, nil}, &Node{15, nil, nil}}
    inorder(root) // Output: 5 10 15
}
```

---

### **b) Preorder Traversal (Root → Left → Right)**
- Used for **creating a copy of the tree** or **prefix expressions**.

```go
func preorder(root *Node) {
    if root != nil {
        fmt.Print(root.data, " ")
        preorder(root.left)
        preorder(root.right)
    }
}
```

**Output for the same tree** → `10 5 15`

---

### **c) Postorder Traversal (Left → Right → Root)**
- Used for **deleting nodes** or **postfix expressions**.

```go
func postorder(root *Node) {
    if root != nil {
        postorder(root.left)
        postorder(root.right)
        fmt.Print(root.data, " ")
    }
}
```

**Output for the same tree** → `5 15 10`

---

## **2. Breadth-First Traversal (BFS) / Level Order Traversal**
- Visits nodes **level by level** (top to bottom, left to right).  
- Implemented using a **queue**.

```go
package main

import (
    "fmt"
    "container/list"
)

func levelOrder(root *Node) {
    if root == nil {
        return
    }
    queue := list.New()
    queue.PushBack(root)

    for queue.Len() > 0 {
        node := queue.Remove(queue.Front()).(*Node)
        fmt.Print(node.data, " ")

        if node.left != nil {
            queue.PushBack(node.left)
        }
        if node.right != nil {
            queue.PushBack(node.right)
        }
    }
}

func main() {
    root := &Node{10, &Node{5, nil, nil}, &Node{15, nil, nil}}
    levelOrder(root) // Output: 10 5 15
}
```

---

## **Summary Table**
| Traversal Type  | Order | Use Case |
|-----------------|-------|----------|
| **Inorder**     | L → Root → R | Sorting BST |
| **Preorder**    | Root → L → R | Tree copying |
| **Postorder**   | L → R → Root | Node deletion |
| **Level Order** | Level by level | BFS search |

Would you like a different variation or explanation?