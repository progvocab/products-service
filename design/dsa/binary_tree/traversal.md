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