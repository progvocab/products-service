### **Searching in a Binary Search Tree (BST)**  

A **Binary Search Tree (BST)** is a tree where:  
- The **left child** contains values **smaller** than the root.  
- The **right child** contains values **greater** than the root.  

Searching in a BST is efficient, with an **average time complexity of O(log n)**.

---

## **1. Recursive Search in BST**  
- If the key is **equal** to the root, return the node.  
- If the key is **less than** the root, search in the **left subtree**.  
- If the key is **greater than** the root, search in the **right subtree**.  

### **Code Example (Recursive Search in Go)**
```go
package main

import "fmt"

type Node struct {
    data  int
    left  *Node
    right *Node
}

// Recursive function to search for a value in a BST
func search(root *Node, key int) *Node {
    if root == nil || root.data == key {
        return root
    }
    if key < root.data {
        return search(root.left, key)
    }
    return search(root.right, key)
}

func main() {
    root := &Node{10, &Node{5, nil, nil}, &Node{15, nil, nil}}

    key := 15
    result := search(root, key)
    if result != nil {
        fmt.Println("Found:", result.data)
    } else {
        fmt.Println("Not found")
    }
}
```
**Output:**  
```
Found: 15
```

---

## **2. Iterative Search in BST**  
- Uses a **loop instead of recursion** (saves function call overhead).  
- Similar logic as recursion, but traverses the tree iteratively.

### **Code Example (Iterative Search in Go)**
```go
func iterativeSearch(root *Node, key int) *Node {
    for root != nil {
        if key == root.data {
            return root
        } else if key < root.data {
            root = root.left
        } else {
            root = root.right
        }
    }
    return nil
}
```

---

## **3. Time Complexity**
| Case | Complexity |
|------|------------|
| **Best Case** (Root is the target) | **O(1)** |
| **Average Case** (Balanced BST) | **O(log n)** |
| **Worst Case** (Skewed BST) | **O(n)** |

🔹 **Balanced BST (O(log n))** → Efficient search.  
🔹 **Unbalanced BST (O(n))** → Degrades to linear search (like a linked list).  

---

## **4. Example BST**
```
        10
       /  \
      5    15
     / \   /  \
    2   7 12  18
```
- Searching for `7` → **Root → Left → Right**  
- Searching for `12` → **Root → Right → Left**  

---

## **5. Key Takeaways**
✅ **Recursive search** is elegant but uses extra function calls.  
✅ **Iterative search** avoids recursion overhead.  
✅ **Time complexity is O(log n) for balanced BSTs** but **O(n) for skewed trees**.  

Would you like an example of handling an unbalanced BST?