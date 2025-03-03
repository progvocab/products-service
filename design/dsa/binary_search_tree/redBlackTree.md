### **Red-Black Tree Explained with Code Examples and Comparison to Binary Search Tree (BST)**  

A **Red-Black Tree (RBT)** is a **self-balancing Binary Search Tree (BST)** where each node has an extra bit for denoting the color: **Red or Black**. It ensures that the tree remains balanced, making insertions, deletions, and lookups efficient.  

---

## **1. Properties of Red-Black Tree**
A Red-Black Tree follows these rules to maintain balance:  

1. **Every node is either Red or Black.**  
2. **The root node is always Black.**  
3. **A Red node cannot have a Red child (No two consecutive Red nodes).**  
4. **Every path from a node to its descendant NULL nodes must have the same number of Black nodes (Black-Height Property).**  
5. **Newly inserted nodes are always Red.**  

✅ **Time Complexity**:  
- **Insertion, Deletion, Search**: **O(log n)** (because the tree remains balanced).  

✅ **Space Complexity**:  
- **O(n)** (same as a regular BST, storing extra color bits).  

---

## **2. Comparison: Red-Black Tree vs. Binary Search Tree (BST)**  

| Feature                | **Red-Black Tree**            | **Binary Search Tree (BST)** |
|------------------------|-----------------------------|-----------------------------|
| **Balance**            | Self-balancing               | Can become skewed (unbalanced) |
| **Search Complexity**  | O(log n)                     | O(n) in worst case (skewed tree) |
| **Insertion Complexity** | O(log n)                    | O(n) in worst case (if unbalanced) |
| **Deletion Complexity** | O(log n)                    | O(n) in worst case |
| **Use Cases**          | Database indexes, memory allocators | Basic search & sorting algorithms |

**Example:**  
- A **BST** can become skewed (like a linked list) if elements are inserted in sorted order.  
- A **Red-Black Tree** keeps itself balanced, preventing worst-case O(n) operations.  

---

## **3. Implementation of a Red-Black Tree in Golang**  

### **A. Node Structure in Golang**  
Each node contains:  
- A **color** (Red or Black).  
- **Left and Right child pointers**.  
- **Parent pointer** (for easier rebalancing).  
- A **Key (value stored in the node)**.  

```go
package main

import (
	"fmt"
)

// Define Colors
const (
	RED   = true
	BLACK = false
)

// Node structure
type Node struct {
	data   int
	color  bool // true for RED, false for BLACK
	left   *Node
	right  *Node
	parent *Node
}

// Red-Black Tree Structure
type RedBlackTree struct {
	root *Node
}

// Create new node (Always Red initially)
func newNode(data int) *Node {
	return &Node{data: data, color: RED, left: nil, right: nil, parent: nil}
}
```

---

### **B. Left & Right Rotation Functions**  
Rotations are used to maintain balance during insertions and deletions.  

#### **Left Rotation**  
```go
func (tree *RedBlackTree) leftRotate(x *Node) {
	y := x.right
	x.right = y.left
	if y.left != nil {
		y.left.parent = x
	}
	y.parent = x.parent
	if x.parent == nil {
		tree.root = y
	} else if x == x.parent.left {
		x.parent.left = y
	} else {
		x.parent.right = y
	}
	y.left = x
	x.parent = y
}
```

#### **Right Rotation**  
```go
func (tree *RedBlackTree) rightRotate(x *Node) {
	y := x.left
	x.left = y.right
	if y.right != nil {
		y.right.parent = x
	}
	y.parent = x.parent
	if x.parent == nil {
		tree.root = y
	} else if x == x.parent.right {
		x.parent.right = y
	} else {
		x.parent.left = y
	}
	y.right = x
	x.parent = y
}
```

---

### **C. Insert and Fix Violations**  
When inserting a node, the **Red-Black Tree properties** might be violated.  
- If a violation occurs (e.g., two consecutive Red nodes), we **fix it using rotations and recoloring**.  

#### **Insert Function**
```go
func (tree *RedBlackTree) insert(data int) {
	newNode := newNode(data)

	// Standard BST Insert
	var parent *Node
	current := tree.root

	for current != nil {
		parent = current
		if newNode.data < current.data {
			current = current.left
		} else {
			current = current.right
		}
	}
	newNode.parent = parent

	if parent == nil {
		tree.root = newNode
	} else if newNode.data < parent.data {
		parent.left = newNode
	} else {
		parent.right = newNode
	}

	// Fix violations after insertion
	tree.fixInsert(newNode)
}
```

#### **Fixing Violations**  
```go
func (tree *RedBlackTree) fixInsert(node *Node) {
	for node.parent != nil && node.parent.color == RED {
		grandparent := node.parent.parent

		if node.parent == grandparent.left {
			uncle := grandparent.right
			if uncle != nil && uncle.color == RED {
				// Case 1: Uncle is Red → Recolor
				node.parent.color = BLACK
				uncle.color = BLACK
				grandparent.color = RED
				node = grandparent
			} else {
				if node == node.parent.right {
					// Case 2: Triangle Formation → Left Rotate
					node = node.parent
					tree.leftRotate(node)
				}
				// Case 3: Line Formation → Right Rotate
				node.parent.color = BLACK
				grandparent.color = RED
				tree.rightRotate(grandparent)
			}
		} else {
			uncle := grandparent.left
			if uncle != nil && uncle.color == RED {
				// Mirror Case 1: Uncle is Red
				node.parent.color = BLACK
				uncle.color = BLACK
				grandparent.color = RED
				node = grandparent
			} else {
				if node == node.parent.left {
					// Mirror Case 2: Right Rotate
					node = node.parent
					tree.rightRotate(node)
				}
				// Mirror Case 3: Left Rotate
				node.parent.color = BLACK
				grandparent.color = RED
				tree.leftRotate(grandparent)
			}
		}
	}
	tree.root.color = BLACK
}
```

---

## **4. Running the Red-Black Tree Implementation**  
### **Main Function**
```go
func main() {
	tree := &RedBlackTree{}
	tree.insert(10)
	tree.insert(20)
	tree.insert(30)
	tree.insert(40)
	tree.insert(50)
	tree.insert(25)

	fmt.Println("Red-Black Tree Insertions Completed!")
}
```
✅ **The tree remains balanced, ensuring O(log n) operations.**  

---

## **5. Summary**  

| Feature               | **Red-Black Tree**               | **Binary Search Tree (BST)**  |
|----------------------|--------------------------------|-----------------------------|
| **Balance**         | Always balanced                | Can become skewed (unbalanced) |
| **Search Time**     | O(log n)                        | O(n) in worst case |
| **Insertion Time**  | O(log n) with rotations        | O(n) in worst case |
| **Deletion Time**   | O(log n)                        | O(n) in worst case |
| **Best Use Cases**  | Databases (MongoDB, PostgreSQL), OS schedulers | General search trees |

---
## **6. Conclusion**  
- A **Red-Black Tree** ensures **logarithmic time complexity** for insertion, deletion, and search.  
- It is preferred over a **BST** when performance needs to be consistent and balanced.  
- Used in **databases (MongoDB, PostgreSQL)** and **memory allocators** for efficient management.  

Would you like additional details on **deletion in Red-Black Trees**?