
---

# 🌳 **Inorder Traversal in Python**

## 1. What is Inorder Traversal?

* **Definition**: In a **Binary Tree**, **inorder traversal** visits nodes in this order:

  1. Traverse the **left subtree**
  2. Visit the **root node**
  3. Traverse the **right subtree**

* For a **Binary Search Tree (BST)**, inorder traversal always gives a **sorted list** of values.

---

## 2. Example Tree

```
        4
       / \
      2   6
     / \  / \
    1  3 5  7
```

* Inorder traversal of this tree: **\[1, 2, 3, 4, 5, 6, 7]**

---

## 3. Python Code

### Recursive Inorder Traversal

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def inorder_recursive(root):
    if root:
        inorder_recursive(root.left)      # Visit left subtree
        print(root.value, end=" ")        # Visit root
        inorder_recursive(root.right)     # Visit right subtree


# Example usage
root = Node(4)
root.left = Node(2)
root.right = Node(6)
root.left.left = Node(1)
root.left.right = Node(3)
root.right.left = Node(5)
root.right.right = Node(7)

print("Inorder Traversal (Recursive):")
inorder_recursive(root)
```

**Output:**

```
Inorder Traversal (Recursive):
1 2 3 4 5 6 7
```

---

### Iterative Inorder Traversal (Using Stack)

```python
def inorder_iterative(root):
    stack = []
    current = root
    
    while stack or current:
        while current:              # Reach the leftmost node
            stack.append(current)
            current = current.left
        
        current = stack.pop()       # Visit node
        print(current.value, end=" ")
        
        current = current.right     # Move to right subtree


print("\nInorder Traversal (Iterative):")
inorder_iterative(root)
```

**Output:**

```
Inorder Traversal (Iterative):
1 2 3 4 5 6 7
```

---

## 4. Complexity

* **Time Complexity**:

  * Each node is visited once → **O(n)**
* **Space Complexity**:

  * Recursive → **O(h)** (stack space, where h = height of tree)
  * Iterative (stack) → **O(h)**

---

👉 Would you like me to also explain **Morris Traversal** for inorder (which avoids stack/recursion and works in **O(1) space**)?
