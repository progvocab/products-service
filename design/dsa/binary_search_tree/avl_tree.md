You can store a list of numbers in a **Binary Search Tree (BST)** to achieve **O(log n)** search time instead of **O(n)** in an unsorted list. Below is a Python implementation of a BST with **insert** and **search** operations.

### **Implementation**
```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        """ Insert value into the BST """
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)

    def search(self, value):
        """ Search for a value in the BST """
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def inorder_traversal(self):
        """ Inorder traversal (returns sorted values) """
        result = []
        self._inorder_helper(self.root, result)
        return result

    def _inorder_helper(self, node, result):
        if node:
            self._inorder_helper(node.left, result)
            result.append(node.value)
            self._inorder_helper(node.right, result)

# Example Usage
numbers = [10, 5, 20, 3, 7, 15, 25]
bst = BST()
for num in numbers:
    bst.insert(num)

# Searching for a number
print(bst.search(15))  # Output: True
print(bst.search(8))   # Output: False

# Printing the BST in sorted order
print(bst.inorder_traversal())  # Output: [3, 5, 7, 10, 15, 20, 25]
```

### **Time Complexity**
- **Insertion:** O(log n) (on average, but O(n) in the worst case if the tree is unbalanced)
- **Search:** O(log n) (same as above)
- **Inorder Traversal:** O(n)

If you want to keep the BST **balanced** (to ensure O(log n) search even in the worst case), consider using **AVL Tree** or **Red-Black Tree**.

Would you like an AVL Tree implementation as well?

# Adelson Velsky and Landis tree


Here's an **AVL Tree** implementation in Python. An **AVL Tree** is a self-balancing **Binary Search Tree (BST)** where the height difference (balance factor) between the left and right subtrees is at most 1. This ensures that search, insert, and delete operations remain **O(log n)** even in the worst case.  

---

### **AVL Tree Implementation**
```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  # Height is initially set to 1 for leaf nodes

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, root, value):
        """ Insert value into AVL tree and balance it """
        if not root:
            return Node(value)

        if value < root.value:
            root.left = self.insert(root.left, value)
        else:
            root.right = self.insert(root.right, value)

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)

        # Perform rotations if necessary
        if balance > 1 and value < root.left.value:
            return self.right_rotate(root)
        if balance < -1 and value > root.right.value:
            return self.left_rotate(root)
        if balance > 1 and value > root.left.value:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance < -1 and value < root.right.value:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def search(self, root, value):
        """ Search for a value in the AVL tree """
        if not root:
            return False
        if root.value == value:
            return True
        elif value < root.value:
            return self.search(root.left, value)
        else:
            return self.search(root.right, value)

    def left_rotate(self, node):
        """ Perform left rotation """
        right_child = node.right
        temp = right_child.left
        right_child.left = node
        node.right = temp
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        right_child.height = 1 + max(self.get_height(right_child.left), self.get_height(right_child.right))
        return right_child

    def right_rotate(self, node):
        """ Perform right rotation """
        left_child = node.left
        temp = left_child.right
        left_child.right = node
        node.left = temp
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        left_child.height = 1 + max(self.get_height(left_child.left), self.get_height(left_child.right))
        return left_child

    def get_height(self, node):
        """ Get height of a node """
        return node.height if node else 0

    def get_balance(self, node):
        """ Get balance factor of a node """
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def inorder_traversal(self, root, result=[]):
        """ Inorder traversal (sorted order) """
        if root:
            self.inorder_traversal(root.left, result)
            result.append(root.value)
            self.inorder_traversal(root.right, result)
        return result

# Example Usage
numbers = [10, 20, 30, 40, 50, 25]
avl = AVLTree()
root = None
for num in numbers:
    root = avl.insert(root, num)

# Searching for a number
print(avl.search(root, 25))  # Output: True
print(avl.search(root, 15))  # Output: False

# Printing the AVL Tree in sorted order
print(avl.inorder_traversal(root))  # Output: [10, 20, 25, 30, 40, 50]
```

---

### **Time Complexity**
| Operation | Complexity |
|-----------|-----------|
| Insertion | **O(log n)** |
| Deletion  | **O(log n)** |
| Search    | **O(log n)** |

---

### **Why Use an AVL Tree?**
1. **Guaranteed O(log n) search time**, unlike an unbalanced BST which can degrade to O(n).
2. **Self-balancing** ensures efficient insertions and deletions.
3. **Better for search-intensive applications**, such as database indexing.

Would you like a **delete operation** for the AVL tree as well?


AVL trees and Red-Black trees are both **self-balancing Binary Search Trees (BSTs)**, but they differ in how they maintain balance and their performance in different scenarios.

---

### **1. Key Differences Between AVL Tree and Red-Black Tree**
| Feature | **AVL Tree** | **Red-Black Tree** |
|---------|------------|----------------|
| **Balancing Factor** | Uses **height balancing** (balance factor: -1, 0, or 1). | Uses **color balancing** (Red/Black rules). |
| **Balance Condition** | Strictly balanced, meaning height difference between left and right subtrees is at most 1. | Less strictly balanced, meaning it allows slightly unbalanced trees. |
| **Rotations** | Requires **more rotations** to maintain strict balance, making insertions/deletions slightly slower. | Requires **fewer rotations**, making insertions and deletions faster. |
| **Search Time Complexity** | **O(log n)** (more efficient than Red-Black tree due to better balance). | **O(log n)** but can be slightly worse than AVL due to looser balance. |
| **Insertion Complexity** | **O(log n)** but needs **more rotations** to rebalance the tree. | **O(log n)** but with **fewer rotations**, making it faster for insert-heavy workloads. |
| **Deletion Complexity** | More complex than Red-Black trees, requires **more rotations** to maintain balance. | Simpler deletion handling, requiring **fewer rotations**. |
| **Best Use Case** | **Search-heavy** applications where read performance is critical (e.g., database indexing). | **Insert/delete-heavy** applications where writes are more frequent (e.g., memory allocators, OS schedulers). |

---

### **2. Example to Illustrate the Difference**
#### **AVL Tree Example (Height-Balanced)**
```plaintext
   30
  /  \
 20   40
/  \
10  25
```
- **Strict balancing** ensures all operations stay close to **O(log n)**.

#### **Red-Black Tree Example (Color-Balanced)**
```plaintext
    30 (Black)
   /     \
 20 (Red) 40 (Red)
 /
10 (Black)
```
- **Less strict balancing** allows **faster insertions/deletions**.

---

### **3. When to Use Which?**
| Scenario | Preferred Tree |
|----------|---------------|
| **Frequent reads/searches** | **AVL Tree** (because of better balance and faster lookups). |
| **Frequent insertions/deletions** | **Red-Black Tree** (because it requires fewer rotations). |
| **Memory-constrained environment** | **Red-Black Tree** (because it allows slightly unbalanced trees and uses fewer rotations). |
| **Databases & indexing** | **AVL Tree** (better search performance). |
| **OS schedulers, memory allocators** | **Red-Black Tree** (efficient insert/delete operations). |

---

### **4. Summary**
- **AVL Trees**: Better for read-heavy applications because they maintain strict balance.
- **Red-Black Trees**: Better for write-heavy applications because they require fewer rotations.

Would you like a code implementation for both AVL and Red-Black trees to compare them?


Here are Java implementations of **AVL Tree** and **Red-Black Tree** to illustrate the differences.  

---

## **1. AVL Tree Implementation (Self-Balancing)**
```java
class AVLTree {
    class Node {
        int key, height;
        Node left, right;

        Node(int key) {
            this.key = key;
            height = 1;
        }
    }

    private Node root;

    // Get height of a node
    private int height(Node node) {
        return (node == null) ? 0 : node.height;
    }

    // Get balance factor
    private int getBalance(Node node) {
        return (node == null) ? 0 : height(node.left) - height(node.right);
    }

    // Right rotate
    private Node rightRotate(Node y) {
        Node x = y.left;
        Node T2 = x.right;

        x.right = y;
        y.left = T2;

        y.height = Math.max(height(y.left), height(y.right)) + 1;
        x.height = Math.max(height(x.left), height(x.right)) + 1;

        return x;
    }

    // Left rotate
    private Node leftRotate(Node x) {
        Node y = x.right;
        Node T2 = y.left;

        y.left = x;
        x.right = T2;

        x.height = Math.max(height(x.left), height(x.right)) + 1;
        y.height = Math.max(height(y.left), height(y.right)) + 1;

        return y;
    }

    // Insert a node and rebalance
    private Node insert(Node node, int key) {
        if (node == null) return new Node(key);

        if (key < node.key)
            node.left = insert(node.left, key);
        else if (key > node.key)
            node.right = insert(node.right, key);
        else
            return node; // No duplicate keys

        node.height = 1 + Math.max(height(node.left), height(node.right));
        int balance = getBalance(node);

        // Left Heavy
        if (balance > 1 && key < node.left.key)
            return rightRotate(node);

        // Right Heavy
        if (balance < -1 && key > node.right.key)
            return leftRotate(node);

        // Left-Right Case
        if (balance > 1 && key > node.left.key) {
            node.left = leftRotate(node.left);
            return rightRotate(node);
        }

        // Right-Left Case
        if (balance < -1 && key < node.right.key) {
            node.right = rightRotate(node.right);
            return leftRotate(node);
        }

        return node;
    }

    // Public insert method
    public void insert(int key) {
        root = insert(root, key);
    }

    // In-order traversal
    public void inOrder(Node node) {
        if (node != null) {
            inOrder(node.left);
            System.out.print(node.key + " ");
            inOrder(node.right);
        }
    }

    public void printTree() {
        inOrder(root);
        System.out.println();
    }

    public static void main(String[] args) {
        AVLTree tree = new AVLTree();
        tree.insert(10);
        tree.insert(20);
        tree.insert(30);
        tree.insert(40);
        tree.insert(50);
        tree.insert(25);
        tree.printTree(); // Output: 10 20 25 30 40 50
    }
}
```
### **Key Features of AVL Tree**
- Maintains **strict balance**.
- Performs **more rotations** when inserting/deleting nodes.
- **Best for read-heavy applications** (faster searches).

---

## **2. Red-Black Tree Implementation (Less Strict Balancing)**
```java
class RedBlackTree {
    private static final boolean RED = true;
    private static final boolean BLACK = false;

    class Node {
        int key;
        Node left, right, parent;
        boolean color;

        Node(int key) {
            this.key = key;
            this.color = RED;
        }
    }

    private Node root;
    private Node NIL = new Node(-1); // Sentinel NIL node

    public RedBlackTree() {
        root = NIL;
    }

    // Left rotate
    private void leftRotate(Node x) {
        Node y = x.right;
        x.right = y.left;
        if (y.left != NIL)
            y.left.parent = x;

        y.parent = x.parent;
        if (x.parent == null)
            root = y;
        else if (x == x.parent.left)
            x.parent.left = y;
        else
            x.parent.right = y;

        y.left = x;
        x.parent = y;
    }

    // Right rotate
    private void rightRotate(Node x) {
        Node y = x.left;
        x.left = y.right;
        if (y.right != NIL)
            y.right.parent = x;

        y.parent = x.parent;
        if (x.parent == null)
            root = y;
        else if (x == x.parent.right)
            x.parent.right = y;
        else
            x.parent.left = y;

        y.right = x;
        x.parent = y;
    }

    // Insert and fix violations
    public void insert(int key) {
        Node newNode = new Node(key);
        newNode.left = newNode.right = NIL;

        Node parent = null;
        Node current = root;

        while (current != NIL) {
            parent = current;
            if (newNode.key < current.key)
                current = current.left;
            else
                current = current.right;
        }

        newNode.parent = parent;
        if (parent == null)
            root = newNode;
        else if (newNode.key < parent.key)
            parent.left = newNode;
        else
            parent.right = newNode;

        newNode.color = RED;
        fixInsert(newNode);
    }

    // Fix Red-Black Tree properties after insertion
    private void fixInsert(Node node) {
        while (node.parent != null && node.parent.color == RED) {
            Node grandparent = node.parent.parent;
            if (node.parent == grandparent.left) {
                Node uncle = grandparent.right;
                if (uncle.color == RED) {
                    node.parent.color = BLACK;
                    uncle.color = BLACK;
                    grandparent.color = RED;
                    node = grandparent;
                } else {
                    if (node == node.parent.right) {
                        node = node.parent;
                        leftRotate(node);
                    }
                    node.parent.color = BLACK;
                    grandparent.color = RED;
                    rightRotate(grandparent);
                }
            } else {
                Node uncle = grandparent.left;
                if (uncle.color == RED) {
                    node.parent.color = BLACK;
                    uncle.color = BLACK;
                    grandparent.color = RED;
                    node = grandparent;
                } else {
                    if (node == node.parent.left) {
                        node = node.parent;
                        rightRotate(node);
                    }
                    node.parent.color = BLACK;
                    grandparent.color = RED;
                    leftRotate(grandparent);
                }
            }
        }
        root.color = BLACK;
    }

    // In-order traversal
    private void inOrder(Node node) {
        if (node != NIL) {
            inOrder(node.left);
            System.out.print(node.key + " ");
            inOrder(node.right);
        }
    }

    public void printTree() {
        inOrder(root);
        System.out.println();
    }

    public static void main(String[] args) {
        RedBlackTree tree = new RedBlackTree();
        tree.insert(10);
        tree.insert(20);
        tree.insert(30);
        tree.insert(40);
        tree.insert(50);
        tree.insert(25);
        tree.printTree(); // Output: 10 20 25 30 40 50 (balanced)
    }
}
```
### **Key Features of Red-Black Tree**
- Maintains **less strict balance**.
- **Faster insertions & deletions** (fewer rotations).
- **Best for insert-heavy applications**.

---

### **Summary**
| Feature | **AVL Tree** | **Red-Black Tree** |
|---------|------------|----------------|
| **Read Performance** | Faster | Slightly slower |
| **Insertion/Deletion** | More rotations | Fewer rotations |
| **Best Use Case** | Search-heavy apps | Insert-heavy apps |

Would you like a comparison of performance benchmarks?

Here are Python implementations of **AVL Tree** and **Red-Black Tree** to illustrate their differences.  

---

## **1. AVL Tree Implementation**
```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def get_height(self, node):
        return node.height if node else 0

    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def right_rotate(self, y):
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1

        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1

        return y

    def insert(self, root, key):
        if not root:
            return Node(key)

        if key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:
            return root

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))

        balance = self.get_balance(root)

        # Left Heavy
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)

        # Right Heavy
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)

        # Left-Right Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # Right-Left Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def inorder(self, root):
        if root:
            self.inorder(root.left)
            print(root.key, end=" ")
            self.inorder(root.right)

tree = AVLTree()
root = None
for key in [10, 20, 30, 40, 50, 25]:
    root = tree.insert(root, key)

tree.inorder(root)  # Output: 10 20 25 30 40 50
```
### **Key Features of AVL Tree**
- Strictly balanced.
- More rotations on insert/delete.
- Ideal for read-heavy applications.

---

## **2. Red-Black Tree Implementation**
```python
class Node:
    def __init__(self, key, color="RED"):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

class RedBlackTree:
    def __init__(self):
        self.NIL = Node(0, "BLACK")
        self.root = self.NIL

    def insert(self, key):
        new_node = Node(key)
        new_node.left = new_node.right = self.NIL
        parent = None
        current = self.root

        while current != self.NIL:
            parent = current
            if new_node.key < current.key:
                current = current.left
            else:
                current = current.right

        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        new_node.color = "RED"
        self.fix_insert(new_node)

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y

        y.right = x
        x.parent = y

    def fix_insert(self, node):
        while node.parent and node.parent.color == "RED":
            grandparent = node.parent.parent
            if node.parent == grandparent.left:
                uncle = grandparent.right
                if uncle.color == "RED":
                    node.parent.color = uncle.color = "BLACK"
                    grandparent.color = "RED"
                    node = grandparent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.left_rotate(node)
                    node.parent.color = "BLACK"
                    grandparent.color = "RED"
                    self.right_rotate(grandparent)
            else:
                uncle = grandparent.left
                if uncle.color == "RED":
                    node.parent.color = uncle.color = "BLACK"
                    grandparent.color = "RED"
                    node = grandparent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.right_rotate(node)
                    node.parent.color = "BLACK"
                    grandparent.color = "RED"
                    self.left_rotate(grandparent)
        self.root.color = "BLACK"

    def inorder(self, node):
        if node != self.NIL:
            self.inorder(node.left)
            print(node.key, end=" ")
            self.inorder(node.right)

tree = RedBlackTree()
for key in [10, 20, 30, 40, 50, 25]:
    tree.insert(key)

tree.inorder(tree.root)  # Output: 10 20 25 30 40 50 (balanced)
```
### **Key Features of Red-Black Tree**
- Less strict balancing.
- Fewer rotations than AVL tree.
- Ideal for insert-heavy applications.

---

### **Summary of Differences**
| Feature | **AVL Tree** | **Red-Black Tree** |
|---------|------------|----------------|
| **Balancing** | Strict balancing | Less strict balancing |
| **Read Performance** | Faster | Slightly slower |
| **Insertion/Deletion** | More rotations | Fewer rotations |
| **Best For** | Read-heavy applications | Insert-heavy applications |

Would you like performance benchmarks comparing them?