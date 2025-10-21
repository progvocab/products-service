# Tree
Directed Acyclic Graph
### Binary Tree

```python
from collections import deque

# ------------------------------------------------------------
#   Base Node class (used by all trees)
# ------------------------------------------------------------
class Node:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None
        self.height = 1  # Used only by AVL


# ------------------------------------------------------------
#   BinaryTree (Generic, Unordered)
# ------------------------------------------------------------
class BinaryTree:
    def __init__(self):
        self.root = None

    def add_left(self, parent, val):
        if parent.left is None:
            parent.left = Node(val)

    def add_right(self, parent, val):
        if parent.right is None:
            parent.right = Node(val)

    # Generic traversals (same for all types)
    def inorder(self, root):
        return self.inorder(root.left) + [root.val] + self.inorder(root.right) if root else []

    def preorder(self, root):
        return [root.val] + self.preorder(root.left) + self.preorder(root.right) if root else []

    def postorder(self, root):
        return self.postorder(root.left) + self.postorder(root.right) + [root.val] if root else []

    def bfs(self, root):
        if not root: return []
        q, res = deque([root]), []
        while q:
            node = q.popleft()
            res.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        return res
```
### Binary Search Tree
```python
# ------------------------------------------------------------
#   BinarySearchTree (Ordered, Unbalanced)
# ------------------------------------------------------------
class BinarySearchTree(BinaryTree):

    def _insert(self, root, val):
        if not root:
            return Node(val)
        if val < root.val:
            root.left = self._insert(root.left, val)
        elif val > root.val:
            root.right = self._insert(root.right, val)
        return root

    def add(self, val):
        self.root = self._insert(self.root, val)

    def _min_node(self, node):
        while node.left:
            node = node.left
        return node

    def _delete(self, root, val):
        if not root:
            return None
        if val < root.val:
            root.left = self._delete(root.left, val)
        elif val > root.val:
            root.right = self._delete(root.right, val)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            temp = self._min_node(root.right)
            root.val = temp.val
            root.right = self._delete(root.right, temp.val)
        return root

    def remove(self, val):
        self.root = self._delete(self.root, val)

```
### AVL Tree
```python
# ------------------------------------------------------------
# 🌲 3️⃣ AVLTree (Balanced Binary Search Tree)
# ------------------------------------------------------------
class AVLTree(BinarySearchTree):

    def height(self, node):
        return node.height if node else 0

    def get_balance(self, node):
        return self.height(node.left) - self.height(node.right) if node else 0

    # --- Rotations ---
    def right_rotate(self, y):
        x, T2 = y.left, y.left.right
        x.right, y.left = y, T2
        y.height = 1 + max(self.height(y.left), self.height(y.right))
        x.height = 1 + max(self.height(x.left), self.height(x.right))
        return x

    def left_rotate(self, x):
        y, T2 = x.right, x.right.left
        y.left, x.right = x, T2
        x.height = 1 + max(self.height(x.left), self.height(x.right))
        y.height = 1 + max(self.height(y.left), self.height(y.right))
        return y

    # --- AVL Insert ---
    def _insert(self, root, val):
        if not root:
            return Node(val)
        if val < root.val:
            root.left = self._insert(root.left, val)
        elif val > root.val:
            root.right = self._insert(root.right, val)
        else:
            return root

        # Update height
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        balance = self.get_balance(root)

        # Balance the node
        if balance > 1 and val < root.left.val:
            return self.right_rotate(root)
        if balance < -1 and val > root.right.val:
            return self.left_rotate(root)
        if balance > 1 and val > root.left.val:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance < -1 and val < root.right.val:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

```



