The **Union-Find** data structure (also known as **Disjoint Set Union – DSU**) is a powerful structure used to manage a collection of disjoint sets and perform two key operations efficiently:

### ✅ Key Operations

1. **Find**: Determine which set a particular element belongs to (i.e., find the representative/parent of the set).
2. **Union**: Merge two sets into a single set.

This is widely used in problems involving:

* **Connected components in a graph**
* **Kruskal's algorithm for Minimum Spanning Tree**
* **Cycle detection in undirected graphs**
* **Network connectivity**

---

### 🔧 Python Implementation with Path Compression and Union by Rank

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))  # Initially, each node is its own parent
        self.rank = [0] * size           # Rank (or depth) for union by rank optimization

    def find(self, x):
        if self.parent[x] != x:
            # Path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in the same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True  # Union performed
```

---

### 📌 Example Usage

```python
uf = UnionFind(5)

uf.union(0, 1)
uf.union(1, 2)

print(uf.find(0))  # Output: representative of set containing 0
print(uf.find(2))  # Should be the same as find(0)
print(uf.find(3))  # Different set
```

---

### 🔍 Visualization

Initial:

```
0 1 2 3 4
| | | | |
0 1 2 3 4
```

After `union(0, 1)` and `union(1, 2)`:

```
0 - 1 - 2   3   4
All point to 0 (or compressed to 0)
```

---

### 🧠 Optimizations

* **Path Compression**: Makes future `find()` operations faster by flattening the structure.
* **Union by Rank/Size**: Ensures the tree remains shallow by attaching the smaller tree under the root of the larger one.

---

### 📚 Applications

* Kruskal’s MST Algorithm
* Dynamic connectivity
* Image segmentation
* Percolation theory
* Detecting cycles in undirected graphs

Would you like an example problem solved using Union-Find (like Kruskal’s MST or cycle detection)?
