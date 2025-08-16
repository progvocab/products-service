Disjoint set (Union–Find) detects cycles by checking whether an **edge connects two vertices that are already in the same set**.

If they’re already in the same set, it means there’s **already a path between them**, so adding this edge would create a **cycle**.

---

## **Step-by-step Idea**

1. Start with all vertices in their own sets.
2. For each edge `(u, v)`:

   * **Find** the root of `u` and `v`.
   * If they have the **same root**, a cycle exists.
   * Otherwise, **Union** them into the same set.
3. Continue until either a cycle is found or all edges are processed.

---

## **Example**

Graph:

```
1 -- 2
|    |
4 -- 3
```

Edges:

```
(1, 2), (2, 3), (3, 4), (4, 1)
```

**Process:**

* **(1, 2)**: Different sets → union → `{1, 2}`
* **(2, 3)**: Different sets → union → `{1, 2, 3}`
* **(3, 4)**: Different sets → union → `{1, 2, 3, 4}`
* **(4, 1)**: Same set → **Cycle detected**

---

## **Python Code**

```python
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n+1))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Cycle detected
        self.parent[py] = px
        return True

def has_cycle(n, edges):
    ds = DisjointSet(n)
    for u, v in edges:
        if not ds.union(u, v):
            return True
    return False

# Test
edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
print(has_cycle(4, edges))  # True
```

---

## **Why it works**

* Union–Find tracks **connected components**.
* If two vertices are already in the same component, adding another edge between them **closes a loop**.

---

If you want, I can also show **how this works differently for directed vs undirected graphs**, because Union–Find works directly for **undirected** but needs adjustments for **directed**.

Do you want me to walk you through that difference?
