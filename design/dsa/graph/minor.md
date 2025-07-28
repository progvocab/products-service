A **minor** of a graph is another graph that can be formed from it by using **any or all** of the following operations:

### ✅ Operations to form a minor:

1. **Delete a vertex**
2. **Delete an edge**
3. **Contract an edge** (merge its two endpoints into a single vertex)

---

## 🔍 Example

Let's say we have a graph:

```
A -- B
|    |
C -- D
```

This is a square with vertices A, B, C, D.

### Step-by-step Minor:

1. **Edge contraction**: Contract edge A-B → merge A and B into one node AB
   Result:

   ```
   AB
   | \
   C  D
   ```

2. **Delete edge AB-D** → optional, still a minor.

This simplified graph is a **minor** of the original.

---

## 🧠 Why is this useful?

Graph minors are crucial in:

| Use Case                   | Example                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| **Planarity testing**      | Wagner’s theorem: planar graphs do **not** contain **K₅ or K₃,₃** as **minors**                    |
| **Graph structure theory** | Robertson–Seymour theorem: Every infinite graph family has a finite set of forbidden minors        |
| **Algorithm design**       | Some problems become easier if you know certain minors don’t exist (e.g., treewidth-based methods) |

---

## 🔁 Minor vs Subgraph vs Subdivision

| Term            | Operation Allowed                                | Key Difference          |
| --------------- | ------------------------------------------------ | ----------------------- |
| **Minor**       | Delete vertices/edges + contract edges           | More general            |
| **Subgraph**    | Only delete vertices/edges                       | Simpler                 |
| **Subdivision** | Replace edges with paths (insert degree-2 nodes) | Opposite of contraction |

---

## 📌 Real-World Analogy:

Think of a minor as a **compressed or simplified version** of a graph — like reducing a circuit diagram by merging equivalent wires.

---

Would you like code to compute graph minors or check for planarity using Python (e.g., NetworkX)?
