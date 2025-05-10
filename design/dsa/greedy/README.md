### **Greedy Algorithm: Theory Explained**

A **greedy algorithm** builds up a solution piece by piece, always choosing the next piece that offers the most immediate benefit or **greedy choice**, without worrying about the global consequences.

---

### **Key Concepts**

1. **Greedy Choice Property**:
   A global optimum can be arrived at by selecting a local optimum.

2. **Optimal Substructure**:
   An optimal solution to the problem contains optimal solutions to subproblems.

If both conditions are met, greedy algorithms can often provide an optimal solution efficiently.

---

### **When to Use Greedy Algorithms**

* Problems where local choices lead to global optimum.
* Faster alternatives to dynamic programming when optimal substructure is present and overlapping subproblems are not.

---

### **Time Complexity**

Typically ranges from **O(n log n)** (e.g., for sorting-based problems) to **O(n)** depending on implementation.

---

### **Examples of Greedy Algorithms**

| Problem                           | Greedy Strategy                                |
| --------------------------------- | ---------------------------------------------- |
| **Activity Selection**            | Choose next activity with earliest end time    |
| **Huffman Coding**                | Merge two least frequent nodes each time       |
| **Fractional Knapsack**           | Take item with highest value/weight ratio      |
| **Prim’s MST Algorithm**          | Choose edge with lowest weight to grow MST     |
| **Dijkstra’s Algorithm**          | Pick shortest unvisited vertex                 |
| **Job Sequencing with Deadlines** | Pick jobs with highest profit fitting schedule |

---

### **Example: Fractional Knapsack (Python)**

```python
def fractional_knapsack(weights, values, capacity):
    ratio = [(v / w, w, v) for w, v in zip(weights, values)]
    ratio.sort(reverse=True)

    total_value = 0
    for r, w, v in ratio:
        if capacity >= w:
            capacity -= w
            total_value += v
        else:
            total_value += r * capacity
            break
    return total_value
```

---

Would you like a visual or step-by-step walkthrough of a specific greedy problem?
