

The **Wait-For Graph (WFG)** is a concept from **Operating Systems** and **Distributed Systems**, mainly used in **deadlock detection**.

---

## 📖 Definition

A **Wait-For Graph** is a **directed graph** used to represent which **process is waiting for which other process** in a system.

* **Vertices (nodes)** → Processes (P1, P2, …, Pn)
* **Edge (Pi → Pj)** → Process `Pi` is waiting for process `Pj` to release a resource.

---

## 🔑 Key Idea

* If a **cycle** exists in the Wait-For Graph → **deadlock** occurs.
* If the graph is **acyclic** → no deadlock.

---

## 📌 Example

### Scenario:

* P1 holds R1, waiting for R2
* P2 holds R2, waiting for R3
* P3 holds R3, waiting for R1

### WFG Representation:

```
P1 → P2 → P3 → P1
```

This cycle means **deadlock** exists.

---

## 🐍 Python Example (Deadlock Detection)

```python
from collections import defaultdict

class WaitForGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        """Process u is waiting for Process v"""
        self.graph[u].append(v)

    def has_cycle(self):
        visited = {}
        
        def dfs(u):
            visited[u] = 1  # visiting
            for v in self.graph[u]:
                if v not in visited:
                    if dfs(v):
                        return True
                elif visited[v] == 1:
                    return True
            visited[u] = 2  # done
            return False

        for node in self.graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False


# 🔹 Example: P1 → P2 → P3 → P1
wfg = WaitForGraph()
wfg.add_edge("P1", "P2")
wfg.add_edge("P2", "P3")
wfg.add_edge("P3", "P1")

print("Deadlock detected?" , wfg.has_cycle())  # ✅ True
```

---

## ✅ Output

```
Deadlock detected? True
```

---

## 🔎 Summary

* **Wait-For Graph** → directed graph of process dependencies.
* **Cycle in WFG** → **deadlock**.
* **No cycle** → system is safe.

---

👉 Do you want me to also show how a **Wait-For Graph** is constructed from a **Resource Allocation Graph (RAG)** (which includes both processes and resources)?
