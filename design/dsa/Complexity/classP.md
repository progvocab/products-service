In computational complexity theory, **class P** and **class NP** are fundamental complexity classes used to measure the difficulty of solving computational problems.

---

### **1. Class P (Polynomial Time)**
- **Definition**: P is the class of problems that can be **solved** by a deterministic Turing machine in **polynomial time**.
- **Mathematically**: A problem is in **P** if there exists an algorithm that solves it in **O(n^k)** time for some constant **k**.
- **Intuition**: These are problems that can be solved efficiently.
- **Examples**:
  - Sorting numbers (**Merge Sort** → O(n log n))
  - Finding the shortest path in a graph (**Dijkstra’s Algorithm** → O(n^2))
  - Checking if a number is prime (**AKS primality test** → O(n^6))

---

### **2. Class NP (Nondeterministic Polynomial Time)**
- **Definition**: NP is the class of problems for which a given **solution can be verified** in **polynomial time** by a deterministic Turing machine.
- **Mathematically**: A problem is in **NP** if, given a possible solution, we can verify its correctness in **O(n^k)** time.
- **Intuition**: Finding the solution may be hard, but verifying a solution is easy.
- **Examples**:
  - **Traveling Salesman Problem (TSP)**: Given a path, checking if its total distance is ≤ some threshold is polynomial.
  - **Boolean Satisfiability (SAT)**: Given an assignment of variables, checking if it satisfies a Boolean formula is polynomial.
  - **Integer Factorization**: Given a number and its factors, verifying the factorization is easy.

---

### **3. Relationship Between P and NP**
- **Every problem in P is also in NP**, because if we can solve a problem in polynomial time, we can also verify its solution in polynomial time.
  - **P ⊆ NP**
- **The big question (P vs NP)**:  
  - If **P = NP**, then all problems that can be verified in polynomial time can also be solved in polynomial time (which means no hard problems!).
  - If **P ≠ NP**, there exist problems in NP that cannot be solved efficiently.

This is one of the biggest open problems in computer science, and solving it has a **$1 million prize** (Millennium Prize Problems).

---

### **4. NP-Complete Problems**
- **Definition**: NP-complete (NPC) problems are the hardest problems in NP.
- **Properties**:
  1. They are **in NP** (solutions can be verified in polynomial time).
  2. Every NP problem can be **reduced** to them in polynomial time.
- **Examples**:
  - **SAT (Boolean Satisfiability Problem)**
  - **3-SAT (Satisfiability with 3 literals per clause)**
  - **Traveling Salesman Problem (TSP)**
  - **Knapsack Problem**
  - **Graph Coloring**
  
If any NP-complete problem is solved in polynomial time, then **P = NP**.

---

### **5. NP-Hard Problems**
- **Definition**: These are problems at least as hard as NP problems but are not necessarily in NP.
- **Example**: The **Halting Problem** (undecidable, harder than NP problems).

---

### **6. Summary Table**
| Complexity Class | Definition | Example Problems |
|----------------|------------|-----------------|
| **P** | Problems solvable in polynomial time | Sorting, Shortest Path, Primality Testing |
| **NP** | Problems verifiable in polynomial time | SAT, TSP, Factorization |
| **NP-Complete** | Hardest problems in NP | 3-SAT, Knapsack, Graph Coloring |
| **NP-Hard** | At least as hard as NP problems | Halting Problem |

Would you like a real-world analogy to understand **P vs NP** better?