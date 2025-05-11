Greedy algorithms follow a consistent **pattern of steps**, regardless of the specific problem. Here's a breakdown of the **common steps in all greedy algorithms**:

---

### **1. Define the Objective Function**

* Determine what you want to **optimize**: e.g., minimum cost, maximum profit, fewest steps, etc.

---

### **2. Identify Greedy Choice Property**

* Find a **local optimal choice** that seems best at the current step, with the hope that this leads to a global optimum.

---

### **3. Sort or Prioritize Options (if needed)**

* Often, the options must be **sorted** by some criteria (e.g., value, weight, cost-benefit ratio) to make the greedy decision.

---

### **4. Iteratively Make the Greedy Choice**

* At each step:

  * Choose the **best available option**.
  * Make the choice **irrevocably** (i.e., don't revisit it).
  * **Update** the current state/problem constraints.

---

### **5. Repeat Until Goal is Reached or No Choices Remain**

---

### **6. Validate Solution**

* Not all problems have greedy solutions.
* Sometimes you **must prove** that the greedy strategy yields the optimal result (e.g., using exchange argument or greedy stays ahead method).

---

### **Optional Step: Compare with Brute Force or DP**

* Test your solution on edge cases.
* If greedy fails, the problem might require **dynamic programming or backtracking** instead.

---

### Example Use Cases:

| Problem Type            | Greedy Step                                  |
| ----------------------- | -------------------------------------------- |
| Activity Selection      | Pick next activity with earliest finish time |
| Huffman Encoding        | Merge two smallest frequencies               |
| Coin Change (canonical) | Pick largest coin ≤ remaining amount         |
| Fractional Knapsack     | Pick item with highest value/weight ratio    |

---

Would you like a greedy problem with visual explanation or decision proof?
