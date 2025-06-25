The **two-pointer algorithm** (including variants like slow-fast pointers) is not attributed to a single individual — it's a **technique** that evolved over time as part of algorithmic problem solving in **computer science** and **competitive programming**.

---

## 🧠 Origins & Evolution

* The **idea of using two indices/pointers** (one from the start, one from the end) predates modern programming contests.
* It likely evolved **organically** from:

  * Early **search and sort algorithms** (like binary search, merge sort).
  * **Linked list traversal** (e.g., finding the middle, detecting cycles).
  * **Sliding window** optimizations (e.g., substring problems).

---

### 📚 Documented Use

| Technique           | Early Known Use                                                                   |
| ------------------- | --------------------------------------------------------------------------------- |
| Two-pointer (array) | Described in early algorithm textbooks and ACM papers from the 1960s–1980s        |
| Slow-fast pointers  | Credited to **Floyd’s Cycle Detection Algorithm** (1967)                          |
| Sliding window      | Used in streaming data problems, introduced in parsing and compression algorithms |

---

### 🏆 Floyd’s Cycle Detection (1967)

* **Robert W. Floyd** (of *Floyd-Warshall* fame) introduced the **slow-fast pointer method** for cycle detection.
* Also known as the **"tortoise and hare" algorithm**.

---

## 💡 In Practice

Today, two-pointer techniques are common in:

* **Competitive programming** (Codeforces, Leetcode, etc.)
* **Interview problems**
* **Textbooks** (like CLRS, "The Algorithm Design Manual")

---

## ✅ Conclusion

* No single person "invented" the **general two-pointer technique**.
* The **slow-fast pointer** (for cycles) is credited to **Robert Floyd** (1967).
* The broader method evolved over time and is now a **standard tool** in the algorithm toolbox.

---

Let me know if you’d like:

* A history timeline of key algorithm innovations
* A visual explanation of how Floyd’s cycle detection works
* Common patterns where two-pointer shines (sorted arrays, linked lists, etc.)



Absolutely! Here's a Python implementation to **detect a cycle** in a **singly linked list** using the **slow-fast (tortoise and hare)** pointer technique (also known as **Floyd's Cycle Detection Algorithm**).

---

## ✅ Python Code: Detect Cycle in Linked List

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # move one step
        fast = fast.next.next     # move two steps
        if slow == fast:
            return True           # cycle detected
    return False                  # no cycle
```

---

## 🔁 Example Usage

```python
# Create a linked list with a cycle
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)

a.next = b
b.next = c
c.next = d
d.next = b  # cycle here: d → b

print("Cycle detected:", has_cycle(a))  # Output: True
```

### 🧪 No Cycle Case

```python
# Create a linked list with no cycle
x = Node(10)
y = Node(20)
z = Node(30)

x.next = y
y.next = z

print("Cycle detected:", has_cycle(x))  # Output: False
```

---

## 🧠 How It Works

| Pointer                          | Moves   | Purpose                          |
| -------------------------------- | ------- | -------------------------------- |
| `slow`                           | 1 step  | normal traversal                 |
| `fast`                           | 2 steps | faster — catches `slow` in cycle |
| If they meet → cycle exists      |         |                                  |
| If `fast` hits `None` → no cycle |         |                                  |

---

Let me know if you also want:

* To **find the starting node** of the cycle
* To **remove the cycle**
* Or convert this into a class-based `LinkedList` implementation.
