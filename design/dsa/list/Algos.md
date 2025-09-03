### **Add Two Numbers problem** (LeetCode #2).

This is a **linked list problem** where each linked list stores a number in **reverse order** (least significant digit first).

---

## 📌 Problem Statement

You are given two non-empty linked lists representing two non-negative integers.

* Digits are stored in **reverse order**.
* Each node contains a single digit.
* Add the two numbers and return the sum as a linked list.

⚡ Example:

```
Input:  (2 → 4 → 3) + (5 → 6 → 4)
Output: 7 → 0 → 8
```

Explanation: `342 + 465 = 807`

---

## 📖 Approach

1. Use two pointers (`l1`, `l2`) to traverse both lists.
2. Maintain a `carry` variable.
3. Add corresponding digits + carry.
4. Create new nodes for the result list.
5. Continue until both lists and carry are empty.

---

## ⚡ Python Code

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1, l2):
    dummy = ListNode()  # Placeholder for result
    curr = dummy
    carry = 0

    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0

        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10

        curr.next = ListNode(digit)
        curr = curr.next

        # Move to next nodes
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return dummy.next
```

---

## ✅ Example Run

```python
# Create first number: 342 as (2→4→3)
l1 = ListNode(2, ListNode(4, ListNode(3)))

# Create second number: 465 as (5→6→4)
l2 = ListNode(5, ListNode(6, ListNode(4)))

# Add them
res = addTwoNumbers(l1, l2)

# Print result
while res:
    print(res.val, end=" ")
    res = res.next
```

👉 Output:

```
7 0 8
```

---

## ⏱ Complexity

* **Time:** `O(max(m, n))` → where m and n are lengths of the lists
* **Space:** `O(max(m, n))` → for the result list

---

✅ **Answer:** The program adds two numbers represented by linked lists (digits stored in reverse order), returning their sum as a new linked list.

---

Do you also want me to show the **variation where digits are stored in forward order** (most significant digit first)?
