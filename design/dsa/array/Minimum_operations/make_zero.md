about the **“Reduce Array Elements to Zero”** type of problem (very common in CP and LeetCode).

---

# 📘 Problem: Reduce Array Elements to Zero

You are given an array `arr`.
In one operation, you can **choose any positive integer `x`** and subtract it from **all elements greater than 0**.
Find the **minimum number of operations** required to make all elements equal to `0`.

---

## 🔹 Key Insight

* Each operation subtracts the **same positive integer** from all positive elements.
* The only numbers that matter are the **distinct positive values** in the array.
* Because in each operation, at least one distinct value disappears.

👉 Therefore:

$$
\text{Minimum operations} = \text{Number of distinct positive elements}
$$

---

## 🔹 Example Walkthrough

### Example 1

```
arr = [1, 5, 0, 3, 5]
Distinct positive values = {1, 3, 5}
Answer = 3
```

**Steps:**

* Subtract 1 → arr becomes \[0, 4, 0, 2, 4]
* Subtract 2 → arr becomes \[0, 2, 0, 0, 2]
* Subtract 2 → arr becomes \[0, 0, 0, 0, 0]

3 operations ✅

---

### Example 2

```
arr = [0, 0, 0]
Distinct positive values = {}
Answer = 0
```

---

## 🔹 Python Solution

```python
def min_operations_reduce_to_zero(arr):
    # use a set to collect unique positive elements
    distinct_positive = {x for x in arr if x > 0}
    return len(distinct_positive)
```

---

## 🔹 Test

```python
print(min_operations_reduce_to_zero([1, 5, 0, 3, 5]))  # 3
print(min_operations_reduce_to_zero([0, 0, 0]))        # 0
print(min_operations_reduce_to_zero([4, 4, 4]))        # 1
print(min_operations_reduce_to_zero([2, 3, 6]))        # 3
```

---

✅ **Time Complexity** = $O(n)$
✅ **Space Complexity** = $O(n)$ (due to set)

---

👉 Do you also want me to extend this with a **step-by-step simulation function** (that prints the array after each operation), so you can visualize the process?
