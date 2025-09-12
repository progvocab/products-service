 **LeetCode 1769: Minimum Number of Operations to Move All Balls to Each Box**.

---

##  Problem Statement

You’re given a binary string `boxes` of length `n`, where:

* `boxes[i] == '1'` means there’s a ball in box `i`.
* `boxes[i] == '0'` means the box is empty.

In one operation, you can move **one ball** from one box to an adjacent box (`i → i+1` or `i → i-1`).

Return an integer array `answer` of length `n`, where `answer[i]` is the **minimum number of operations** needed to move **all balls into box `i`**.

---

## 🔑 Key Observations

* Naive way: For each `i`, compute sum of distances to all `j` with a ball. → `O(n^2)` (too slow).
* Optimized way:

  * We can solve in **two passes** (prefix + suffix).
  * Idea:

    * Keep track of how many balls we’ve seen (`count`) and total operations needed so far (`moves`).
    * Do one left-to-right pass and one right-to-left pass.
    * Combine results.

---

## 🛠️ Approach

1. Create two arrays `left` and `right` to track moves from left side and right side.
2. **Left pass:**

   * Start from left, track how many balls have been seen.
   * Update cumulative moves.
3. **Right pass:**

   * Same idea, but from right.
4. Final answer = `left[i] + right[i]`.



---
Solution

---

## ✅ Optimized Python Solution

```python
def minOperations(boxes: str):
    n = len(boxes)
    res = [0] * n 

    # Left to right pass
    count = 0  # number of balls so far
    moves = 0  # total moves
    for i in range(n):
        res[i] += moves
        if boxes[i] == '1':
            count += 1
        moves += count

    # Right to left pass
    count = 0
    moves = 0
    for i in range(n - 1, -1, -1):
        res[i] += moves
        if boxes[i] == '1':
            count += 1
        moves += count

    return res
```

---

## 🔎 Example Walkthrough

```
boxes = "110"
n = 3

Left-to-right:
res = [0,0,0]
i=0: res[0]=0, count=1, moves=1
i=1: res[1]=1, count=2, moves=3
i=2: res[2]=3, count=2, moves=5
→ res = [0,1,3]

Right-to-left:
i=2: res[2]+=0=3, count=0, moves=0
i=1: res[1]+=0=1, count=1, moves=1
i=0: res[0]+=1=1, count=2, moves=3
→ res = [1,1,3]
```

✅ Output = `[1,1,3]`

---

## ⏱️ Complexity

* **Time:** `O(n)`
* **Space:** `O(1)` extra (just result array)

---
 
