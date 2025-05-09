Here’s how to structure a **backtracking solution to find all subsets** of an integer array with clean modular functions like `is_goal`, `record_solution`, `is_valid`, `make_move`, `undo_move`, and `backtrack`.

---

### **Problem Statement**

Given an integer array `nums`, generate all possible subsets (the power set).

---

### **Structured Backtracking Approach**

```python
def subsets(nums):
    result = []
    path = []

    def is_goal(start):
        # We consider every path as a valid subset
        return True

    def record_solution(path):
        # Record a copy of the current path (subset)
        result.append(path[:])

    def is_valid(next_elem):
        # All elements are valid in subset; no constraint
        return True

    def make_move(path, num):
        path.append(num)

    def undo_move(path):
        path.pop()

    def backtrack(start):
        if is_goal(start):
            record_solution(path)

        for i in range(start, len(nums)):
            if is_valid(nums[i]):
                make_move(path, nums[i])
                backtrack(i + 1)
                undo_move(path)

    backtrack(0)
    return result
```

---

### **Example**

```python
nums = [1, 2, 3]
print(subsets(nums))
```

**Output:**

```
[[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

---

### **Key Points**

* `is_goal` always returns True since every subset is valid.
* `make_move` and `undo_move` help backtracking track and revert state cleanly.
* The function is easily extensible (e.g., for constraints on subset size or sum).

Would you like to see how this structure could be adapted for combinations or permutations?
