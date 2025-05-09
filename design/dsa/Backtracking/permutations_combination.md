Great! Let's adapt the same **modular backtracking structure** to generate:

---

### **1. Combinations: Choose `k` numbers from `n`**

#### **Problem**

Given `nums = [1, 2, 3, 4]` and `k = 2`, generate all combinations of 2 numbers.

#### **Code**

```python
def combinations(nums, k):
    result = []
    path = []

    def is_goal(start):
        return len(path) == k

    def record_solution(path):
        result.append(path[:])

    def is_valid(next_elem):
        return True

    def make_move(path, num):
        path.append(num)

    def undo_move(path):
        path.pop()

    def backtrack(start):
        if is_goal(start):
            record_solution(path)
            return

        for i in range(start, len(nums)):
            if is_valid(nums[i]):
                make_move(path, nums[i])
                backtrack(i + 1)
                undo_move(path)

    backtrack(0)
    return result
```

---

### **2. Permutations: All orderings of `n` numbers**

#### **Problem**

Given `nums = [1, 2, 3]`, generate all permutations.

#### **Code**

```python
def permutations(nums):
    result = []
    path = []
    used = [False] * len(nums)

    def is_goal(start):
        return len(path) == len(nums)

    def record_solution(path):
        result.append(path[:])

    def is_valid(i):
        return not used[i]

    def make_move(path, i):
        path.append(nums[i])
        used[i] = True

    def undo_move(path, i):
        path.pop()
        used[i] = False

    def backtrack(start):
        if is_goal(start):
            record_solution(path)
            return

        for i in range(len(nums)):
            if is_valid(i):
                make_move(path, i)
                backtrack(i + 1)
                undo_move(path, i)

    backtrack(0)
    return result
```

---

### Would You Like:

* To add constraints (like sum or uniqueness)?
* To visualize the recursion tree for one of them?
