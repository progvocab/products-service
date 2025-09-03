Good one 👍 Rotating an array is a **very common interview question**. Let’s go step by step.

---

## 📌 Problem Statement

Rotate an array `nums` of length `n` by `k` steps (to the right).

* Each element shifts `k` places forward.
* Wrap around at the end.

---

## Example

```python
nums = [1,2,3,4,5,6,7], k = 3
```

👉 Output:

```
[5,6,7,1,2,3,4]
```

---

## 📖 Approaches

### 1. **Using Extra Array** (Simple, O(n) time, O(n) space)

```python
def rotate(nums, k):
    n = len(nums)
    k = k % n
    nums[:] = nums[-k:] + nums[:-k]

nums = [1,2,3,4,5,6,7]
rotate(nums, 3)
print(nums)  # [5,6,7,1,2,3,4]
```

---

### 2. **Reverse Method** (In-place, O(n) time, O(1) space) ✅

This is the most common trick.

Steps:

1. Reverse entire array.
2. Reverse first `k` elements.
3. Reverse remaining `n-k` elements.

```python
def rotate(nums, k):
    n = len(nums)
    k = k % n
    
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end = start+1, end-1
    
    reverse(0, n-1)
    reverse(0, k-1)
    reverse(k, n-1)

nums = [1,2,3,4,5,6,7]
rotate(nums, 3)
print(nums)  # [5,6,7,1,2,3,4]
```

---

### 3. **Cyclic Replacements** (O(n) time, O(1) space)

Move each number directly to its final position, cycle by cycle.

```python
def rotate(nums, k):
    n = len(nums)
    k = k % n
    count = 0  # number of elements moved
    start = 0

    while count < n:
        current = start
        prev = nums[start]
        while True:
            next_idx = (current + k) % n
            nums[next_idx], prev = prev, nums[next_idx]
            current = next_idx
            count += 1
            if start == current:
                break
        start += 1

nums = [1,2,3,4,5,6,7]
rotate(nums, 3)
print(nums)  # [5,6,7,1,2,3,4]
```

---

## ⏱ Complexity

* **Time:** O(n) for all approaches.
* **Space:** O(1) for reverse & cyclic, O(n) for extra array.

---

✅ **Answer:** Rotating an array means shifting elements by `k` positions with wrap-around. The most elegant way is the **reverse method** (in-place, O(1) extra space).

---

Do you want me to also cover **left rotation** (instead of right rotation) and its code?
