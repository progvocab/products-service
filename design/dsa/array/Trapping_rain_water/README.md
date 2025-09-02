 The **Trapping Rain Water** problem is a classic medium-level algorithm question (LeetCode #42).

---

## 🔹 Problem Statement

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

**Example:**

```
Input:  height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

---

## 🔹 Approach 1: Two Pointers (Optimal, O(n) time, O(1) space)

We keep two pointers `left` and `right`, and track the max height seen so far from each side (`left_max` and `right_max`).
At each step, we trap water based on the smaller max.

---

### ✅ Python Implementation

```python
def trap(height):
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water


# Example usage
heights = [0,1,0,2,1,0,1,3,2,1,2,1]
print("Trapped Water:", trap(heights))  # Output: 6
```

---

## 🔹 Approach 2: Prefix & Suffix Max Arrays (O(n) time, O(n) space)

```python
def trap_prefix_suffix(height):
    n = len(height)
    if n == 0:
        return 0

    left_max = [0]*n
    right_max = [0]*n
    water = 0

    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])

    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])

    for i in range(n):
        water += min(left_max[i], right_max[i]) - height[i]

    return water


print("Trapped Water (prefix/suffix):", trap_prefix_suffix(heights))  # Output: 6
```

---

## 🔹 Complexity

* **Two Pointers:**

  * Time = O(n)
  * Space = O(1) ✅ Best
* **Prefix & Suffix Arrays:**

  * Time = O(n)
  * Space = O(n)

---

👉 Do you want me to also show a **heap-based solution** (priority queue approach, useful when extended to 2D “Trapping Rain Water II”)?
