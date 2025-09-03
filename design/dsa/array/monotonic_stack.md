Perfect 👌 — let’s go deep into **Monotonic Stack**.

---

# 🔹 What is a Monotonic Stack?

A **monotonic stack** is a stack that keeps elements in a **monotonic order** (increasing or decreasing) as you push/pop.

* **Monotonic Increasing Stack**: elements from bottom → top are increasing.
* **Monotonic Decreasing Stack**: elements from bottom → top are decreasing.

This structure is useful for problems where you need to find **next greater element, next smaller element, previous greater/smaller element**, etc.

---

# 🔹 Classic Problems with Monotonic Stack

## 1. **Next Greater Element (NGE)**

👉 For each element, find the next element on the right that is greater.

### Example

```python
nums = [2, 1, 2, 4, 3]
Output = [4, 2, 4, -1, -1]
```

### Solution (Decreasing Stack)

```python
def nextGreaterElements(nums):
    n = len(nums)
    res = [-1] * n
    stack = []  # store indices

    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = nums[i]
        stack.append(i)
    return res

print(nextGreaterElements([2,1,2,4,3]))
```

---

## 2. **Daily Temperatures (LeetCode 739)**

👉 For each day, how many days until a warmer temperature?

### Example

```python
temps = [73,74,75,71,69,72,76,73]
Output = [1,1,4,2,1,1,0,0]
```

### Solution

```python
def dailyTemperatures(T):
    n = len(T)
    res = [0] * n
    stack = []  # store indices

    for i in range(n):
        while stack and T[i] > T[stack[-1]]:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res
```

---

## 3. **Largest Rectangle in Histogram (LeetCode 84)**

👉 Given bar heights, find largest rectangle area.

### Idea

* Use a **monotonic increasing stack** to keep indices of bars.
* Pop when current height < stack’s top height.

### Solution

```python
def largestRectangleArea(heights):
    heights.append(0)  # sentinel
    stack = [-1]
    max_area = 0

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
```

---

# 🔹 General Pattern

Monotonic Stack problems follow a **template**:

```python
stack = []
for i, num in enumerate(nums):
    while stack and condition(num, stack[-1]):
        # process
        stack.pop()
    stack.append(i)
```

* **Condition** depends on whether you want *next greater*, *next smaller*, etc.
* The stack ensures we don’t need nested loops → **O(n)** instead of **O(n²)**.

---

# 🔹 Summary

| Problem                        | Stack Type | Goal                               |
| ------------------------------ | ---------- | ---------------------------------- |
| Next Greater Element           | Decreasing | First bigger element to the right  |
| Next Smaller Element           | Increasing | First smaller element to the right |
| Daily Temperatures             | Decreasing | Days until warmer temperature      |
| Largest Rectangle in Histogram | Increasing | Max rectangle area in histogram    |

---

✅ **Monotonic stack** = efficient tool for "next greater/smaller" type problems, turning brute force O(n²) into O(n).

---

👉 Do you want me to also show how monotonic stack is applied in **2D problems** (like trapping rain water in 2D or maximal rectangle)?
