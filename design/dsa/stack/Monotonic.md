### ✅ What is a Monotonic Stack?

A **Monotonic Stack** is a special type of stack that maintains its elements in either **increasing** or **decreasing** order (monotonically), based on the use case.

---

### 🔧 Two Types

| Type                           | Maintains...            | Use Case                               |
| ------------------------------ | ----------------------- | -------------------------------------- |
| **Monotonic Increasing Stack** | Top is the **smallest** | Next greater element, span problems    |
| **Monotonic Decreasing Stack** | Top is the **largest**  | Next smaller element, stock span, etc. |

---

### 📦 Why Use It?

Monotonic stacks help **solve problems in linear time** (`O(n)`) that would otherwise take `O(n²)`, particularly in:

* **Next Greater/Smaller Element**
* **Histogram Area**
* **Stock Span Problem**
* **Rain Water Trapping**

---

### ✅ Example: Next Greater Element

```python
def next_greater_elements(nums):
    result = [-1] * len(nums)
    stack = []  # stores indices

    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result
```

#### 🔍 Example Input:

```python
nums = [2, 1, 5, 3]
```

#### 🔄 Output:

```python
[5, 5, -1, -1]
```

---

### 💡 How It Works

* The stack maintains a **monotonic decreasing** order of values from left to right.
* When a greater element is found, we resolve the top and pop it.
* We only go through the array once = `O(n)` time.

---

### 🧠 Summary

| Operation          | Monotonic Stack Helps With           |
| ------------------ | ------------------------------------ |
| Find next greater  | Use **increasing** stack             |
| Find next smaller  | Use **decreasing** stack             |
| Sliding window max | Use deque, sometimes monotonic stack |
| Largest rectangle  | Use increasing stack                 |

---

Let me know if you’d like a Java or C++ version, or examples for more problems like:

* Largest Rectangle in Histogram
* Daily Temperatures
* Stock Span Problem



Here are some **Python examples using a monotonic stack** for different problems, showing how powerful and efficient this technique is:

---

## ✅ 1. Next Greater Element (Monotonic Increasing Stack)

```python
def next_greater_elements(nums):
    result = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result

# Example
print(next_greater_elements([2, 1, 5, 3]))  # Output: [5, 5, -1, -1]
```

---

## ✅ 2. Daily Temperatures (Leetcode-style)

```python
def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []  # stores indices

    for i, temp in enumerate(temperatures):
        while stack and temp > temperatures[stack[-1]]:
            j = stack.pop()
            result[j] = i - j
        stack.append(i)

    return result

# Example
print(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))
# Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

---

## ✅ 3. Largest Rectangle in Histogram

```python
def largest_rectangle_area(heights):
    stack = []  # (index, height)
    max_area = 0

    for i, h in enumerate(heights + [0]):
        while stack and h < stack[-1][1]:
            _, height = stack.pop()
            width = i if not stack else i - stack[-1][0] - 1
            max_area = max(max_area, height * width)
        stack.append((i, h))

    return max_area

# Example
print(largest_rectangle_area([2, 1, 5, 6, 2, 3]))  # Output: 10
```

---

## 🧠 Recap: When to Use Monotonic Stack?

| Problem Type                  | Stack Order Used                |
| ----------------------------- | ------------------------------- |
| Next Greater Element          | Increasing                      |
| Next Smaller Element          | Decreasing                      |
| Histogram / Largest Rectangle | Increasing                      |
| Daily Temperatures            | Increasing                      |
| Trapping Rain Water           | Increasing/Decreasing as needed |

---

Would you like to explore:

* Circular arrays (e.g., Next Greater Element II)?
* Monotonic stack in **2D problems**?
* Convert one to **Java/C++**?

Let me know!


Great! We’ll walk through:

---

## ✅ 1. **Circular Arrays (Next Greater Element II)**

In circular arrays, you iterate twice:

```python
def next_greater_elements_circular(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(2 * n):
        idx = i % n
        while stack and nums[idx] > nums[stack[-1]]:
            result[stack.pop()] = nums[idx]
        if i < n:
            stack.append(idx)
    return result

# Example
print(next_greater_elements_circular([1, 2, 1]))  # Output: [2, -1, 2]
```

---

## ✅ 2. **2D Problem: Maximum Rectangle in Binary Matrix**

Uses largest rectangle in histogram logic for each row.

```python
def maximal_rectangle(matrix):
    if not matrix: return 0
    cols = len(matrix[0])
    height = [0] * cols
    max_area = 0

    for row in matrix:
        for i in range(cols):
            height[i] = height[i] + 1 if row[i] == '1' else 0
        max_area = max(max_area, largest_rectangle_area(height))

    return max_area

# Reuse the function from earlier:
def largest_rectangle_area(heights):
    stack = []
    max_area = 0

    for i, h in enumerate(heights + [0]):
        while stack and h < stack[-1][1]:
            _, height = stack.pop()
            width = i if not stack else i - stack[-1][0] - 1
            max_area = max(max_area, height * width)
        stack.append((i, h))

    return max_area

# Example
matrix = [
    ["1", "0", "1", "0", "0"],
    ["1", "0", "1", "1", "1"],
    ["1", "1", "1", "1", "1"],
    ["1", "0", "0", "1", "0"]
]
print(maximal_rectangle(matrix))  # Output: 6
```

---

## 🟧 Java Version: Next Greater Element

```java
public int[] nextGreaterElements(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    Deque<Integer> stack = new ArrayDeque<>();

    Arrays.fill(result, -1);

    for (int i = 0; i < 2 * n; i++) {
        int idx = i % n;
        while (!stack.isEmpty() && nums[idx] > nums[stack.peek()]) {
            result[stack.pop()] = nums[idx];
        }
        if (i < n) stack.push(idx);
    }

    return result;
}
```

---

## 🟦 C++ Version: Largest Rectangle in Histogram

```cpp
int largestRectangleArea(vector<int>& heights) {
    heights.push_back(0);
    stack<int> st;
    int maxArea = 0;

    for (int i = 0; i < heights.size(); ++i) {
        while (!st.empty() && heights[i] < heights[st.top()]) {
            int h = heights[st.top()];
            st.pop();
            int w = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, h * w);
        }
        st.push(i);
    }
    return maxArea;
}
```

---

Let me know if you want:

* Monotonic **deque** (for sliding window problems)?
* Convert to **Rust**, **Go**, or **TypeScript**?
* Performance tips or visual explanation?

I'm happy to go deeper!
