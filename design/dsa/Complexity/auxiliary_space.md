### **Auxiliary Space in Data Structures & Algorithms (DSA)**

#### **Definition:**
**Auxiliary Space** refers to the extra space or temporary storage used by an algorithm beyond the input data. It is part of the **space complexity** of an algorithm but excludes the input size itself.

#### **Formula for Total Space Complexity:**
\[
\text{Total Space} = \text{Auxiliary Space} + \text{Input Space}
\]

---

## **1️⃣ Understanding Auxiliary Space with Examples**

### **🔹 Example 1: Recursive Fibonacci (High Auxiliary Space)**
Recursive algorithms often use **extra space** due to function call stack storage.

#### **Code:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(5))
```
#### **Analysis:**
- **Call Stack:** Recursive calls are stored in memory.
- **Auxiliary Space:** **O(n)** (due to recursion depth).
- **Total Space Complexity:** **O(n)**.

---

### **🔹 Example 2: Iterative Fibonacci (Low Auxiliary Space)**
An **iterative approach** reduces the need for extra space.

#### **Code:**
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(5))
```
#### **Analysis:**
- **No recursion, no call stack usage.**
- **Auxiliary Space:** **O(1)** (only a few variables).
- **Total Space Complexity:** **O(1)**.

---

### **2️⃣ Auxiliary Space in Sorting Algorithms**
#### **🔹 Example: QuickSort (Recursive)**
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
```
✅ **Auxiliary Space:** **O(n) (due to extra arrays)**  
✅ **Total Space Complexity:** **O(n log n) (including recursion stack)**  

#### **🔹 Example: MergeSort**
✅ **Auxiliary Space:** **O(n) (extra array for merging)**  
✅ **Total Space Complexity:** **O(n log n)**  

#### **🔹 Example: Insertion Sort (In-Place)**
✅ **Auxiliary Space:** **O(1)** (no extra storage used)  
✅ **Total Space Complexity:** **O(1)**  

---

## **3️⃣ Key Takeaways**
| Algorithm | Auxiliary Space |
|-----------|----------------|
| **Bubble Sort** | O(1) (In-place) |
| **Selection Sort** | O(1) (In-place) |
| **Insertion Sort** | O(1) (In-place) |
| **Merge Sort** | O(n) (Extra array) |
| **Quick Sort** | O(log n) (Recursion stack) |
| **DFS (Recursive)** | O(n) (Recursion stack) |
| **BFS (Queue-based)** | O(n) (Queue storage) |

**💡 Rule of Thumb:**  
✔ **Recursive algorithms** often use **O(n) auxiliary space** due to the call stack.  
✔ **In-place algorithms** like QuickSort and Bubble Sort use **O(1) auxiliary space**.  
✔ **Divide and Conquer algorithms** like MergeSort use **O(n) auxiliary space**.

Would you like an **in-depth comparison of space complexity for different algorithms**?