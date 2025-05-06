To **find the time complexity of Linear Search**, let's walk through it step-by-step, using both **conceptual reasoning** and **Big-O notation**.

---

### **What Is Linear Search?**

Linear Search is a method to find a specific value in a list/array by checking each element **one by one** from the beginning until the target is found or the list ends.

---

### **Step-by-Step Explanation**

#### **Step 1: Analyze the Algorithm**

Here's a basic implementation:

```java
int linearSearch(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
```

#### **Step 2: Count the Number of Operations**

You're checking each element once, so:

* In the **best case**, the element is at index `0` → **1 operation**.
* In the **worst case**, the element is at index `n-1` or not present → **n operations**.
* In the **average case**, the element is somewhere in the middle → **n/2 operations**.

#### **Step 3: Use Big-O Notation**

We simplify the number of operations and ignore constants:

* **Best case**: `O(1)`
* **Worst case**: `O(n)`
* **Average case**: `O(n)`

Thus, **time complexity = O(n)** (dominant case: worst/average).

---

### **Step 4: Space Complexity**

* Linear search doesn’t use extra space.
* **Space complexity = O(1)**

---

### **Conclusion:**

| Case    | Time Complexity |
| ------- | --------------- |
| Best    | O(1)            |
| Average | O(n)            |
| Worst   | O(n)            |
| Space   | O(1)            |

---

 Let's break down how the **time complexity of binary search** is determined as **O(log n)** — step by step.

---

### **What Binary Search Does**

Binary search works by **repeatedly dividing** the search interval in half.

It assumes the array is **sorted**.

---

### **Step-by-Step Process**

Let’s say we are searching in an array of size `n`:

#### Step 1:

* Check the **middle element**.
* If it matches the target → done (1 step).
* If not, eliminate **half** the array.

#### Step 2:

* Now search in the **remaining half** (size = n/2).

#### Step 3:

* Again check the middle → eliminate half → size = n/4

#### Step 4:

* Continue until the array size becomes 1.

---

### **Pattern Formed**

At each step, size of the array becomes:

* `n → n/2 → n/4 → n/8 → ... → 1`

This is a **geometric progression**, and the number of steps `k` needed is when:

```
n / 2^k = 1
```

Multiply both sides by 2^k:

```
n = 2^k
```

Take log base 2:

```
log₂(n) = k
```

So **k = log₂(n)** → that’s the number of steps (comparisons) needed.

Hence, the time complexity is:

### **T(n) = O(log n)**

---

### **What Does That Mean Practically?**

| Size of Array (`n`) | Max Comparisons (log₂n) |
| ------------------- | ----------------------- |
| 8                   | 3                       |
| 16                  | 4                       |
| 1024                | 10                      |
| 1,048,576           | 20                      |

Instead of scanning all elements like linear search, binary search only makes `log₂(n)` comparisons in the worst case.

---

Let's go step-by-step through the part where we analyze the **number of steps** in binary search using a **geometric sequence**, and why we arrive at **log₂(n)**.

---

### **Context Recap**

Binary search keeps halving the size of the search space:

* Start: `n` elements
* After 1 comparison: `n / 2`
* After 2 comparisons: `n / 4`
* After 3 comparisons: `n / 8`
* ...
* After `k` comparisons: `n / 2^k`

We want to find how many steps `k` it takes until only **1 element** is left:

```
n / 2^k = 1
```

---

### **Solve the Equation**

#### Step 1: Multiply both sides by `2^k`:

```
n = 2^k
```

#### Step 2: Take logarithm base 2 on both sides:

```
log₂(n) = log₂(2^k)
```

By log rule:

```
log₂(2^k) = k
```

So:

```
k = log₂(n)
```

---

### **What This Means**

The number of steps required to reduce `n` to 1 by halving it in each step is `log₂(n)`.

This isn't about **sum** of the geometric progression (which would be relevant if you were adding the elements). Instead, it's about how **many times** you can divide `n` by 2 before reaching 1.

This is why we say binary search has a time complexity of:

> **O(log n)** — base 2

---

Would you like to see how this compares to geometric **sums** for something like merge sort or exponential growth?
