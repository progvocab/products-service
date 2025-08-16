# **How to Find Time and Space Complexity of Sorting Algorithms**  

Sorting algorithms have different **time complexity** (how fast they run) and **space complexity** (how much extra memory they use). Understanding these complexities helps in choosing the best algorithm for a given problem.  

---

## **1. Time Complexity of Sorting Algorithms**  

### **Definition:**  
Time complexity measures how the **execution time** of an algorithm increases with **input size (n)**.  

### **Common Time Complexities in Sorting Algorithms:**  
| Complexity | Sorting Algorithms |
|------------|-------------------|
| **O(n²) (Quadratic)** | Bubble Sort, Selection Sort, Insertion Sort |
| **O(n log n) (Log-Linear)** | Merge Sort, Quick Sort, Heap Sort |
| **O(n) (Linear - Best Case Only)** | Counting Sort, Radix Sort (for some cases) |
| **O(1) (Constant - Best Case Only)** | Already sorted arrays in Bubble/Insertion Sort |

### **How to Determine Time Complexity:**  
1. **Count the number of comparisons and swaps** (or recursive calls for divide-and-conquer algorithms).  
2. **Find the worst, average, and best-case scenarios** based on input order.  
3. **Analyze loops and recursion depth** to derive the complexity.  

---

### **Example: Analyzing Time Complexity of Sorting Algorithms**

#### **(a) Bubble Sort – O(n²)**
```go
func bubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n-1; i++ {      // Outer loop runs n-1 times
        for j := 0; j < n-i-1; j++ { // Inner loop runs (n-i-1) times
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j] // Swap
            }
        }
    }
}
```
### **Time Complexity Breakdown:**  
- **Best Case (Already Sorted):** O(n)  
- **Worst/Average Case (Random Input):** O(n²)  
- The **nested loops** make it **O(n²)** because each element is compared with others.

---

#### **(b) Merge Sort – O(n log n)**
```go
func mergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    mid := len(arr) / 2
    left := mergeSort(arr[:mid])
    right := mergeSort(arr[mid:])
    return merge(left, right)
}

func merge(left, right []int) []int {
    result := []int{}
    i, j := 0, 0
    for i < len(left) && j < len(right) {
        if left[i] < right[j] {
            result = append(result, left[i])
            i++
        } else {
            result = append(result, right[j])
            j++
        }
    }
    result = append(result, left[i:]...)
    result = append(result, right[j:]...)
    return result
}
```
### **Time Complexity Breakdown:**  
- Merge Sort recursively divides the array **(log n levels deep)**.  
- At each level, merging takes **O(n) time**.  
- **Total Complexity:** O(n log n)  

---

## **2. Space Complexity of Sorting Algorithms**  

### **Definition:**  
Space complexity measures **extra memory used** by an algorithm **beyond input storage**.

### **Common Space Complexities:**
| Complexity | Sorting Algorithms |
|------------|-------------------|
| **O(1) (Constant)** | Bubble Sort, Selection Sort, Insertion Sort (In-place sorting) |
| **O(n) (Linear)** | Merge Sort (requires extra space for merging) |
| **O(n+k) (Linear + Extra Space)** | Counting Sort (depends on range of numbers) |

---

### **Example: Analyzing Space Complexity of Sorting Algorithms**

#### **(a) Quick Sort – O(log n)**
```go
func quickSort(arr []int) []int {
    if len(arr) < 2 {
        return arr
    }
    pivot := arr[len(arr)/2]
    left, right := []int{}, []int{}
    for _, num := range arr {
        if num < pivot {
            left = append(left, num)
        } else if num > pivot {
            right = append(right, num)
        }
    }
    return append(append(quickSort(left), pivot), quickSort(right)...)
}
```
### **Space Complexity Breakdown:**
- **Best/Average Case:** O(log n) (due to recursive calls)  
- **Worst Case (Already Sorted Input, No Random Pivoting):** O(n)  

---

#### **(b) Merge Sort – O(n) (Extra Space)**
- Since **Merge Sort** uses extra arrays for left and right halves, its space complexity is **O(n)**.

---

## **3. Summary of Time & Space Complexities**
| Algorithm | Best Time | Average Time | Worst Time | Space Complexity | In-Place? |
|-----------|----------|--------------|------------|------------------|-----------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ Yes |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | ✅ Yes |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ Yes |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | ❌ No |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | ✅ Yes |
| **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | ✅ Yes |
| **Counting Sort** | O(n+k) | O(n+k) | O(n+k) | O(n+k) | ❌ No |

---

## **Key Takeaways**
✅ **Time Complexity:** Depends on **number of comparisons/swaps** or **recursion depth**.  
✅ **Space Complexity:** Depends on **extra memory used** (in-place algorithms use O(1) space).  
✅ **O(n log n) sorts (Merge Sort, Quick Sort, Heap Sort) are more efficient than O(n²) sorts (Bubble Sort, Selection Sort).**  
✅ **Merge Sort requires extra space (O(n)), while Quick Sort can be in-place (O(log n)).**  

Would you like a **detailed breakdown for a specific sorting algorithm**?