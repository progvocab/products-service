# **Kadane’s Algorithm**  

Kadane’s Algorithm is an **efficient approach** to find the **maximum subarray sum** in an array. It runs in **O(n) time complexity**, making it one of the best solutions for this problem.

---

## **1. Problem Statement**  
Given an array **nums[]** of size **n**, find the **contiguous subarray** (containing at least one number) which has the **largest sum**, and return that sum.  

### **Example 1**
#### **Input:**  
```plaintext
nums = [-2,1,-3,4,-1,2,1,-5,4]
```
#### **Output:**  
```plaintext
6
```
#### **Explanation:**  
The subarray `[4, -1, 2, 1]` has the maximum sum `4 + (-1) + 2 + 1 = 6`.

---

## **2. Kadane's Algorithm Approach**  
1. Initialize:  
   - `maxSoFar = nums[0]` → Stores the **maximum sum found so far**.  
   - `maxEndingHere = nums[0]` → Tracks the **sum of the current subarray**.  
2. Iterate from index **1 to n-1**:
   - Update `maxEndingHere = max(nums[i], maxEndingHere + nums[i])`  
   - Update `maxSoFar = max(maxSoFar, maxEndingHere)`  
3. Return `maxSoFar` as the **maximum subarray sum**.  

---

## **3. Kadane's Algorithm in Go**
```go
package main

import (
	"fmt"
	"math"
)

// Function implementing Kadane's Algorithm
func maxSubArray(nums []int) int {
	maxSoFar := math.MinInt32
	maxEndingHere := 0

	for _, num := range nums {
		maxEndingHere += num
		if maxEndingHere > maxSoFar {
			maxSoFar = maxEndingHere
		}
		if maxEndingHere < 0 {
			maxEndingHere = 0
		}
	}
	return maxSoFar
}

func main() {
	arr := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
	fmt.Println("Maximum Subarray Sum:", maxSubArray(arr)) // Output: 6
}
```

---

## **4. Understanding the Algorithm with an Example**
Let's take `nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]` and apply Kadane’s Algorithm step by step.

| Index | nums[i] | maxEndingHere (`max(nums[i], maxEndingHere + nums[i])`) | maxSoFar (`max(maxSoFar, maxEndingHere)`) |
|--------|--------|----------------------|----------------------|
| 0 | -2 | -2 | -2 |
| 1 | 1 | **1** | **1** |
| 2 | -3 | -2 | 1 |
| 3 | 4 | **4** | **4** |
| 4 | -1 | 3 | 4 |
| 5 | 2 | **5** | **5** |
| 6 | 1 | **6** | **6** |
| 7 | -5 | 1 | 6 |
| 8 | 4 | **5** | **6** |

✅ **Final Answer:** **6**

---

## **5. Kadane’s Algorithm with Subarray Tracking**
If we also want to find the **starting and ending indices** of the maximum subarray:
```go
package main

import (
	"fmt"
	"math"
)

// Function to find max subarray sum and its indices
func maxSubArrayWithIndices(nums []int) (int, int, int) {
	maxSoFar := math.MinInt32
	maxEndingHere := 0
	start, end, tempStart := 0, 0, 0

	for i, num := range nums {
		maxEndingHere += num

		if maxEndingHere > maxSoFar {
			maxSoFar = maxEndingHere
			start = tempStart
			end = i
		}
		if maxEndingHere < 0 {
			maxEndingHere = 0
			tempStart = i + 1
		}
	}
	return maxSoFar, start, end
}

func main() {
	arr := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
	maxSum, start, end := maxSubArrayWithIndices(arr)
	fmt.Println("Maximum Subarray Sum:", maxSum) // Output: 6
	fmt.Println("Subarray:", arr[start:end+1])  // Output: [4 -1 2 1]
}
```
✅ **This approach also returns the indices of the maximum sum subarray.**

---

## **6. Time & Space Complexity**
- **Time Complexity:** `O(n)` (Single pass through array)  
- **Space Complexity:** `O(1)` (No extra space used)

---

## **7. When Does Kadane’s Algorithm Fail?**
- **If the problem asks for a non-contiguous subarray sum**, Kadane's will not work.
- If the **array contains only negative numbers**, Kadane’s needs modification.

#### **Handling All Negative Numbers**
To handle cases where all numbers are negative:
```go
func maxSubArrayAllNegatives(nums []int) int {
	maxSoFar := nums[0] // Instead of MinInt32
	maxEndingHere := nums[0]

	for i := 1; i < len(nums); i++ {
		maxEndingHere = max(nums[i], maxEndingHere+nums[i])
		maxSoFar = max(maxSoFar, maxEndingHere)
	}
	return maxSoFar
}

// Helper function for max
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	arr := []int{-3, -1, -4, -2}
	fmt.Println("Maximum Subarray Sum:", maxSubArrayAllNegatives(arr)) // Output: -1
}
```

✅ **Handles cases where all numbers are negative.**

---

## **8. Variations of Kadane’s Algorithm**
1. **2D Kadane’s Algorithm** → Finds the **maximum submatrix sum** in a 2D array.
2. **Circular Subarray Maximum Sum** → Used when the subarray can wrap around the end of the array.
3. **K-Concatenation Maximum Sum** → Used when an array is repeated `k` times.

---

## **Key Takeaways**
✅ **Kadane’s Algorithm finds the maximum sum of a contiguous subarray in O(n).**  
✅ **Uses dynamic programming by maintaining `maxSoFar` and `maxEndingHere`.**  
✅ **Can be extended to track subarray indices.**  
✅ **Needs modification for all-negative arrays.**  

Would you like a **2D Kadane's Algorithm explanation**?