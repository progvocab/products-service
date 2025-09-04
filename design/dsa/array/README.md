array problems are some of the **most common algorithm interview questions** because arrays are fundamental and allow testing **time complexity, data structures, and problem-solving techniques**.

Here’s a **curated list of the most common array algorithm questions**, grouped by concept 

---
# 🔑 Common Array Algorithms with Python Examples

---

## 1. **Two Pointers**

Example: Move Zeroes ( in place )

```python
def moveZeroes(nums):
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
    return nums

print(moveZeroes([0,1,0,3,12]))  # [1,3,12,0,0]
```

---

## 2. **Sliding Window**

Example: Maximum sum subarray of size k

```python
def maxSubarraySum(nums, k):
    window_sum = sum(nums[:k])
# find the sum of all elements in the array
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i-k]
# find the window sum by adding current element and removing element before the window
        max_sum = max(max_sum, window_sum)
# check if the current window sum is greater than earlier window sums
    return max_sum

print(maxSubarraySum([1,2,3,4,5,6], 3))  # 15
```

---

## 3. **Prefix Sum**

Example: Count of Subarray Sum which Equals K

```python
def subarraySum(nums, k):
    prefix_sum = 0
    count = 0
    seen = {0: 1}
# # prefix sum count hash table
    for num in nums:
        prefix_sum += num 
        count += seen.get(prefix_sum - k, 0)
# check if there is any prefix sum which when excluded from total sum till now will match the target
# if yes count how many are there
        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1
#add new prefix sum entry or update the count of existing prefix sum entry in hash table 
    return count

print(subarraySum([1,1,1], 2))  # 2
# input  array  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# target 5
# prefix sum hash table : {0: 1, 1: 1, 3: 1, 6: 1, 10: 1, 15: 1, 21: 1, 28: 1, 36: 1, 45: 1, 55: 1}
# number of Subarray Sum which Equals target is 2 - 2,3 and 1,4
```

---

## 4. **Kadane’s Algorithm**

Example: Maximum Subarray

```python
def maxSubArray(nums):
    curr_sum = max_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
```
### Maximum Product Subarray is a classic array DP problem

```python
def maxProduct(nums):
    if not nums:
        return 0
    
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        # if num is negative, swap max and min
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, num * max_prod)
        min_prod = min(num, num * min_prod)
        
        result = max(result, max_prod)
    
    return result


# Example usage
print(maxProduct([2,3,-2,4]))      # 6 (subarray [2,3])
print(maxProduct([-2,0,-1]))       # 0 (subarray [0])
print(maxProduct([-2,3,-4]))       # 24 (subarray [-2,3,-4])
```

---

## 5. **Sorting and  Two Pointers**

### Example: Sum of Triplets
- Given an integer array nums, return all unique triplets [a, b, c] such that: ```a+b+c = 0```
- Triplets should not repeat.
- Order of elements inside a triplet doesn’t matter.

```python
def sumOfTriplets(nums):
    nums.sort() # Sort the array 
    res = []
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i+1, len(nums)-1 # the two pointers
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0: # check if sum is equal to zero or target K
                res.append([nums[i], nums[l], nums[r]])
                l += 1; r -= 1
                # Skip duplicates for left & right 
                while l < r and nums[l] == nums[l-1]: l += 1
                while l < r and nums[r] == nums[r+1]: r -= 1
            elif s < 0: # move right if sum is less than target
                l += 1
            else:  # move left
                r -= 1
    return res

print(sumOfTriplets([-1,0,1,2,-1,-4]))  
# [[-1,-1,2], [-1,0,1]]
```

---

## 6. **Binary Search**

- Example : rotate an array

```python
# arr [1,2,3,4,5] k : 3
# 1.   reverse all         [5,4,3,2,1]
# 2.   reverse k           [3,4,5,2,1]
# 3.   reverse remaining   [3,4,5,1,2]
def rotatearray(arr,k) :
    
    n=len(arr)
    k=k % n
    reverse(arr,0, n-1)
    reverse(arr,0,k-1 )
    reverse(arr,k,n-1)

def reverse(arr,start,end):
    while start < end :
        arr[start] , arr[end] = arr[end] , arr[start]
        start +=1
        end -=1
```  

- Example: Search in Rotated Sorted Array

```python
def search(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        mid = (l+r)//2
        if nums[mid] == target:
            return mid
        if nums[l] <= nums[mid]:  # left is sorted
            if nums[l] <= target < nums[mid]:
                r = mid-1
            else:
                l = mid+1
        else:  # right sorted
            if nums[mid] < target <= nums[r]:
                l = mid+1
            else:
                r = mid-1
    return -1

print(search([4,5,6,7,0,1,2], 0))  # 4
```

---

## 7. **Hashing**

Example:  Sum of two numbers matching the target

```python
def sumOfTwo(nums, target):
    seen = {} #hash table 
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target-num], i]
        seen[num] = i
    return []

print(sumOfTwo([2,7,11,15], 9))  # [0,1]
```

---

## 8. **Greedy**

Example: Jump Game - can be solved by greedy algorithm as well as dynamic programming 
- You are given an array of non-negative integers nums, where each element represents your maximum jump length from that position.
Your task is to determine if you can reach the last index starting from index 0

```python
def canJump(nums):
    max_reach = 0
    for i, num in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i+num)
    return True

print(canJump([2,3,1,1,4]))  # True
```

---

## 9. **Divide & Conquer**

Example: Maximum Subarray (recursive)

```python
def maxCrossingSum(nums, l, m, r):
    left_sum = right_sum = float('-inf')
    s = 0
    for i in range(m, l-1, -1):
        s += nums[i]
        left_sum = max(left_sum, s)
    s = 0
    for i in range(m+1, r+1):
        s += nums[i]
        right_sum = max(right_sum, s)
    return left_sum + right_sum

def maxSubArray(nums):
    def helper(l, r):
        if l == r:
            return nums[l]
        m = (l+r)//2
        return max(helper(l,m), helper(m+1,r), maxCrossingSum(nums,l,m,r))
    return helper(0, len(nums)-1)

print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
```

---

## 10. **Monotonic Stack**

Example: Next Greater Element

```python
def nextGreater(nums):
    res = [-1]*len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]: 
# loop till current number is greater than previous numbers
            res[stack.pop()] = nums[i]
# next greater number found for previous index
        stack.append(i)
# loop by adding next index of array to the stack one at a time
    return res

print(nextGreater([2,1,2,4,3]))  # [4,2,4,-1,-1]
```

---

## 11. **Dynamic Programming**

Example: Longest Increasing Subsequence (LIS)
- Subsequence is derived from a sequence , by removing any or none elements . [7,2,3,4,5] -> [7,2,4] or [7,2,5] or [7.3.5]
- Elements in Subsequence should follow the same order as the Sequence .
- For Increasing subsequence the current element should be greter than previous element [7,2,3,4,5] -> [ 2,4] or [ 2,5] or [ 3.5]
- Longest Increasing Subsequence [2,3,4,5]

```python
def lengthOfLIS(nums):
    dp = [1]*len(nums)
    # each element in array has LIS as 1 initially
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
    # pick an element and traverse to that from the first element
    # if current element is less than the picked element LIS of picked element extends the LIS of current element 
    # If the LIS of current element it is greater previous LIS of picked element 
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

print(lengthOfLIS([10,9,2,5,3,7,101,18]))  # 4
```

---

## 12. **2D Array / Matrix Extension**

Example: 2D Prefix Sum (sumRegion query)

```python
class NumMatrix:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                self.prefix[i+1][j+1] = matrix[i][j] + self.prefix[i][j+1] + self.prefix[i+1][j] - self.prefix[i][j]

    def sumRegion(self, r1, c1, r2, c2):
        return self.prefix[r2+1][c2+1] - self.prefix[r1][c2+1] - self.prefix[r2+1][c1] + self.prefix[r1][c1]

m = NumMatrix([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])
print(m.sumRegion(2,1,4,3))  # 8
```

---
#  ** Most Common Array Algorithm Questions**

## 1. **Searching & Sorting**

* **Find the missing number** in `1…n` (sum or XOR method).
* **Find the duplicate number** (Floyd’s cycle detection / Hashing).
* **Binary Search** in a sorted array. 
* **Search in rotated sorted array**. ( Binary Search )
* ** Sum of Two Numbers ** (Hash Table).
* ** Sum of Triplets ** (two pointers).
* **Merge Intervals**.
* **Sort Colors (Dutch National Flag problem)**.

---

## 2. **Prefix Sum & Subarray**

* **Maximum Subarray Sum** (Kadane’s algorithm).
* **Subarray Sum Equals K** (Prefix sum + HashMap).
* **Maximum Product Subarray**.
* **Equilibrium index** (left sum = right sum).
* **Rainwater Trapping** (prefix max + suffix max / 2 pointers).

---

## 3. **Sliding Window**

* **Longest Substring Without Repeating Characters**.
* **Minimum Window Substring**.
* **Sliding Window Maximum**.
* **Permutation in String** (check if s1 is substring of s2).
* **Longest Subarray with Sum ≤ K**.

---

## 4. **Greedy / Heap-based**

* **Kth Largest Element** (heap / quickselect).
* **Top K Frequent Elements**.
* **Merge K Sorted Arrays**.
* **Find Median from Data Stream**.

---

## 5. **Binary Search on Answer**

* **Find Peak Element**.
* **Minimum in Rotated Sorted Array**.
* **Search in 2D Matrix**.
* **Allocate Minimum Pages / Split Array Largest Sum**.

---

## 6. **Matrix (2D Array)**

* **Rotate Matrix by 90 degrees**.
* **Spiral Order Traversal**.
* **Set Matrix Zeroes**.
* **Word Search** (DFS backtracking).
* **Maximal Rectangle / Largest Square of 1s** (DP).

---

## 7. **Advanced**

* **Longest Increasing Subsequence (LIS)**.
* **Minimum Jumps to Reach End**.
* **Maximum Profit in Stock Buy-Sell** (single / multiple transactions).
* **Count Inversions** (merge sort based).
* **Subarray with XOR = K**.

---

# Summary Table

| Category                | Common Questions                                                           |
| ----------------------- | -------------------------------------------------------------------------- |
| Searching & Sorting     | Missing number, Duplicate number, Two Sum, Three Sum, Rotated Array Search |
| Prefix Sum & Subarray   | Kadane’s, Subarray Sum = K, Rainwater Trapping                             |
| Sliding Window          | Longest Substring w/o Repeat, Min Window Substring, Sliding Window Max     |
| Heap/Greedy             | Kth Largest, Top K Frequent, Merge K Arrays                                |
| Binary Search on Answer | Peak Element, Min in Rotated Array, Split Array                            |
| Matrix                  | Rotate 90°, Spiral Traversal, Set Matrix Zeroes                            |
| Advanced                | LIS, Stock Buy-Sell, Count Inversions                                      |

---

Most interviews pick **medium-level problems** like **Two Sum, Kadane’s, Rainwater Trapping, Rotate Matrix, Kth Largest, Longest Substring Without Repeating Characters**.

arrays are probably the **most common data structure** in coding interviews, and there are several “go-to” algorithmic techniques you’ll see repeatedly.

Here’s a structured list:

---

#  Common Algorithms & Techniques for Array Problems

### 1. **Two-Pointer Technique**

* Used for searching, partitioning, and in-place operations.
* Variants: **fast/slow pointers**, **sliding window**, **meet in the middle**.
* **Examples:**

  * Reverse an array in-place.
  * Two Sum II (sorted array).
  * Move Zeroes.
  * Container With Most Water.

---

### 2. **Sliding Window**

* Optimized way to handle subarray problems (fixed or variable size).
* Keeps a window of elements and slides it across the array.
* **Examples:**

  * Maximum sum subarray of size `k`.
  * Longest substring without repeating characters.
  * Minimum window substring (extension with hashmap).

---

### 3. **Prefix Sum / Cumulative Sum**

* Precompute sums for quick range queries.
* Often combined with hash maps for subarray problems.
* **Examples:**

  * Subarray Sum Equals K.
  * Range Sum Query (LeetCode 303).
  * Maximum Subarray (Kadane’s Algorithm is related).

---

### 4. **Kadane’s Algorithm**

* Special case of prefix sums.
* Finds the maximum subarray sum in **O(n)**.
* **Example:**

  * Maximum Subarray (LeetCode 53).

---

### 5. **Sorting-based Techniques**

* Sort + Two Pointers.
* Sort + Binary Search.
* Sort + Greedy.
* **Examples:**

  * 3Sum, 4Sum.
  * Merge Intervals.
  * Meeting Rooms problems.

---

### 6. **Binary Search on Arrays**

* Search in sorted arrays or “search space”.
* Variants:

  * Lower bound / upper bound.
  * Binary search on answer (search space reduction).
* **Examples:**

  * Search in Rotated Sorted Array.
  * Find Minimum in Rotated Sorted Array.
  * Median of Two Sorted Arrays.

---

### 7. **Hashing (HashSet / HashMap)**

* For quick lookups (O(1) average).
* Common in duplicate, sum, or frequency problems.
* **Examples:**

  * Two Sum.
  * Contains Duplicate.
  * Longest Consecutive Sequence.

---

### 8. **Greedy Algorithms**

* Make local optimal choices for global solution.
* **Examples:**

  * Jump Game.
  * Gas Station problem.
  * Assign Cookies.

---

### 9. **Divide and Conquer**

* Recursively break down array into halves.
* Often used with mergesort / quicksort.
* **Examples:**

  * Maximum Subarray (divide & conquer version).
  * Count Inversions.
  * Majority Element (Boyer-Moore voting is simpler though).

---

### 10. **Monotonic Stack / Monotonic Queue**

* Stack/queue that keeps elements in sorted order.
* Useful for “next greater/smaller element” or sliding window max.
* **Examples:**

  * Trapping Rain Water.
  * Largest Rectangle in Histogram.
  * Daily Temperatures.

---

### 11. **Dynamic Programming on Arrays**

* Used when subarray/subsequence problems have overlapping subproblems.
* **Examples:**

  * Longest Increasing Subsequence (LIS).
  * Maximum Product Subarray.
  * House Robber.

---

### 12. **Matrix/2D Array Extensions**

* Many 1D techniques extend to 2D.
* **Examples:**

  * Prefix Sum → 2D Prefix Sum.
  * Sliding window → Submatrix sliding.
  * Binary Search → Search in 2D Matrix.

---

**Quick Table Summary**

| Technique           | Typical Problems                          |
| ------------------- | ----------------------------------------- |
| Two Pointers        | Reverse, 2Sum, 3Sum                       |
| Sliding Window      | Longest substring, max subarray of size k |
| Prefix Sum / Kadane | Subarray sums, max subarray               |
| Sorting + Greedy    | Merge intervals, jump game                |
| Binary Search       | Rotated array search, median              |
| Hashing             | Duplicates, longest consecutive sequence  |
| Monotonic Stack     | Next greater element, histogram           |
| Dynamic Programming | LIS, house robber, product subarray       |

---

let’s go through **each common array algorithm/technique** with a **Python code example** (easy-to-follow).  each example short and well-commented so you can see the **pattern**.

---



👉 Do you want me to also prepare a **progressive practice plan** (easy → medium → hard LeetCode problems mapped to these techniques) so you can study them step by step?


