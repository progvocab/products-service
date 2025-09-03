Great question 🚀 Let’s carefully break down the **Search in Rotated Sorted Array** problem (LeetCode #33).

---

## 🔹 Problem Statement

You’re given a **sorted array** that has been rotated at some pivot (unknown).
You need to find the **index of a target element** in `O(log n)` time (binary search).
If not found, return `-1`.

👉 Example:
Input: `nums = [4,5,6,7,0,1,2], target = 0`
Output: `4`

Input: `nums = [4,5,6,7,0,1,2], target = 3`
Output: `-1`

---

## 🔹 Key Insight

* Normally, binary search works on a **sorted** array.
* Here, the array is "rotated", but **at least one half (left or right) is always sorted**.
* We can use this property to decide which side to search.

---

## 🔹 Algorithm (Modified Binary Search)

1. Initialize `left = 0`, `right = n-1`.
2. While `left <= right`:

   * Find `mid = (left + right) // 2`.
   * If `nums[mid] == target`, return `mid`.
   * Now check **which half is sorted**:

     * If `nums[left] <= nums[mid]`, then **left half is sorted**:

       * If `nums[left] <= target < nums[mid]`, search left → `right = mid - 1`.
       * Else search right → `left = mid + 1`.
     * Else, **right half is sorted**:

       * If `nums[mid] < target <= nums[right]`, search right → `left = mid + 1`.
       * Else search left → `right = mid - 1`.
3. If not found, return `-1`.

---

## 🔹 Python Code

```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


# Example usage
nums = [4,5,6,7,0,1,2]
print(search(nums, 0))  # Output: 4
print(search(nums, 3))  # Output: -1
```

---

## 🔹 Complexity

* **Time Complexity**: `O(log n)` (binary search).
* **Space Complexity**: `O(1)`.

---

✅ Works for positive & negative numbers, and handles rotation at any pivot.

Do you also want me to explain the **Search in Rotated Sorted Array II** (LeetCode #81), where duplicates are allowed?
