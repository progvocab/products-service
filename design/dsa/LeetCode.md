### 1 

Problem (Two Sum)
Given an integer array nums and a target, find two numbers whose sum equals target.
Return their indices.
Each input has exactly one solution.
You may not use the same element twice.
Order of answer does not matter.


---

Solution Steps (HashMap):

1. Initialize an empty hashmap.


2. Traverse array from left to right.


3. For each nums[i], compute complement = target - nums[i].


4. Check if complement exists in hashmap.


5. If yes → return [map[complement], i].


6. Else store nums[i] → i in hashmap.


7. Continue until solution is found.




---

Pattern:
Hashing (Lookup Optimization)


---

Advantage over Brute Force:
Reduces from O(n²) pair checking to O(n).
Constant-time lookup using hashmap.
Single-pass efficient solution.

### 2

Problem (Add Two Numbers)
Given two non-empty linked lists representing two non-negative integers.
Digits are stored in reverse order.
Add the two numbers and return the sum as a linked list.
Each node contains a single digit.
Handle carry properly.


---

Solution Steps (Linked List Traversal):

1. Create a dummy node and curr pointer.


2. Initialize carry = 0.


3. Traverse both lists while nodes exist or carry remains.


4. Take values from current nodes (0 if null).


5. Compute sum = val1 + val2 + carry.


6. Create new node with sum % 10.


7. Update carry = sum / 10.


8. Move pointers forward.


9. Return dummy.next.




---

Pattern:
Linked List Simulation


---

Advantage over Brute Force:
Single traversal → O(max(n,m)).
No conversion to integers (avoids overflow).
Efficient carry handling node by node.


### 3

Problem (Longest Substring Without Repeating Characters)
Given a string s, find the length of the longest substring without repeating characters.
Substring must be contiguous.
Characters can repeat outside the chosen substring.
Return only the maximum length.
String may be empty.


---

Solution Steps (Sliding Window):

1. Use two pointers left and right.


2. Maintain a hashmap/set for character positions.


3. Expand right one character at a time.


4. If duplicate found inside window:


5. Move left to max(left, lastSeen[char] + 1).


6. Update current character’s latest index.


7. Compute window size = right - left + 1.


8. Update maximum length.


9. Continue until end.




---

Pattern:
Sliding Window + HashMap


---

Advantage over Brute Force:
Avoids checking all substrings O(n²).
Single pass → O(n).
Efficient duplicate tracking with hashmap.


### 4


Problem (Median of Two Sorted Arrays)
Given two sorted arrays nums1 and nums2, find the median of the combined sorted array.
Overall runtime must be O(log(m+n)).
Arrays may have different sizes.
Median is middle element(s) after merge.
Avoid full merge.


---

Solution Steps (Binary Search Partition):

1. Always binary search on the smaller array.


2. Partition both arrays so left half size equals right half.


3. Compute partition indices i and j.


4. Find left max and right min around partitions.


5. If maxLeft1 <= minRight2 and maxLeft2 <= minRight1:


6. Correct partition found.


7. If total length odd → median = max(left values).


8. Else → median = average of max(left) and min(right).


9. Adjust binary search if partition invalid.




---

Pattern:
Binary Search on Partition


---

Advantage over Brute Force:
Avoids full merge O(m+n).
Achieves O(log(min(m,n))).
Efficient for large sorted arrays.


### 5


Problem (Longest Palindromic Substring)
Given a string s, find the longest palindromic substring.
Substring must be contiguous.
Palindrome reads same forward and backward.
Return the longest such substring.
If multiple exist, return any.


---

Solution Steps (Expand Around Center):

1. Iterate through each index as palindrome center.


2. Expand for odd-length palindrome (i, i).


3. Expand for even-length palindrome (i, i+1).


4. While chars match, expand outward.


5. Track longest palindrome boundaries.


6. Update max length if larger found.


7. Continue for all centers.


8. Return substring using saved boundaries.




---

Pattern:
Two Pointers (Center Expansion)


---

Advantage over Brute Force:
Avoids checking all substrings O(n³).
Runs in O(n²).
Simple and space-efficient O(1).


### 6

Problem (Zigzag Conversion)
Given a string s and number of rows numRows, write the characters in zigzag pattern.
Read row by row to form the result.
Pattern alternates down and diagonally up.
Return the converted string.
If numRows == 1, return original string.


---

Solution Steps (Simulation):

1. Create numRows string builders.


2. Track current row and direction (down/up).


3. Traverse each character in s.


4. Append character to current row.


5. If top row → move downward.


6. If bottom row → move upward.


7. Update current row based on direction.


8. After traversal, concatenate all rows.


9. Return final string.




---

Pattern:
String Simulation / Matrix Traversal


---

Advantage over Brute Force:
Avoids building full 2D grid.
Single pass → O(n).
Space optimized using row builders only.


### 7

Problem (Reverse Integer)
Given a signed 32-bit integer x, reverse its digits.
Preserve the sign.
If reversed integer overflows 32-bit signed range, return 0.
Do not use 64-bit storage for final answer.
Return the reversed integer.


---

Solution Steps (Math / Digit Extraction):

1. Initialize rev = 0.


2. While x != 0:


3. Extract last digit: digit = x % 10.


4. Remove last digit: x = x / 10.


5. Check overflow before updating rev.


6. If overflow possible → return 0.


7. Update rev = rev * 10 + digit.


8. Continue until x becomes 0.


9. Return rev.




---

Pattern:
Math Simulation / Digit Manipulation


---

Advantage over Brute Force:
Processes digits in single pass O(log n).
No string conversion needed.
Efficient overflow-safe reversal.


### 8
Problem (String to Integer (atoi))
Convert a string into a 32-bit signed integer.
Ignore leading spaces.
Handle optional + or - sign.
Stop reading at first invalid character.
Clamp result within integer range if overflow occurs.


---

Solution Steps (String Parsing):

1. Skip leading spaces.


2. Check for optional sign (+/-).


3. Initialize result = 0.


4. Traverse digits until non-digit appears.


5. Before adding digit, check overflow.


6. If overflow → return INT_MAX or INT_MIN.


7. Update result = result * 10 + digit.


8. Continue until end or invalid character.


9. Return sign * result.




---

Pattern:
String Parsing / State Handling


---

Advantage over Brute Force:
Single pass → O(n).
Efficient overflow-safe conversion.
Avoids unnecessary substring operations.


### 9
Problem (Palindrome Number)
Given an integer x, determine if it is a palindrome.
A palindrome reads the same forward and backward.
Negative numbers are not palindromes.
Do not convert integer to string.
Return true if palindrome, else false.


---

Solution Steps (Reverse Half):

1. If x < 0 or ends with 0 (except 0 itself) → return false.


2. Initialize reversedHalf = 0.


3. While x > reversedHalf:


4. Extract last digit and append to reversedHalf.


5. Remove last digit from x.


6. Continue until half is reversed.


7. For even digits: check x == reversedHalf.


8. For odd digits: check x == reversedHalf / 10.


9. Return result.




---

Pattern:
Math / Two-Half Reversal


---

Advantage over Brute Force:
Avoids full reversal.
Runs in O(log n).
No extra string conversion space.

### 10
Problem (Regular Expression Matching)
Given a string s and pattern p, implement regex matching.
. matches any single character.
* matches zero or more of the preceding element.
Match must cover the entire string.
Return true if fully matched.


---

Solution Steps (Dynamic Programming):

1. Create DP table dp[m+1][n+1].


2. dp[0][0] = true (empty matches empty).


3. Pre-fill patterns like a*, a*b* for empty string.


4. Traverse s and p.


5. If chars match or . → dp[i][j] = dp[i-1][j-1].


6. If *:


7. Treat as zero occurrence → dp[i][j] = dp[i][j-2].


8. Or one/more occurrence if preceding char matches.


9. Final answer is dp[m][n].




---

Pattern:
Dynamic Programming (String Matching)


---

Advantage over Brute Force:
Avoids exponential recursion/backtracking.
Efficient O(m * n) solution.
Stores overlapping subproblem states.


### 11

Problem (Container With Most Water)
Given an array height[], each element represents a vertical line.
Pick two lines such that they form a container with the x-axis.
The container holds water equal to width × min(height).
Return the maximum water that can be stored.
You cannot tilt the container.


---

Solution Steps (Two Pointer):

1. Initialize two pointers: left = 0, right = n-1.


2. Compute area = (right - left) * min(height[left], height[right]).


3. Track maximum area.


4. Move the pointer with smaller height inward.


5. Reason: larger height may improve min(height).


6. If equal heights, move any pointer.


7. Repeat until left < right.


8. Continuously update max area.


9. Return the maximum found.




---

Pattern:
Two Pointers + Greedy Optimization


---

Advantage over Brute Force:
Brute force checks all pairs → O(n²).
Two-pointer approach reduces to O(n).
Avoids redundant comparisons intelligently.


Problem (Integer to Roman)
Given an integer num, convert it to a Roman numeral.
Roman numerals use symbols like I, V, X, L, C, D, M.
Specific subtraction rules apply (e.g., IV = 4, IX = 9).
Input is guaranteed within range 1 to 3999.
Return the Roman numeral string.


---

Solution Steps (Greedy):

1. Maintain arrays of values: [1000,900,500,...,1].


2. Maintain corresponding symbols: ["M","CM","D",...,"I"].


3. Initialize empty result string.


4. Iterate through values from largest to smallest.


5. While num >= value[i]:


6. Append symbol[i] to result.


7. Subtract value[i] from num.


8. Continue until num becomes 0.


9. Return the result string.




---

Pattern:
Greedy (largest value first)


---

Advantage over Brute Force:
Direct mapping avoids repeated checks.
Linear pass over fixed-size arrays → O(1).
Handles subtractive cases efficiently.


Problem (Roman to Integer)
Given a Roman numeral string s, convert it to an integer.
Symbols: I, V, X, L, C, D, M with fixed values.
Subtractive cases exist (e.g., IV = 4, IX = 9).
String is valid and within range 1 to 3999.
Return the integer value.


---

Solution Steps (Greedy / Traversal):

1. Create a map of Roman symbols to values.


2. Initialize result = 0.


3. Traverse string from left to right.


4. If current value < next value:


5. Subtract current value from result.


6. Else add current value to result.


7. Continue till end of string.


8. Return final result.




---

Pattern:
Greedy + String Traversal


---

Advantage over Brute Force:
Single pass → O(n).
No need to check all combinations.
Handles subtractive pairs efficiently.

Problem (Longest Common Prefix)
Given an array of strings, find the longest common prefix.
If no common prefix exists, return an empty string.
Prefix must be shared from the start of all strings.
All strings contain lowercase letters.
Return the common prefix.


---

Solution Steps (Horizontal Scanning):

1. Initialize prefix as first string.


2. Iterate through remaining strings.


3. Compare current string with prefix.


4. Reduce prefix until it matches the start.


5. Use substring to shrink prefix.


6. If prefix becomes empty, return "".


7. Continue until all strings processed.


8. Return final prefix.




---

Pattern:
String Traversal / Greedy Reduction


---

Advantage over Brute Force:
Avoids checking all substrings.
Reduces comparisons progressively.
Efficient O(n * m) vs exponential checks.

Problem (3Sum)
Given an integer array nums, find all unique triplets [a,b,c] such that a + b + c = 0.
Triplets must be unique (no duplicates).
Order of output does not matter.
Array may contain positive, negative, and zero values.
Return list of all valid triplets.


---

Solution Steps (Two Pointer after Sorting):

1. Sort the array.


2. Iterate i from 0 to n-1.


3. Skip duplicates for i.


4. Set left = i+1, right = n-1.


5. Compute sum = nums[i] + nums[left] + nums[right].


6. If sum == 0 → add triplet, move both pointers skipping duplicates.


7. If sum < 0 → move left++.


8. If sum > 0 → move right--.


9. Repeat until left < right.




---

Pattern:
Sorting + Two Pointers


---

Advantage over Brute Force:
Reduces from O(n³) to O(n²).
Eliminates duplicates efficiently.
Uses sorted structure for optimal traversal.


Problem (3Sum Closest)
Given an integer array nums and a target value, find three integers whose sum is closest to the target.
Return the sum of the three integers.
Exactly one solution exists.
Array may contain positive, negative, and zero values.
Minimize absolute difference from target.


---

Solution Steps (Two Pointer after Sorting):

1. Sort the array.


2. Initialize closest sum with first three elements.


3. Iterate i from 0 to n-1.


4. Set left = i+1, right = n-1.


5. Compute current sum = nums[i] + nums[left] + nums[right].


6. Update closest if |sum - target| is smaller.


7. If sum < target → move left++.


8. If sum > target → move right--.


9. If sum == target → return immediately.


10. Continue until all possibilities checked.




---

Pattern:
Sorting + Two Pointers


---

Advantage over Brute Force:
Reduces time from O(n³) to O(n²).
Efficiently narrows search using pointers.
Avoids checking all triplets explicitly.


Problem (Letter Combinations of a Phone Number)
Given a string of digits (2–9), return all possible letter combinations.
Digits map to letters like a phone keypad.
Each digit contributes multiple characters.
Return all possible combinations.
If input is empty, return empty list.


---

Solution Steps (Backtracking):

1. Create digit → letters mapping.


2. Initialize result list.


3. Use recursive backtracking function.


4. At each step, pick letters for current digit.


5. Append letter to current combination.


6. Recurse for next digit.


7. When length == digits length → add to result.


8. Backtrack by removing last character.


9. Start recursion from index 0.


10. Return result list.




---

Pattern:
Backtracking (Combinatorial Generation)


---

Advantage over Brute Force:
Prunes invalid paths early.
Generates only valid combinations.
Structured recursion avoids redundant work.

Problem (4Sum)
Given an array nums and a target, find all unique quadruplets [a,b,c,d] such that their sum equals target.
Quadruplets must be unique (no duplicates).
Array may contain positive, negative, and zero values.
Return all valid quadruplets.
Order of output does not matter.


---

Solution Steps (Sorting + Two Pointers):

1. Sort the array.


2. Fix first index i from 0 to n-1 (skip duplicates).


3. Fix second index j from i+1 to n-1 (skip duplicates).


4. Set left = j+1, right = n-1.


5. Compute sum = nums[i] + nums[j] + nums[left] + nums[right].


6. If sum == target → add quadruplet, move both pointers skipping duplicates.


7. If sum < target → left++.


8. If sum > target → right--.


9. Repeat until left < right.




---

Pattern:
Sorting + Two Pointers (k-Sum reduction)


---

Advantage over Brute Force:
Reduces from O(n⁴) to O(n³).
Efficient duplicate handling.
Leverages sorted order for pruning.

Problem (Remove Nth Node From End of List)
Given a singly linked list, remove the nth node from the end.
Return the head of the modified list.
List size is at least 1.
You must do it in one pass.
n is always valid.


---

Solution Steps (Two Pointer):

1. Create a dummy node pointing to head.


2. Initialize two pointers: fast and slow at dummy.


3. Move fast n+1 steps ahead.


4. Move both fast and slow until fast reaches null.


5. Now slow is before the target node.


6. Update slow.next = slow.next.next.


7. This removes the nth node from end.


8. Return dummy.next.




---

Pattern:
Two Pointers (Fast & Slow)


---

Advantage over Brute Force:
Single pass → O(n).
No need to calculate length first.
Efficient pointer manipulation.


### 23

Problem (Merge k Sorted Lists)
Given an array of k sorted linked lists, merge them into one sorted list.
Return the merged sorted linked list.
Total number of nodes can be large.
Lists may be empty.
Maintain sorted order.


---

Solution Steps (Min Heap / Priority Queue):

1. Initialize a min-heap.


2. Push head of each non-empty list into heap.


3. Create a dummy node and curr pointer.


4. While heap is not empty:


5. Pop smallest node from heap.


6. Attach it to curr.next.


7. Move curr forward.


8. If popped node has next → push into heap.


9. Continue until heap is empty.


10. Return dummy.next.




---

Pattern:
Heap (Priority Queue) + k-way Merge


---

Advantage over Brute Force:
Reduces from O(nk) to O(n log k).
Efficiently finds smallest among k lists.
Scales well for large k.

### 24


Problem (Swap Nodes in Pairs)
Given a linked list, swap every two adjacent nodes.
Do not modify node values, only change pointers.
Return the modified list head.
If odd number of nodes, last node remains as is.
List can be empty.


---

Solution Steps (Iterative Pointer Manipulation):

1. Create a dummy node pointing to head.


2. Initialize prev = dummy.


3. While prev.next and prev.next.next exist:


4. Let first = prev.next, second = first.next.


5. Update first.next = second.next.


6. Update second.next = first.


7. Update prev.next = second.


8. Move prev = first (next pair).


9. Continue until no more pairs.


10. Return dummy.next.




---

Pattern:
Linked List Pointer Manipulation


---

Advantage over Brute Force:
Single pass → O(n).
No extra space required.
Efficient in-place swapping without value changes.


### 25

Problem (Reverse Nodes in k-Group)
Given a linked list, reverse nodes in groups of size k.
If remaining nodes are fewer than k, keep them as is.
Only modify pointers, not values.
Return the modified list head.
List length ≥ 0.


---

Solution Steps (Iterative Reversal in Groups):

1. Create a dummy node pointing to head.


2. Use prevGroupEnd to track previous group end.


3. Find the kth node from current position.


4. If less than k nodes → break.


5. Mark group start and next group head.


6. Reverse nodes within the group.


7. Connect reversed group with previous part.


8. Update prevGroupEnd to new end of group.


9. Move to next group and repeat.


10. Return dummy.next.




---

Pattern:
Linked List + Reversal (Chunk Processing)


---

Advantage over Brute Force:
Processes in-place → O(1) space.
Avoids repeated traversal per group.
Efficient O(n) single-pass approach.


### 26

Problem (Remove Duplicates from Sorted Array)
Given a sorted array nums, remove duplicates in-place.
Each unique element should appear only once.
Return the number of unique elements k.
First k elements should hold the result.
Order must be preserved.


---

Solution Steps (Two Pointer):

1. Initialize i = 0 (slow pointer).


2. Iterate j from 1 to n-1 (fast pointer).


3. If nums[j] != nums[i]:


4. Increment i.


5. Set nums[i] = nums[j].


6. Continue till end.


7. Return i + 1 as count of unique elements.




---

Pattern:
Two Pointers (Slow & Fast)


---

Advantage over Brute Force:
In-place → O(1) extra space.
Single pass → O(n).
Avoids shifting elements repeatedly.

### 27


Problem (Remove Element)
Given an array nums and a value val, remove all occurrences of val in-place.
Return the new length k after removal.
First k elements should not contain val.
Order of elements can be changed.
Use O(1) extra space.


---

Solution Steps (Two Pointer):

1. Initialize i = 0 (position for next valid element).


2. Iterate j from 0 to n-1.


3. If nums[j] != val:


4. Assign nums[i] = nums[j].


5. Increment i.


6. Continue till end.


7. Return i as new length.




---

Pattern:
Two Pointers (Filter In-Place)


---

Advantage over Brute Force:
Single pass → O(n).
In-place → no extra space.
Avoids costly element shifting repeatedly.

### 28 

Problem (Find the Index of the First Occurrence in a String)
Given two strings haystack and needle, find the first occurrence of needle in haystack.
Return the starting index if found, else return -1.
Matching must be exact and contiguous.
Both strings consist of lowercase letters.
Return 0 if needle is empty.


---

Solution Steps (Sliding Window / Brute Optimized):

1. Let n = len(haystack), m = len(needle).


2. Iterate i from 0 to n - m.


3. For each i, compare substring of length m.


4. Check character by character.


5. If all match → return i.


6. If mismatch → move to next index.


7. Continue till end.


8. If no match found → return -1.




---

Pattern:
Sliding Window (Fixed Size) / String Matching


---

Advantage over Brute Force:
Avoids checking unnecessary indices beyond n - m.
Simple and efficient for moderate input sizes.
Time complexity O(n * m), optimized iteration bounds.

### 29

Problem (Divide Two Integers)
Given two integers dividend and divisor, divide them without using multiplication, division, or mod.
Return the quotient after division.
Truncate toward zero.
Handle overflow within 32-bit signed integer range.
Return 2³¹ - 1 if overflow occurs.


---

Solution Steps (Bit Manipulation):

1. Handle edge case: overflow (INT_MIN / -1).


2. Convert both numbers to absolute long values.


3. Initialize result = 0.


4. Loop while dividend >= divisor:


5. Use temp = divisor, multiple = 1.


6. Double temp (temp << 1) while ≤ dividend.


7. Subtract temp from dividend.


8. Add multiple to result.


9. Repeat until dividend < divisor.


10. Apply sign and return result.




---

Pattern:
Bit Manipulation + Greedy (Exponential Subtraction)


---

Advantage over Brute Force:
Reduces repeated subtraction to logarithmic steps.
Time complexity O(log n).
Efficient handling of large values.




