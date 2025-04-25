Here’s an implementation of **all major search algorithms** in Go, including:

1. **Linear Search** (O(n))
2. **Binary Search** (O(log n))
3. **Jump Search** (O(√n))
4. **Interpolation Search** (O(log log n))
5. **Exponential Search** (O(log n))
6. **Ternary Search** (O(log n))

---

## **1. Linear Search (O(n))**
Linear search iterates through the entire array.

```go
package main
import "fmt"

func linearSearch(arr []int, target int) int {
	for i, val := range arr {
		if val == target {
			return i
		}
	}
	return -1
}

func main() {
	arr := []int{4, 2, 8, 1, 5}
	target := 8

	index := linearSearch(arr, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Best for unsorted arrays but inefficient for large datasets.**

---

## **2. Binary Search (O(log n))**
Binary search works on **sorted** arrays.

```go
package main
import "fmt"

func binarySearch(arr []int, target int) int {
	left, right := 0, len(arr)-1

	for left <= right {
		mid := left + (right-left)/2

		if arr[mid] == target {
			return mid
		} else if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

func main() {
	arr := []int{1, 3, 5, 7, 9, 11, 15}
	target := 7

	index := binarySearch(arr, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Fast for sorted arrays.**

---

## **3. Jump Search (O(√n))**
Jump Search is **faster than linear search** but works on sorted arrays.

```go
package main
import (
	"fmt"
	"math"
)

func jumpSearch(arr []int, target int) int {
	n := len(arr)
	step := int(math.Sqrt(float64(n))) // Block size to jump
	prev := 0

	for arr[int(math.Min(float64(step), float64(n)))-1] < target {
		prev = step
		step += int(math.Sqrt(float64(n)))
		if prev >= n {
			return -1
		}
	}

	for arr[prev] < target {
		prev++
		if prev == int(math.Min(float64(step), float64(n))) {
			return -1
		}
	}

	if arr[prev] == target {
		return prev
	}
	return -1
}

func main() {
	arr := []int{1, 3, 5, 7, 9, 11, 15, 18, 21}
	target := 9

	index := jumpSearch(arr, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Best for large sorted arrays.**

---

## **4. Interpolation Search (O(log log n))**
Interpolation search works **best for uniformly distributed data**.

```go
package main
import "fmt"

func interpolationSearch(arr []int, target int) int {
	low, high := 0, len(arr)-1

	for low <= high && target >= arr[low] && target <= arr[high] {
		if low == high {
			if arr[low] == target {
				return low
			}
			return -1
		}

		pos := low + ((target - arr[low]) * (high - low) / (arr[high] - arr[low]))

		if arr[pos] == target {
			return pos
		}

		if arr[pos] < target {
			low = pos + 1
		} else {
			high = pos - 1
		}
	}
	return -1
}

func main() {
	arr := []int{10, 20, 30, 40, 50, 60, 70, 80}
	target := 40

	index := interpolationSearch(arr, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Faster than binary search for uniform distributions.**

---

## **5. Exponential Search (O(log n))**
Exponential Search is **useful when we don’t know the size of the array**.

```go
package main
import (
	"fmt"
	"math"
)

func binarySearch(arr []int, left, right, target int) int {
	for left <= right {
		mid := left + (right-left)/2
		if arr[mid] == target {
			return mid
		} else if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

func exponentialSearch(arr []int, target int) int {
	if arr[0] == target {
		return 0
	}

	n := len(arr)
	i := 1
	for i < n && arr[i] <= target {
		i *= 2
	}

	return binarySearch(arr, i/2, int(math.Min(float64(i), float64(n-1))), target)
}

func main() {
	arr := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50}
	target := 10

	index := exponentialSearch(arr, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Efficient for unbounded or large arrays.**

---

## **6. Ternary Search (O(log n))**
Ternary search **divides the array into three parts** instead of two.

```go
package main
import "fmt"

func ternarySearch(arr []int, left, right, target int) int {
	if right >= left {
		mid1 := left + (right-left)/3
		mid2 := right - (right-left)/3

		if arr[mid1] == target {
			return mid1
		}
		if arr[mid2] == target {
			return mid2
		}

		if target < arr[mid1] {
			return ternarySearch(arr, left, mid1-1, target)
		} else if target > arr[mid2] {
			return ternarySearch(arr, mid2+1, right, target)
		} else {
			return ternarySearch(arr, mid1+1, mid2-1, target)
		}
	}
	return -1
}

func main() {
	arr := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
	target := 6

	index := ternarySearch(arr, 0, len(arr)-1, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **More efficient than binary search for some cases.**

---

## **📝 Summary of Time Complexities**
| Algorithm          | Best Case | Average Case | Worst Case |
|-------------------|----------|--------------|------------|
| **Linear Search** | O(1) | O(n) | O(n) |
| **Binary Search** | O(1) | O(log n) | O(log n) |
| **Jump Search** | O(1) | O(√n) | O(√n) |
| **Interpolation Search** | O(1) | O(log log n) | O(n) |
| **Exponential Search** | O(1) | O(log n) | O(log n) |
| **Ternary Search** | O(1) | O(log n) | O(log n) |

Would you like **parallel search algorithms** as well?
Parallel search algorithms can be implemented using **Goroutines** in Go to speed up searching in large datasets. Here are **parallel versions of Linear Search and Binary Search** using Goroutines.

---

## **1. Parallel Linear Search**
We divide the array into multiple chunks and assign each chunk to a separate Goroutine.

```go
package main

import (
	"fmt"
	"sync"
)

// Parallel linear search using Goroutines
func parallelLinearSearch(arr []int, target int, numWorkers int) int {
	var wg sync.WaitGroup
	result := make(chan int, numWorkers)
	chunkSize := len(arr) / numWorkers

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize

		// Ensure the last worker gets the remaining elements
		if i == numWorkers-1 {
			end = len(arr)
		}

		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				if arr[j] == target {
					result <- j
					return
				}
			}
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(result)
	}()

	// Return the first found index
	for index := range result {
		return index
	}

	return -1
}

func main() {
	arr := []int{4, 2, 8, 1, 5, 10, 15, 3, 7, 6, 9, 12}
	target := 7
	numWorkers := 3

	index := parallelLinearSearch(arr, target, numWorkers)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```
✅ **Best for large datasets when no sorting is required.**

---

## **2. Parallel Binary Search**
Binary search is efficient on **sorted arrays**, and we can parallelize the search by dividing the array into two parts.

```go
package main

import (
	"fmt"
	"sync"
)

// Parallel binary search
func parallelBinarySearch(arr []int, target int) int {
	var wg sync.WaitGroup
	result := make(chan int, 1)

	wg.Add(2)
	mid := len(arr) / 2

	// Left half search
	go func() {
		defer wg.Done()
		index := binarySearch(arr[:mid], target)
		if index != -1 {
			result <- index
		}
	}()

	// Right half search
	go func() {
		defer wg.Done()
		index := binarySearch(arr[mid:], target)
		if index != -1 {
			result <- index + mid
		}
	}()

	go func() {
		wg.Wait()
		close(result)
	}()

	for index := range result {
		return index
	}

	return -1
}

// Standard binary search function
func binarySearch(arr []int, target int) int {
	left, right := 0, len(arr)-1
	for left <= right {
		mid := left + (right-left)/2
		if arr[mid] == target {
			return mid
		} else if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

func main() {
	arr := []int{1, 3, 5, 7, 9, 11, 15, 18, 21}
	target := 9

	index := parallelBinarySearch(arr, target)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Useful for large sorted datasets with multi-core processors.**

---

## **3. Concurrent Search Using Worker Pool**
For very large datasets, a worker pool can be used.

```go
package main

import (
	"fmt"
	"sync"
)

// Worker function for searching a segment
func worker(arr []int, target int, start int, wg *sync.WaitGroup, result chan int) {
	defer wg.Done()
	for i, v := range arr {
		if v == target {
			result <- start + i
			return
		}
	}
}

// Parallel search using worker pool
func workerPoolSearch(arr []int, target int, numWorkers int) int {
	var wg sync.WaitGroup
	result := make(chan int, 1)
	chunkSize := len(arr) / numWorkers

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize

		if i == numWorkers-1 {
			end = len(arr)
		}

		go worker(arr[start:end], target, start, &wg, result)
	}

	go func() {
		wg.Wait()
		close(result)
	}()

	for index := range result {
		return index
	}

	return -1
}

func main() {
	arr := []int{5, 12, 8, 1, 15, 22, 30, 3, 10, 9}
	target := 10
	numWorkers := 4

	index := workerPoolSearch(arr, target, numWorkers)
	if index != -1 {
		fmt.Printf("Element found at index %d\n", index)
	} else {
		fmt.Println("Element not found")
	}
}
```

✅ **Best when searching very large arrays using multiple Goroutines.**

---

## **📝 When to Use Parallel Search?**
| Algorithm | Best For | Complexity |
|-----------|---------|-------------|
| **Parallel Linear Search** | Large unsorted arrays | O(n/p) |
| **Parallel Binary Search** | Large sorted arrays | O(log n) |
| **Worker Pool Search** | Massive datasets | O(n/p) |

Would you like an **optimized parallel search for distributed systems** (like MapReduce)?