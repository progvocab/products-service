Here’s an implementation of all major **sorting algorithms** in **Golang**, including:

1. **Bubble Sort**  
2. **Selection Sort**  
3. **Insertion Sort**  
4. **Merge Sort**  
5. **Quick Sort**  
6. **Heap Sort**  
7. **Counting Sort** (For non-negative integers)  
8. **Radix Sort** (For numbers)  

---

## **1️⃣ Bubble Sort (O(n²))**
Bubble Sort repeatedly swaps adjacent elements if they are in the wrong order.

```go
package main
import "fmt"

func bubbleSort(arr []int) {
	n := len(arr)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}

func main() {
	arr := []int{5, 2, 9, 1, 5, 6}
	bubbleSort(arr)
	fmt.Println(arr) // Output: [1 2 5 5 6 9]
}
```

---

## **2️⃣ Selection Sort (O(n²))**
Find the smallest element and place it at the beginning.

```go
package main
import "fmt"

func selectionSort(arr []int) {
	n := len(arr)
	for i := 0; i < n-1; i++ {
		minIdx := i
		for j := i + 1; j < n; j++ {
			if arr[j] < arr[minIdx] {
				minIdx = j
			}
		}
		arr[i], arr[minIdx] = arr[minIdx], arr[i]
	}
}

func main() {
	arr := []int{29, 10, 14, 37, 13}
	selectionSort(arr)
	fmt.Println(arr) // Output: [10 13 14 29 37]
}
```

---

## **3️⃣ Insertion Sort (O(n²))**
Sorts by inserting elements into their correct position.

```go
package main
import "fmt"

func insertionSort(arr []int) {
	n := len(arr)
	for i := 1; i < n; i++ {
		key := arr[i]
		j := i - 1
		for j >= 0 && arr[j] > key {
			arr[j+1] = arr[j]
			j--
		}
		arr[j+1] = key
	}
}

func main() {
	arr := []int{9, 7, 5, 3, 1}
	insertionSort(arr)
	fmt.Println(arr) // Output: [1 3 5 7 9]
}
```

---

## **4️⃣ Merge Sort (O(n log n))**
Recursively splits the array into halves and merges them.

```go
package main
import "fmt"

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

func main() {
	arr := []int{38, 27, 43, 3, 9, 82, 10}
	arr = mergeSort(arr)
	fmt.Println(arr) // Output: [3 9 10 27 38 43 82]
}
```

---

## **5️⃣ Quick Sort (O(n log n), worst case O(n²))**
Chooses a pivot and partitions the array around it.

```go
package main
import "fmt"

func quickSort(arr []int) []int {
	if len(arr) < 2 {
		return arr
	}
	pivot := arr[len(arr)/2]
	left, right := []int{}, []int{}
	for _, v := range arr {
		if v < pivot {
			left = append(left, v)
		} else if v > pivot {
			right = append(right, v)
		}
	}
	return append(append(quickSort(left), pivot), quickSort(right)...)
}

func main() {
	arr := []int{10, 80, 30, 90, 40, 50, 70}
	arr = quickSort(arr)
	fmt.Println(arr) // Output: [10 30 40 50 70 80 90]
}
```

---

## **6️⃣ Heap Sort (O(n log n))**
Uses a **heap data structure** to extract the maximum/minimum element efficiently.

```go
package main
import "fmt"

func heapify(arr []int, n, i int) {
	largest := i
	left, right := 2*i+1, 2*i+2
	if left < n && arr[left] > arr[largest] {
		largest = left
	}
	if right < n && arr[right] > arr[largest] {
		largest = right
	}
	if largest != i {
		arr[i], arr[largest] = arr[largest], arr[i]
		heapify(arr, n, largest)
	}
}

func heapSort(arr []int) {
	n := len(arr)
	for i := n/2 - 1; i >= 0; i-- {
		heapify(arr, n, i)
	}
	for i := n - 1; i > 0; i-- {
		arr[0], arr[i] = arr[i], arr[0]
		heapify(arr, i, 0)
	}
}

func main() {
	arr := []int{4, 10, 3, 5, 1}
	heapSort(arr)
	fmt.Println(arr) // Output: [1 3 4 5 10]
}
```

---

## **7️⃣ Counting Sort (O(n + k))**
Efficient for sorting integers within a small range.

```go
package main
import "fmt"

func countingSort(arr []int) []int {
	max := arr[0]
	for _, v := range arr {
		if v > max {
			max = v
		}
	}
	count := make([]int, max+1)
	for _, v := range arr {
		count[v]++
	}
	index := 0
	for i, c := range count {
		for c > 0 {
			arr[index] = i
			index++
			c--
		}
	}
	return arr
}

func main() {
	arr := []int{4, 2, 2, 8, 3, 3, 1}
	arr = countingSort(arr)
	fmt.Println(arr) // Output: [1 2 2 3 3 4 8]
}
```

---

## **8️⃣ Radix Sort (O(nk))**
Used for sorting numbers digit by digit.

```go
package main
import "fmt"

func getMax(arr []int) int {
	max := arr[0]
	for _, v := range arr {
		if v > max {
			max = v
		}
	}
	return max
}

func countingSortByDigit(arr []int, exp int) {
	n := len(arr)
	output := make([]int, n)
	count := make([]int, 10)

	for _, v := range arr {
		count[(v/exp)%10]++
	}

	for i := 1; i < 10; i++ {
		count[i] += count[i-1]
	}

	for i := n - 1; i >= 0; i-- {
		index := (arr[i] / exp) % 10
		output[count[index]-1] = arr[i]
		count[index]--
	}

	copy(arr, output)
}

func radixSort(arr []int) {
	max := getMax(arr)
	for exp := 1; max/exp > 0; exp *= 10 {
		countingSortByDigit(arr, exp)
	}
}

func main() {
	arr := []int{170, 45, 75, 90, 802, 24, 2, 66}
	radixSort(arr)
	fmt.Println(arr) // Output: [2 24 45 66 75 90 170 802]
}
```

---

Let me know if you need **additional explanations** or **optimizations**! 🚀