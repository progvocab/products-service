### **Generics in Golang**  

**Generics** in Go allow writing reusable functions and data structures that work with multiple types **without type assertion or duplication**.  

---

## **1. Why Use Generics?**  
Without generics, Go functions must:  
1. Use **interface{}** (loses type safety).  
2. Write duplicate functions for different types.  

Generics solve this by allowing **type parameters**.

---

## **2. Syntax of Generics**
```go
func FunctionName[T any](param T) T { ... }
```
- `T` is a **type parameter** (can be `int`, `string`, etc.).
- `any` is a **constraint** (means any type is allowed).

---

## **3. Generic Function Example**  
A **generic function** that works with any type:
```go
package main

import "fmt"

// Generic function to return the same type
func PrintValue[T any](value T) {
    fmt.Println(value)
}

func main() {
    PrintValue(42)       // Output: 42
    PrintValue("Hello")  // Output: Hello
    PrintValue(3.14)     // Output: 3.14
}
```
✅ **`T any`** → Accepts all types.  

---

## **4. Generic Data Structures**  
### **a) Generic Stack (LIFO)**
```go
package main

import "fmt"

// Generic Stack
type Stack[T any] struct {
    items []T
}

// Push element
func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

// Pop element
func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zeroVal T
        return zeroVal, false
    }
    last := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return last, true
}

func main() {
    intStack := Stack[int]{}
    intStack.Push(10)
    intStack.Push(20)
    val, _ := intStack.Pop()
    fmt.Println(val) // Output: 20

    stringStack := Stack[string]{}
    stringStack.Push("Go")
    stringStack.Push("Generics")
    valStr, _ := stringStack.Pop()
    fmt.Println(valStr) // Output: Generics
}
```
✅ **Reuses Stack for multiple types (`int`, `string`)**.  

---

## **5. Constraints in Generics**  
Use constraints to restrict type parameters.

### **a) Using `constraints.Ordered` (For Numeric Comparisons)**
```go
package main

import (
    "fmt"
    "golang.org/x/exp/constraints"
)

// Constraint: Only numeric types
func Min[T constraints.Ordered](a, b T) T {
    if a < b {
        return a
    }
    return b
}

func main() {
    fmt.Println(Min(3, 7))      // Output: 3
    fmt.Println(Min(3.5, 1.2))  // Output: 1.2
}
```
✅ **Prevents passing non-numeric types.**  

---

### **b) Custom Interface Constraint**
Define a constraint for **specific struct types**:
```go
package main

import "fmt"

// Define a constraint
type Adder interface {
    Add() int
}

// Implement Adder interface
type Numbers struct {
    x, y int
}

func (n Numbers) Add() int {
    return n.x + n.y
}

// Generic function with constraint
func Sum[T Adder](n T) int {
    return n.Add()
}

func main() {
    num := Numbers{10, 20}
    fmt.Println(Sum(num)) // Output: 30
}
```
✅ **Ensures only structs implementing `Add()` are used.**  

---

## **6. Using Multiple Type Parameters**
A function with **two different types**:
```go
package main

import "fmt"

// Generic function with multiple types
func Pair[T, U any](first T, second U) {
    fmt.Println("First:", first, "Second:", second)
}

func main() {
    Pair("Hello", 100)   // Output: First: Hello Second: 100
    Pair(3.14, true)     // Output: First: 3.14 Second: true
}
```
✅ **Supports mixing types (`string, int`, `float, bool`).**  

---

## **7. Generic Map Function**
Apply a function to each element of a slice:
```go
package main

import "fmt"

// Generic Map function
func Map[T any, R any](arr []T, fn func(T) R) []R {
    result := make([]R, len(arr))
    for i, v := range arr {
        result[i] = fn(v)
    }
    return result
}

func main() {
    nums := []int{1, 2, 3, 4}
    squares := Map(nums, func(n int) int { return n * n })
    fmt.Println(squares) // Output: [1 4 9 16]
}
```
✅ **Applies a function to all elements in a slice.**  

---

## **Key Takeaways**
✅ **Generics allow reusability** for different types.  
✅ **Type constraints** restrict valid types.  
✅ **Supports multiple type parameters** (`T, U`).  
✅ **Useful for collections like slices, maps, and stacks.**  

Would you like an example of **generic linked lists?**