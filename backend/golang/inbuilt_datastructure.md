### **Inbuilt Data Structures in Golang**  

Golang provides several built-in **data structures** that help in efficient data manipulation. These include **arrays, slices, maps, and structs**, among others.

---

## **1️⃣ Arrays**  
✅ **Fixed-size, sequential collection of elements** of the same type.  
✅ Stored in **contiguous memory** for fast access.

```go
package main
import "fmt"

func main() {
    var arr [3]int = [3]int{10, 20, 30}
    fmt.Println(arr) // Output: [10 20 30]
}
```
📌 **Limitation:** Cannot change size dynamically.  

---

## **2️⃣ Slices (Dynamic Arrays)**  
✅ **Dynamic size, flexible arrays**.  
✅ Backed by an **underlying array**, but allows **resizing**.  

```go
package main
import "fmt"

func main() {
    slice := []int{1, 2, 3, 4}
    slice = append(slice, 5, 6)
    fmt.Println(slice) // Output: [1 2 3 4 5 6]
}
```
📌 **Slices vs. Arrays:** Slices grow dynamically, while arrays are fixed.  

---

## **3️⃣ Maps (Hash Tables / Dictionaries)**  
✅ **Key-value store** with fast lookups.  
✅ Unordered, but efficient for searching.

```go
package main
import "fmt"

func main() {
    student := map[string]int{"Alice": 85, "Bob": 90}
    fmt.Println(student["Alice"]) // Output: 85

    student["Charlie"] = 95 // Adding a new key-value pair
    delete(student, "Bob")  // Deleting a key
}
```
📌 **Keys are unique** and must be **comparable types (strings, numbers, etc.)**.

---

## **4️⃣ Structs (Custom Data Structures)**  
✅ Similar to **classes** in other languages, but **no inheritance**.  
✅ Used to group related **fields**.

```go
package main
import "fmt"

type Person struct {
    Name string
    Age  int
}

func main() {
    p := Person{Name: "John", Age: 30}
    fmt.Println(p.Name, p.Age) // Output: John 30
}
```
📌 **Structs allow complex data modeling** and are widely used in APIs.

---

## **5️⃣ Pointers (Efficient Memory Management)**  
✅ Store **memory addresses**, avoiding unnecessary copying.

```go
package main
import "fmt"

func main() {
    x := 10
    ptr := &x
    fmt.Println(*ptr) // Output: 10 (dereferencing the pointer)
}
```
📌 **Used for performance optimizations** and modifying values efficiently.

---

## **6️⃣ Interfaces (Dynamic Polymorphism)**  
✅ Define **method contracts** without implementations.  
✅ Used for **duck typing** (objects that implement methods automatically satisfy interfaces).

```go
package main
import "fmt"

type Shape interface {
    Area() float64
}

type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14 * c.Radius * c.Radius
}

func main() {
    var s Shape = Circle{Radius: 5}
    fmt.Println(s.Area()) // Output: 78.5
}
```
📌 **Used in Go for abstraction and reusable code**.

---

## **7️⃣ Channels (Concurrency Communication)**  
✅ Used for **goroutine synchronization and communication**.  
✅ Prevents **race conditions**.

```go
package main
import "fmt"

func worker(ch chan string) {
    ch <- "Hello from Goroutine!"
}

func main() {
    ch := make(chan string)
    go worker(ch)
    fmt.Println(<-ch) // Output: Hello from Goroutine!
}
```
📌 **Key for concurrent programming in Golang**.

---

## **8️⃣ Sync Package (Mutexes, RWMutex, WaitGroups, Atomic Operations)**  
✅ Handles **concurrent data access**.

Example: **Mutex for safe concurrent access**
```go
package main
import (
    "fmt"
    "sync"
)

var mu sync.Mutex
var count int

func increment(wg *sync.WaitGroup) {
    mu.Lock()
    count++
    mu.Unlock()
    wg.Done()
}

func main() {
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go increment(&wg)
    }
    wg.Wait()
    fmt.Println(count) // Output: 5
}
```
📌 **Essential for thread safety** in multi-threaded applications.

---

## **9️⃣ Heap & Priority Queue (Using `container/heap`)**  
✅ Implement **priority queues**, **task scheduling**, and **graph algorithms**.

```go
package main
import (
    "container/heap"
    "fmt"
)

// Define a Min-Heap
type MinHeap []int

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
    *h = append(*h, x.(int))
}

func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func main() {
    h := &MinHeap{3, 2, 1}
    heap.Init(h)
    heap.Push(h, 0)
    fmt.Println(heap.Pop(h)) // Output: 0 (Smallest Element)
}
```
📌 **Useful for implementing priority queues**.

---

## **🔟 Linked List (Using Structs)**
✅ **Not built-in**, but can be implemented using **structs & pointers**.

```go
package main
import "fmt"

type Node struct {
    data int
    next *Node
}

type LinkedList struct {
    head *Node
}

func (l *LinkedList) Insert(value int) {
    newNode := &Node{data: value}
    if l.head == nil {
        l.head = newNode
    } else {
        temp := l.head
        for temp.next != nil {
            temp = temp.next
        }
        temp.next = newNode
    }
}

func (l *LinkedList) Print() {
    temp := l.head
    for temp != nil {
        fmt.Print(temp.data, " -> ")
        temp = temp.next
    }
    fmt.Println("nil")
}

func main() {
    list := &LinkedList{}
    list.Insert(1)
    list.Insert(2)
    list.Insert(3)
    list.Print() // Output: 1 -> 2 -> 3 -> nil
}
```
📌 **Go doesn’t have built-in linked lists** but can be created using pointers.

---

## **📌 Summary: Go’s Built-in Data Structures**
| **Data Structure**   | **Use Case** |
|--------------------|-------------|
| **Array**          | Fixed-size collections |
| **Slice**          | Dynamic-size collections |
| **Map**            | Key-value pairs (hash tables) |
| **Struct**         | Custom data types |
| **Pointer**        | Memory-efficient references |
| **Interface**      | Abstraction & polymorphism |
| **Channel**        | Goroutine communication |
| **Heap (PQ)**      | Priority queues |
| **Mutex & RWMutex** | Synchronization for concurrency |
| **Linked List**    | Implemented via structs & pointers |

Would you like **real-world examples** of these structures? 🚀