### **Golang `sync` Package – Concurrency Primitives**  

The **`sync` package** in Golang provides concurrency utilities for **safe access to shared resources** and efficient **synchronization**.

---

## **1️⃣ sync.Mutex (Mutual Exclusion Lock)**
✅ Ensures **only one Goroutine** accesses a shared resource at a time.  

```go
package main
import (
    "fmt"
    "sync"
    "time"
)

var counter int
var mutex sync.Mutex

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    mutex.Lock()   // Lock before modifying shared resource
    counter++
    fmt.Printf("Worker %d incremented counter to %d\n", id, counter)
    mutex.Unlock() // Unlock after modification
}

func main() {
    var wg sync.WaitGroup
    for i := 1; i <= 5; i++ {
        wg.Add(1)
        go worker(i, &wg)
    }
    wg.Wait()
}
```
📌 **Prevents race conditions** when multiple Goroutines access `counter`.

---

## **2️⃣ sync.RWMutex (Read-Write Lock)**
✅ **Multiple readers allowed, only one writer at a time**.  

```go
package main
import (
    "fmt"
    "sync"
    "time"
)

var data int
var rwMutex sync.RWMutex

func reader(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    rwMutex.RLock()
    fmt.Printf("Reader %d read data: %d\n", id, data)
    rwMutex.RUnlock()
}

func writer(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    rwMutex.Lock()
    data++
    fmt.Printf("Writer %d updated data to: %d\n", id, data)
    rwMutex.Unlock()
}

func main() {
    var wg sync.WaitGroup
    for i := 1; i <= 3; i++ {
        wg.Add(1)
        go reader(i, &wg)
    }
    wg.Add(1)
    go writer(1, &wg)
    wg.Wait()
}
```
📌 **Multiple readers can read, but only one writer can modify data at a time.**

---

## **3️⃣ sync.WaitGroup (Waiting for Goroutines)**
✅ **Waits for multiple Goroutines** to finish execution.  

```go
package main
import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done() // Decrements the counter when done
    fmt.Printf("Worker %d is processing\n", id)
    time.Sleep(time.Second)
}

func main() {
    var wg sync.WaitGroup
    for i := 1; i <= 3; i++ {
        wg.Add(1)  // Increments the counter
        go worker(i, &wg)
    }
    wg.Wait() // Blocks until all workers finish
    fmt.Println("All workers completed")
}
```
📌 Ensures **main Goroutine waits** until all workers finish execution.

---

## **4️⃣ sync.Once (Run Function Only Once)**
✅ Ensures a **function is executed only once**, even in multiple Goroutines.

```go
package main
import (
    "fmt"
    "sync"
)

var once sync.Once

func initConfig() {
    fmt.Println("Initializing config...")
}

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    once.Do(initConfig) // Ensures initConfig runs only once
    fmt.Printf("Worker %d started\n", id)
}

func main() {
    var wg sync.WaitGroup
    for i := 1; i <= 5; i++ {
        wg.Add(1)
        go worker(i, &wg)
    }
    wg.Wait()
}
```
📌 **`initConfig()` runs only once, no matter how many Goroutines call it.**

---

## **5️⃣ sync.Cond (Condition Variables)**
✅ **Synchronizes Goroutines** waiting for a condition to be met.

```go
package main
import (
    "fmt"
    "sync"
    "time"
)

var ready = false
var cond = sync.NewCond(&sync.Mutex{})

func producer() {
    cond.L.Lock()
    time.Sleep(time.Second) // Simulate work
    ready = true
    fmt.Println("Producer: Data is ready")
    cond.Broadcast() // Notify all waiting Goroutines
    cond.L.Unlock()
}

func consumer(id int) {
    cond.L.Lock()
    for !ready {
        cond.Wait() // Wait until condition is met
    }
    fmt.Printf("Consumer %d processed data\n", id)
    cond.L.Unlock()
}

func main() {
    var wg sync.WaitGroup
    wg.Add(3)
    go consumer(1)
    go consumer(2)
    go consumer(3)
    
    time.Sleep(time.Second)
    go producer()
    
    wg.Wait()
}
```
📌 **Consumers wait until `ready` is `true`, then process the data.**

---

## **6️⃣ sync.Map (Thread-Safe Map)**
✅ A **concurrent-safe** map for read/write operations.

```go
package main
import (
    "fmt"
    "sync"
)

func main() {
    var sm sync.Map

    // Writing to map
    sm.Store("name", "Alice")
    sm.Store("age", 30)

    // Reading from map
    if value, ok := sm.Load("name"); ok {
        fmt.Println("Name:", value)
    }

    // Iterating over map
    sm.Range(func(key, value interface{}) bool {
        fmt.Println(key, ":", value)
        return true
    })
}
```
📌 **`sync.Map` is optimized for concurrent access**, unlike `map`.

---

### **🔹 Summary Table**
| Sync Type       | Purpose |
|----------------|------------------------------------------------|
| `sync.Mutex` | Ensures **one Goroutine** accesses a resource at a time. |
| `sync.RWMutex` | Allows **multiple readers but only one writer** at a time. |
| `sync.WaitGroup` | Waits for **multiple Goroutines** to complete. |
| `sync.Once` | Ensures **a function runs only once** across Goroutines. |
| `sync.Cond` | **Goroutines wait** until a condition is met. |
| `sync.Map` | **Thread-safe concurrent map** (avoids race conditions). |

Would you like more **real-world applications** of these? 🚀