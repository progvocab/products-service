## **Goroutines in Go**
A **goroutine** is a lightweight thread managed by the Go runtime. It enables concurrent execution of functions with minimal memory overhead, making Go ideal for high-performance and scalable applications.

---

### **🔹 Why Use Goroutines?**
1. **Efficient Concurrency:** Unlike OS threads, goroutines use much less memory (~2 KB).
2. **Automatic Management:** The Go runtime schedules goroutines efficiently.
3. **Scalability:** Can handle millions of goroutines with minimal resource usage.
4. **Non-blocking Execution:** Functions can run asynchronously without blocking execution.

---

## **1. Basic Goroutine Example**
A goroutine is created by prefixing a function call with `go`.

### **Example: Running a Function as a Goroutine**
```go
package main
import (
	"fmt"
	"time"
)

func sayHello() {
	fmt.Println("Hello from Goroutine!")
}

func main() {
	go sayHello() // Starts a new Goroutine
	fmt.Println("Main function execution")
	time.Sleep(time.Second) // Prevent main from exiting immediately
}
```
### **Output**
```
Main function execution
Hello from Goroutine!
```
🔹 **Explanation:**  
- The `sayHello()` function runs in a separate goroutine.  
- The `main()` function does not wait for it to finish.  
- `time.Sleep(1 * time.Second)` allows the goroutine to execute before the program exits.

---

## **2. Goroutine with Anonymous Function**
You can also use **anonymous functions** as goroutines.

```go
package main
import (
	"fmt"
	"time"
)

func main() {
	go func() {
		fmt.Println("Goroutine with anonymous function")
	}()
	time.Sleep(time.Second) // Allows goroutine to complete
}
```

---

## **3. Running Multiple Goroutines**
Goroutines execute independently, and you can run many of them concurrently.

### **Example: Multiple Goroutines**
```go
package main
import (
	"fmt"
	"time"
)

func printNumbers() {
	for i := 1; i <= 5; i++ {
		fmt.Println(i)
		time.Sleep(500 * time.Millisecond)
	}
}

func printLetters() {
	for c := 'A'; c <= 'E'; c++ {
		fmt.Println(string(c))
		time.Sleep(500 * time.Millisecond)
	}
}

func main() {
	go printNumbers()
	go printLetters()
	time.Sleep(3 * time.Second) // Wait for goroutines to finish
}
```

### **Output (order may vary)**
```
1
A
2
B
3
C
4
D
5
E
```
🔹 **Explanation:**  
- `printNumbers()` and `printLetters()` run concurrently.  
- The output order is **non-deterministic** due to concurrency.

---

## **4. Synchronizing Goroutines with `sync.WaitGroup`**
Goroutines run asynchronously, and `sync.WaitGroup` ensures all goroutines complete before `main()` exits.

### **Example: Using `sync.WaitGroup`**
```go
package main
import (
	"fmt"
	"sync"
)

func worker(id int, wg *sync.WaitGroup) {
	defer wg.Done() // Decrements the counter when the function completes
	fmt.Printf("Worker %d started\n", id)
}

func main() {
	var wg sync.WaitGroup

	for i := 1; i <= 3; i++ {
		wg.Add(1)       // Increment counter
		go worker(i, &wg)
	}

	wg.Wait() // Wait for all workers to finish
	fmt.Println("All workers completed")
}
```
### **Output**
```
Worker 1 started
Worker 2 started
Worker 3 started
All workers completed
```
🔹 **Key Points:**  
- `wg.Add(1)`: Increments the counter for each goroutine.  
- `wg.Done()`: Decrements the counter when a goroutine finishes.  
- `wg.Wait()`: Blocks execution until the counter reaches zero.

---

## **5. Communicating Between Goroutines with Channels**
Goroutines can communicate using **channels**, which are safe for concurrent use.

### **Example: Sending & Receiving Data with Channels**
```go
package main
import "fmt"

func sendMessage(ch chan string) {
	ch <- "Hello from Goroutine!"
}

func main() {
	ch := make(chan string) // Create a channel

	go sendMessage(ch)

	msg := <-ch // Receive message from channel
	fmt.Println(msg)
}
```
### **Output**
```
Hello from Goroutine!
```
🔹 **Key Points:**  
- `ch := make(chan string)`: Creates a string channel.  
- `ch <- "Hello"`: Sends data into the channel.  
- `msg := <-ch`: Receives data from the channel.

---

## **6. Buffered Channels for Non-blocking Communication**
Buffered channels allow sending multiple messages **without blocking**.

```go
package main
import "fmt"

func main() {
	ch := make(chan int, 3) // Buffer size of 3

	ch <- 1
	ch <- 2
	ch <- 3

	fmt.Println(<-ch)
	fmt.Println(<-ch)
	fmt.Println(<-ch)
}
```
### **Output**
```
1
2
3
```
🔹 **Buffered channels allow sending multiple messages before a receiver reads them.**

---

## **7. Select Statement for Multiple Goroutines**
The `select` statement allows a goroutine to wait on multiple channels.

```go
package main
import (
	"fmt"
	"time"
)

func main() {
	ch1 := make(chan string)
	ch2 := make(chan string)

	go func() {
		time.Sleep(2 * time.Second)
		ch1 <- "Message from Channel 1"
	}()

	go func() {
		time.Sleep(1 * time.Second)
		ch2 <- "Message from Channel 2"
	}()

	select {
	case msg1 := <-ch1:
		fmt.Println(msg1)
	case msg2 := <-ch2:
		fmt.Println(msg2)
	}
}
```
### **Output (Non-deterministic)**
```
Message from Channel 2
```
🔹 **Explanation:**  
- The `select` statement picks the **first available** message.  
- The message from `ch2` is received first because of a shorter delay.

---

## **📌 Use Cases of Goroutines**
### ✅ **1. Web Servers**
Goroutines allow handling thousands of concurrent requests efficiently.
```go
go http.ListenAndServe(":8080", nil)
```

### ✅ **2. Background Tasks**
Processing logs, analytics, or data pipelines asynchronously.
```go
go processLogs()
```

### ✅ **3. Real-time Applications**
Used in chat applications, gaming, and financial services.

### ✅ **4. Concurrent Database Processing**
Fetching multiple data sources in parallel for better performance.

### ✅ **5. Distributed Systems**
Goroutines enable building microservices that efficiently handle concurrent requests.

---

## **🔹 Key Takeaways**
- **Goroutines** allow concurrent execution of functions.
- **`sync.WaitGroup`** ensures all goroutines complete before exiting.
- **Channels** enable safe communication between goroutines.
- **Buffered channels** help prevent blocking.
- **`select` statement** handles multiple goroutines.

Would you like **more advanced examples**, such as **worker pools or parallel processing**?


### **Threads in Golang**  

Go does not expose **raw OS threads** to developers directly. Instead, it provides **goroutines**, which are lightweight **user-space threads** managed by the Go runtime. However, Go still relies on **OS threads** under the hood.

---

## **1. How Go Uses Threads Internally**
Go employs an **M:N scheduling model**, where:
- **M Goroutines** run on **N OS Threads**.
- The Go runtime manages goroutines efficiently, **multiplexing them onto OS threads**.

### **Go's Scheduler Components**
| Component | Description |
|-----------|-------------|
| **G (Goroutine)** | A lightweight function executing concurrently. |
| **M (Machine/Thread)** | OS thread on which goroutines run. |
| **P (Processor)** | Logical processor managing execution of goroutines on threads. |

**Example:** If a Go program has 1000 goroutines but only **4 OS threads**, Go dynamically schedules goroutines on those 4 threads.

---

## **2. Go’s Runtime Scheduler (Goroutines vs Threads)**
- Go does **not** create an OS thread per goroutine.
- The **Go scheduler** assigns goroutines to available OS threads.
- If a goroutine **blocks** (e.g., I/O operation), Go may **move another goroutine** to a free OS thread.

---

## **3. Controlling OS Threads in Go**
Go allows **some control** over OS threads via `runtime` package.

### **Forcing a Function to Run on a Single OS Thread**
Use **`runtime.LockOSThread()`** to bind a goroutine to an OS thread.

```go
package main

import (
	"fmt"
	"runtime"
)

func main() {
	runtime.LockOSThread() // Ensures this goroutine stays on the same OS thread
	defer runtime.UnlockOSThread()

	fmt.Println("This goroutine is locked to an OS thread")
}
```
🔹 Useful for interacting with **C libraries** or **UI frameworks** that require thread affinity.

---

### **4. Checking Number of OS Threads**
```go
package main

import (
	"fmt"
	"runtime"
)

func main() {
	fmt.Println("Total OS Threads:", runtime.GOMAXPROCS(0))
}
```
- `runtime.GOMAXPROCS(n)` sets **the number of OS threads** Go can use.
- Default: **Number of CPU cores**.

---

## **5. When Go Creates OS Threads**
1. **At Program Start** → Go creates a pool of OS threads.  
2. **When Needed** → If goroutines **block on syscalls**, Go may spawn more OS threads.  
3. **Manual Control** → If `runtime.GOMAXPROCS(n)` is set, Go limits OS thread usage.  

---

## **6. Threads vs Goroutines in Go**
| Feature | OS Thread | Goroutine |
|---------|----------|-----------|
| **Creation Cost** | High (system call) | Low (managed by Go runtime) |
| **Memory Usage** | 1MB stack per thread | ~2KB stack per goroutine |
| **Switching Overhead** | Slow (OS context switch) | Fast (user-space scheduling) |
| **Managed By** | OS Kernel | Go Runtime |
| **Parallel Execution** | Yes | Yes (if GOMAXPROCS > 1) |

---

## **7. When to Use Threads in Go**
✅ When calling **C libraries** that require a specific thread.  
✅ When working with **UI frameworks** that require the same thread.  
✅ When handling **low-level OS interactions**.

However, in most cases, **Goroutines** are preferable over raw OS threads for efficiency.

---

## **Conclusion**
- **Go does not expose raw threads**; it provides **goroutines** instead.
- **The Go runtime dynamically manages OS threads** for optimal performance.
- **`runtime.LockOSThread()` can bind a goroutine to a specific OS thread** when necessary.
- **Use goroutines for concurrency, OS threads only when required (e.g., C bindings).**