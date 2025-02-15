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