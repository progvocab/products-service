## **Channels in Golang**
Channels in Golang provide a way for goroutines to communicate safely and synchronize their execution. They help avoid **race conditions** by enabling **safe data exchange** between concurrent goroutines.

---

## **1️⃣ Creating a Channel**
A channel in Go is declared using the `chan` keyword.

```go
package main
import "fmt"

func main() {
	ch := make(chan int) // Creating an integer channel
	go func() {
		ch <- 42 // Sending data to the channel
	}()
	fmt.Println(<-ch) // Receiving data from the channel
}
```

### **Output**
```
42
```
---

## **2️⃣ Buffered Channels**
A buffered channel allows sending multiple values without requiring an immediate receiver.

```go
package main
import "fmt"

func main() {
	ch := make(chan string, 2) // Buffered channel with capacity 2
	ch <- "Hello"
	ch <- "World"
	fmt.Println(<-ch) // "Hello"
	fmt.Println(<-ch) // "World"
}
```

### **Key Points**
- A **buffered channel** does not block the sender **until** the buffer is full.
- The receiver blocks if the buffer is **empty**.

---

## **3️⃣ Closing a Channel**
Channels can be **closed** using `close(ch)`. Receivers can check if a channel is closed using the **comma-ok** idiom.

```go
package main
import "fmt"

func main() {
	ch := make(chan int, 3)
	ch <- 10
	ch <- 20
	close(ch) // Closing the channel

	for v := range ch { // Range over channel
		fmt.Println(v)
	}
}
```

### **Output**
```
10
20
```

### **Key Points**
- **Closing a channel** signals that no more values will be sent.
- **Reading from a closed channel** returns the zero value of its type.

---

## **4️⃣ Unidirectional Channels**
You can restrict channel direction:
- **Send-only** (`chan<- int`)
- **Receive-only** (`<-chan int`)

```go
package main
import "fmt"

func sendData(ch chan<- int) { // Send-only channel
	ch <- 100
}

func receiveData(ch <-chan int) { // Receive-only channel
	fmt.Println(<-ch)
}

func main() {
	ch := make(chan int)
	go sendData(ch)
	receiveData(ch)
}
```

---

## **5️⃣ Select Statement (Multiple Channels)**
The `select` statement allows handling multiple channels.

```go
package main
import (
	"fmt"
	"time"
)

func main() {
	ch1, ch2 := make(chan string), make(chan string)

	go func() {
		time.Sleep(2 * time.Second)
		ch1 <- "Message from ch1"
	}()

	go func() {
		time.Sleep(1 * time.Second)
		ch2 <- "Message from ch2"
	}()

	select {
	case msg1 := <-ch1:
		fmt.Println(msg1)
	case msg2 := <-ch2:
		fmt.Println(msg2) // This executes first because it has a smaller sleep time
	}
}
```

---

## **6️⃣ Worker Pool Using Channels**
A **worker pool** efficiently processes tasks concurrently.

```go
package main
import (
	"fmt"
	"time"
)

func worker(id int, jobs <-chan int, results chan<- int) {
	for job := range jobs {
		fmt.Printf("Worker %d processing job %d\n", id, job)
		time.Sleep(time.Second)
		results <- job * 2
	}
}

func main() {
	jobs := make(chan int, 5)
	results := make(chan int, 5)

	// Start 3 workers
	for w := 1; w <= 3; w++ {
		go worker(w, jobs, results)
	}

	// Send jobs
	for j := 1; j <= 5; j++ {
		jobs <- j
	}
	close(jobs)

	// Collect results
	for a := 1; a <= 5; a++ {
		fmt.Println("Result:", <-results)
	}
}
```

---

## **7️⃣ Deadlock Example**
A **deadlock** occurs when a goroutine waits on a channel **indefinitely**.

```go
package main
func main() {
	ch := make(chan int)
	ch <- 10  // Deadlock! No receiver
}
```

### **Fix**
Use a **goroutine** to receive:

```go
package main
import "fmt"

func main() {
	ch := make(chan int)
	go func() {
		fmt.Println(<-ch) // Receiver
	}()
	ch <- 10
}
```

---

## **8️⃣ Using `sync.WaitGroup` with Channels**
To wait for multiple goroutines to complete:

```go
package main
import (
	"fmt"
	"sync"
)

func worker(id int, wg *sync.WaitGroup) {
	defer wg.Done()
	fmt.Printf("Worker %d done\n", id)
}

func main() {
	var wg sync.WaitGroup

	for i := 1; i <= 3; i++ {
		wg.Add(1)
		go worker(i, &wg)
	}
	wg.Wait() // Wait for all workers
}
```

---

### **🔹 Summary**
| Concept                | Description |
|------------------------|------------|
| **Unbuffered Channel**  | Blocks sender until a receiver is available |
| **Buffered Channel**    | Allows sending multiple values before blocking |
| **Closing a Channel**   | Stops sending values, receivers get zero values |
| **Unidirectional Channel** | Restricts sending (`chan<-`) or receiving (`<-chan`) |
| **Select Statement**    | Handles multiple channels at once |
| **Worker Pool**         | Uses channels to manage concurrent tasks |
| **Deadlock**            | Occurs when a goroutine waits forever on a channel |

Would you like a **real-world example** for a specific use case?