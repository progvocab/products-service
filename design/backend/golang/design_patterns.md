### **Design Patterns in Go (Golang)**
Design patterns are reusable solutions to common software design problems. In **Go (Golang)**, design patterns are adapted to its unique features, such as **interfaces, concurrency (goroutines, channels), and composition over inheritance**.

---

## **1. Creational Patterns (Object Creation)**
| Pattern | Description | Example |
|---------|------------|---------|
| **Singleton** | Ensures only **one instance** of a struct exists. | Database connection pool |
| **Factory** | Provides a way to create objects **without specifying exact struct types**. | Creating different shapes in a graphics app |
| **Builder** | Simplifies complex object creation step-by-step. | Configuring HTTP requests |
| **Prototype** | Copies an existing object **instead of creating a new one**. | Cloning configurations |

### **Example: Singleton Pattern**
```go
package main

import (
	"fmt"
	"sync"
)

type singleton struct{}

var instance *singleton
var once sync.Once

func GetInstance() *singleton {
	once.Do(func() {
		instance = &singleton{}
	})
	return instance
}

func main() {
	s1 := GetInstance()
	s2 := GetInstance()

	fmt.Println(s1 == s2) // Output: true (same instance)
}
```

---

## **2. Structural Patterns (Class and Object Composition)**
| Pattern | Description | Example |
|---------|------------|---------|
| **Adapter** | Converts one interface to another **without changing the original code**. | Translating API responses |
| **Bridge** | Decouples an abstraction from its implementation **to allow variations**. | Rendering engines (SVG, PNG) |
| **Decorator** | Dynamically adds behavior to an object **without modifying its structure**. | Logging middleware in HTTP handlers |
| **Facade** | Provides a **simplified interface** to a complex subsystem. | Wrapping database interactions |
| **Proxy** | Controls access to an object **(e.g., caching, authentication, lazy loading)**. | Rate-limiting API calls |

### **Example: Decorator Pattern (Middleware in HTTP Server)**
```go
package main

import (
	"fmt"
	"net/http"
)

// Decorator function
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println("Request received:", r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func main() {
	mux := http.NewServeMux()
	mux.Handle("/", loggingMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Hello, world!"))
	})))

	http.ListenAndServe(":8080", mux)
}
```

---

## **3. Behavioral Patterns (Object Communication)**
| Pattern | Description | Example |
|---------|------------|---------|
| **Observer** | Notifies multiple objects about **state changes**. | Event-driven systems, pub-sub |
| **Strategy** | Defines a family of algorithms **that can be swapped at runtime**. | Sorting strategies |
| **Command** | Encapsulates a request as an object. | Undo/redo functionality |
| **Chain of Responsibility** | Passes requests **through handlers** until one processes it. | HTTP request middleware |
| **State** | Allows an object to change its behavior **when its state changes**. | Traffic light system |

### **Example: Strategy Pattern**
```go
package main

import "fmt"

// Strategy Interface
type PaymentStrategy interface {
	Pay(amount float64)
}

// Concrete Strategies
type CreditCard struct{}

func (c *CreditCard) Pay(amount float64) {
	fmt.Println("Paid", amount, "using Credit Card")
}

type PayPal struct{}

func (p *PayPal) Pay(amount float64) {
	fmt.Println("Paid", amount, "using PayPal")
}

// Context
type PaymentProcessor struct {
	strategy PaymentStrategy
}

func (p *PaymentProcessor) SetStrategy(strategy PaymentStrategy) {
	p.strategy = strategy
}

func (p *PaymentProcessor) ProcessPayment(amount float64) {
	p.strategy.Pay(amount)
}

func main() {
	payment := &PaymentProcessor{}

	// Using Credit Card
	payment.SetStrategy(&CreditCard{})
	payment.ProcessPayment(100)

	// Switching to PayPal
	payment.SetStrategy(&PayPal{})
	payment.ProcessPayment(200)
}
```

---

## **4. Concurrency Patterns (Go-Specific)**
| Pattern | Description | Example |
|---------|------------|---------|
| **Worker Pool** | Uses multiple goroutines to process tasks in parallel. | Background jobs, task processing |
| **Publisher-Subscriber** | Objects communicate via **channels**. | Event-driven architecture |
| **Fan-Out/Fan-In** | Multiple producers generate data (**fan-out**) and multiple consumers process it (**fan-in**). | Data streaming |

### **Example: Worker Pool Pattern**
```go
package main

import (
	"fmt"
	"time"
)

// Worker function
func worker(id int, jobs <-chan int, results chan<- int) {
	for job := range jobs {
		fmt.Printf("Worker %d processing job %d\n", id, job)
		time.Sleep(time.Second) // Simulate work
		results <- job * 2
	}
}

func main() {
	jobs := make(chan int, 5)
	results := make(chan int, 5)

	// Start 3 worker goroutines
	for i := 1; i <= 3; i++ {
		go worker(i, jobs, results)
	}

	// Send jobs
	for j := 1; j <= 5; j++ {
		jobs <- j
	}
	close(jobs) // Close jobs channel

	// Collect results
	for r := 1; r <= 5; r++ {
		fmt.Println("Result:", <-results)
	}
}
```

---

## **Key Takeaways**
- **Creational Patterns** → Manage object creation (`Singleton`, `Factory`, `Builder`).
- **Structural Patterns** → Organize object composition (`Decorator`, `Adapter`, `Proxy`).
- **Behavioral Patterns** → Handle communication (`Observer`, `Strategy`, `Command`).
- **Concurrency Patterns** → Go-specific designs (`Worker Pool`, `Pub-Sub`, `Fan-Out`).

Would you like an example of a **specific pattern in a real-world scenario**?