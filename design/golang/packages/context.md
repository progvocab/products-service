### **`context` Package in Go**
The **`context`** package in Go is used for managing deadlines, cancellations, and passing request-scoped values across API boundaries. It is commonly used in **HTTP servers, database queries, and goroutines** to prevent resource leaks and improve efficiency.

---

## **Why Use `context`?**
- **Handles request timeouts** → Automatically cancels long-running operations.
- **Gracefully stops goroutines** → Prevents memory leaks.
- **Passes request-scoped values** → Allows sharing data across API boundaries.

---

## **Creating and Using a Context**
### **1. Creating a Background Context**
The **root context** is created using `context.Background()`. It's often used as a parent context.

```go
package main

import (
	"context"
	"fmt"
)

func main() {
	ctx := context.Background()
	fmt.Println(ctx) // Output: context.Background
}
```

---

### **2. Creating a Context with Cancellation**
Use `context.WithCancel()` to cancel a context manually.

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(2 * time.Second)
		cancel() // Manually cancel after 2 seconds
	}()

	<-ctx.Done() // Blocks until context is canceled
	fmt.Println("Context canceled:", ctx.Err()) // Output: Context canceled: context canceled
}
```

- `ctx.Done()` returns a **channel** that is closed when the context is canceled.
- `ctx.Err()` gives the error reason (`context.Canceled` or `context.DeadlineExceeded`).

---

### **3. Creating a Context with Timeout**
Use `context.WithTimeout()` to **automatically cancel** after a fixed time.

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel() // Ensure context is canceled when done

	select {
	case <-time.After(5 * time.Second): // Simulating long operation
		fmt.Println("Operation completed")
	case <-ctx.Done():
		fmt.Println("Timeout! Context error:", ctx.Err()) // Output: Timeout! Context error: context deadline exceeded
	}
}
```

- If the operation takes longer than 3 seconds, it gets **canceled automatically**.

---

### **4. Creating a Context with Deadline**
`context.WithDeadline()` sets an **absolute time** for cancellation.

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	deadline := time.Now().Add(2 * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	select {
	case <-time.After(3 * time.Second):
		fmt.Println("Operation completed")
	case <-ctx.Done():
		fmt.Println("Context deadline exceeded:", ctx.Err()) // Output: Context deadline exceeded: context deadline exceeded
	}
}
```

- This behaves like `WithTimeout()`, but with an exact deadline.

---

### **5. Passing Values with Context**
Use `context.WithValue()` to attach values to a context.

```go
package main

import (
	"context"
	"fmt"
)

func main() {
	ctx := context.WithValue(context.Background(), "userID", 42)

	userID := ctx.Value("userID") // Retrieve value from context
	fmt.Println("User ID:", userID) // Output: User ID: 42
}
```

- `WithValue()` is useful for passing **request-scoped data** (e.g., authentication info).
- It should **not** be used for large objects or frequent lookups.

---

## **Using `context` in Real-World Scenarios**
### **1. Using Context in HTTP Handlers**
```go
package main

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

func handler(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	select {
	case <-time.After(3 * time.Second):
		fmt.Fprintln(w, "Request processed")
	case <-ctx.Done():
		http.Error(w, "Request timed out", http.StatusRequestTimeout)
	}
}

func main() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)
}
```
- If the request takes longer than **2 seconds**, it will be **automatically canceled**.

---

### **2. Using Context in Database Queries**
```go
package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	db, err := sql.Open("postgres", "your_connection_string")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	query := "SELECT name FROM users WHERE id=$1"
	row := db.QueryRowContext(ctx, query, 1)

	var name string
	if err := row.Scan(&name); err != nil {
		fmt.Println("Query error:", err)
	} else {
		fmt.Println("User name:", name)
	}
}
```
- The query **cancels automatically** after **2 seconds** if the database is slow.

---

## **Summary of `context` Functions**
| Function | Purpose |
|----------|---------|
| `context.Background()` | Root context, used when no other context is available. |
| `context.TODO()` | Placeholder for future context (use when unsure). |
| `context.WithCancel(parent)` | Creates a context that can be manually canceled. |
| `context.WithTimeout(parent, duration)` | Cancels automatically after a timeout. |
| `context.WithDeadline(parent, time)` | Cancels at a specific point in time. |
| `context.WithValue(parent, key, value)` | Attaches request-scoped data to a context. |

---

## **Best Practices for Using `context`**
✅ Always use `defer cancel()` after creating a context.  
✅ Use **timeouts** to avoid long-running operations.  
✅ Use `WithValue()` **sparingly** (avoid large data or frequent lookups).  
✅ Pass `context.Context` as the **first parameter** in functions.  
✅ Never store `context.Context` inside a struct—pass it explicitly.  

Using `context` properly helps **prevent resource leaks**, **manage timeouts**, and **handle cancellations** in concurrent Go programs.