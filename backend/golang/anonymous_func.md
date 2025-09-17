### **Lambda Functions in Golang (Anonymous Functions)**  

Yes, Golang **supports lambda functions**, also known as **anonymous functions**. These are **functions without a name**, defined inline, and often used for short-lived operations like **callbacks, closures, or one-time computations**.  

---

## **1️⃣ Basic Anonymous Function**
✅ You can define and call a function **inline**.

```go
package main
import "fmt"

func main() {
    // Define and immediately invoke an anonymous function
    result := func(a, b int) int {
        return a + b
    }(10, 20)

    fmt.Println(result) // Output: 30
}
```

---

## **2️⃣ Assigning Anonymous Function to a Variable**
✅ You can **store** an anonymous function in a variable and call it later.

```go
package main
import "fmt"

func main() {
    multiply := func(x, y int) int {
        return x * y
    }

    fmt.Println(multiply(3, 4)) // Output: 12
}
```

---

## **3️⃣ Passing Anonymous Functions as Arguments**
✅ Used for **callbacks and higher-order functions**.

```go
package main
import "fmt"

// Function accepting another function as argument
func operate(a, b int, op func(int, int) int) int {
    return op(a, b)
}

func main() {
    sum := func(x, y int) int { return x + y }
    diff := func(x, y int) int { return x - y }

    fmt.Println(operate(10, 5, sum))  // Output: 15
    fmt.Println(operate(10, 5, diff)) // Output: 5
}
```

---

## **4️⃣ Closures (Anonymous Functions with Lexical Scope)**
✅ A closure is an **anonymous function** that captures variables **from its surrounding scope**.

```go
package main
import "fmt"

func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

func main() {
    increment := counter()
    fmt.Println(increment()) // Output: 1
    fmt.Println(increment()) // Output: 2
    fmt.Println(increment()) // Output: 3
}
```
📌 **`increment()` remembers the state of `count` across function calls.**

---

## **5️⃣ Using Anonymous Functions in Goroutines**
✅ Used for **concurrent execution**.

```go
package main
import (
    "fmt"
    "time"
)

func main() {
    go func() {
        fmt.Println("Hello from Goroutine!")
    }()
    
    time.Sleep(time.Second) // Wait for Goroutine to finish
}
```
📌 **Anonymous functions are useful for lightweight concurrent tasks**.

---

## **6️⃣ Returning Anonymous Functions**
✅ **Used in functional programming patterns.**

```go
package main
import "fmt"

func multiplier(factor int) func(int) int {
    return func(num int) int {
        return num * factor
    }
}

func main() {
    double := multiplier(2)
    triple := multiplier(3)

    fmt.Println(double(5)) // Output: 10
    fmt.Println(triple(5)) // Output: 15
}
```
📌 **Functions return functions, allowing powerful custom behavior.**

---

### **🔹 When to Use Anonymous Functions in Go?**
✅ **Short-lived computations** (e.g., sorting, filtering)  
✅ **Closures to retain state**  
✅ **Passing functions as arguments** (e.g., callbacks)  
✅ **Concurrent execution** using **Goroutines**  

Would you like to see real-world examples of **Go lambda functions in APIs or concurrency?** 🚀