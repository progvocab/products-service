### **Keywords in Go (Golang)**  
Go has **25 reserved keywords** that define the syntax and structure of the language. These keywords **cannot** be used as variable names, function names, or identifiers.

---

## **1. Control Flow Keywords**  
Used for loops, conditions, and decision-making.

| Keyword | Description | Example |
|---------|------------|---------|
| `if` | Conditional execution | `if x > 10 { fmt.Println("x is greater than 10") }` |
| `else` | Alternative branch for `if` | `if x > 10 { ... } else { ... }` |
| `switch` | Multi-way branching | `switch x { case 1: ... case 2: ... }` |
| `case` | Defines cases inside `switch` | `case "A": fmt.Println("A selected")` |
| `default` | Fallback case for `switch` | `default: fmt.Println("No match")` |
| `for` | Loop construct | `for i := 0; i < 10; i++ { fmt.Println(i) }` |
| `range` | Iterates over arrays, slices, maps, channels | `for i, v := range arr { fmt.Println(i, v) }` |
| `return` | Returns a value from a function | `return x + y` |
| `goto` | Unconditionally jumps to a labeled statement | `goto End` (not recommended) |
| `fallthrough` | Forces execution of the next `case` in `switch` | `case "A": fmt.Println("A"); fallthrough` |
| `defer` | Delays function execution until surrounding function exits | `defer fmt.Println("Executed last")` |

---

## **2. Variable and Function Declaration Keywords**  
Used for defining variables, constants, and functions.

| Keyword | Description | Example |
|---------|------------|---------|
| `var` | Declares a variable | `var x int = 10` |
| `const` | Declares a constant | `const Pi = 3.14` |
| `func` | Defines a function | `func add(a, b int) int { return a + b }` |
| `type` | Declares a new type | `type Person struct { Name string }` |
| `struct` | Defines a structure | `struct { Name string; Age int }` |
| `interface` | Defines an interface | `type Shape interface { Area() float64 }` |

---

## **3. Concurrency Keywords**  
Used for handling goroutines and synchronization.

| Keyword | Description | Example |
|---------|------------|---------|
| `go` | Starts a new goroutine | `go myFunction()` |
| `chan` | Declares a channel for goroutine communication | `ch := make(chan int)` |
| `select` | Works like `switch` but for channels | `select { case msg := <-ch: fmt.Println(msg) }` |

---

## **4. Package Management Keywords**  
Used for package organization.

| Keyword | Description | Example |
|---------|------------|---------|
| `package` | Declares the package of a file | `package main` |
| `import` | Imports packages | `import "fmt"` |

---

## **5. Special Keywords**  
Other important Go keywords.

| Keyword | Description | Example |
|---------|------------|---------|
| `break` | Exits a loop or switch | `break` |
| `continue` | Skips to the next iteration of a loop | `continue` |
| `map` | Defines a map (key-value store) | `m := map[string]int{"A": 1, "B": 2}` |

---

### **Full List of Go Keywords**
```
break      case       chan       const     continue  
default    defer      else       fallthrough  for  
func       go        goto       if        import  
interface  map       package    range     return  
select     struct    switch     type      var  
```

Would you like examples for **specific keywords** in real-world scenarios?