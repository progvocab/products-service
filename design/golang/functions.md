# **Functions in Golang**
Functions in Golang are reusable blocks of code that perform a specific task. They improve code modularity and reusability. Go supports various types of functions, each serving different purposes.

---

## **1️⃣ Basic Function**
A simple function that takes parameters and returns a value.

### **Syntax**
```go
func functionName(param1 type, param2 type) returnType {
    // Function body
    return value
}
```

### **Example**
```go
package main
import "fmt"

func add(a int, b int) int {
    return a + b
}

func main() {
    result := add(5, 3)
    fmt.Println("Sum:", result)  // Output: Sum: 8
}
```

---

## **2️⃣ Function with Multiple Return Values**
Golang allows returning multiple values.

### **Example**
```go
package main
import "fmt"

func divide(a, b int) (int, int) {
    quotient := a / b
    remainder := a % b
    return quotient, remainder
}

func main() {
    q, r := divide(10, 3)
    fmt.Println("Quotient:", q, "Remainder:", r)  // Output: Quotient: 3 Remainder: 1
}
```

---

## **3️⃣ Named Return Values**
Named return values allow returning values without explicitly specifying them in the `return` statement.

### **Example**
```go
package main
import "fmt"

func rectangleDimensions(length, width int) (area int, perimeter int) {
    area = length * width
    perimeter = 2 * (length + width)
    return  // No need to explicitly return area, perimeter
}

func main() {
    a, p := rectangleDimensions(5, 3)
    fmt.Println("Area:", a, "Perimeter:", p)  // Output: Area: 15 Perimeter: 16
}
```

---

## **4️⃣ Variadic Functions**
A function that accepts a variable number of arguments.

### **Example**
```go
package main
import "fmt"

func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

func main() {
    fmt.Println(sum(1, 2, 3, 4, 5))  // Output: 15
}
```

---

## **5️⃣ Anonymous Functions**
Functions without a name, often used for short-lived tasks.

### **Example**
```go
package main
import "fmt"

func main() {
    add := func(a, b int) int {  // Anonymous function assigned to a variable
        return a + b
    }

    fmt.Println(add(5, 3))  // Output: 8
}
```

---

## **6️⃣ Higher-Order Functions (Function as Argument)**
Functions can accept other functions as arguments.

### **Example**
```go
package main
import "fmt"

func applyOperation(a, b int, operation func(int, int) int) int {
    return operation(a, b)
}

func main() {
    multiply := func(x, y int) int { return x * y }
    result := applyOperation(4, 5, multiply)
    fmt.Println(result)  // Output: 20
}
```

---

## **7️⃣ Defer in Functions**
`defer` postpones function execution until the surrounding function exits.

### **Example**
```go
package main
import "fmt"

func example() {
    defer fmt.Println("Deferred Execution")  // Executes at the end
    fmt.Println("Normal Execution")
}

func main() {
    example()
}
```
**Output:**
```
Normal Execution
Deferred Execution
```

---

## **8️⃣ Recursion in Functions**
A function calling itself.

### **Example: Factorial Calculation**
```go
package main
import "fmt"

func factorial(n int) int {
    if n == 0 {
        return 1
    }
    return n * factorial(n-1)
}

func main() {
    fmt.Println(factorial(5))  // Output: 120
}
```

---

## **9️⃣ Methods (Functions with Structs)**
Methods are functions tied to a struct type.

### **Example**
```go
package main
import "fmt"

type Rectangle struct {
    length, width int
}

func (r Rectangle) area() int {  // Method on struct
    return r.length * r.width
}

func main() {
    rect := Rectangle{length: 5, width: 3}
    fmt.Println("Area:", rect.area())  // Output: Area: 15
}
```

---

## **🔟 Function Pointers**
Functions can be assigned to variables and passed around.

### **Example**
```go
package main
import "fmt"

func greet(name string) {
    fmt.Println("Hello,", name)
}

func main() {
    var sayHello func(string) = greet
    sayHello("Alice")  // Output: Hello, Alice
}
```

---

## **Summary of Function Types in Golang**
| **Function Type**  | **Description** |
|------------------|----------------|
| **Basic Function** | Regular function with parameters & return value |
| **Multiple Return Values** | Returns multiple values |
| **Named Return Values** | Named variables in return statement |
| **Variadic Function** | Accepts variable number of arguments |
| **Anonymous Function** | Function without a name |
| **Higher-Order Function** | Function that takes another function as argument |
| **Defer in Functions** | Postpones execution until function exit |
| **Recursive Function** | Calls itself |
| **Methods** | Functions associated with a struct |
| **Function Pointers** | Assign functions to variables |

Would you like more details or a specific example? 🚀