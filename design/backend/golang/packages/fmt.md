### **Golang `fmt` Package – Formatting and Printing**  

The **`fmt` package** in Golang provides functions for **formatted I/O** (input and output), including **printing, scanning, and string formatting**.

---

## **1️⃣ fmt.Print, fmt.Println, fmt.Printf (Standard Output)**
✅ **Prints output to the console.**  

```go
package main
import "fmt"

func main() {
    name := "Alice"
    age := 30

    fmt.Print("Hello, ")              // No newline
    fmt.Println("World!")              // With newline
    fmt.Println("Name:", name, "Age:", age)

    fmt.Printf("My name is %s and I am %d years old.\n", name, age)
}
```
📌 **`Print`** (no newline), **`Println`** (newline), **`Printf`** (formatted).  

---

## **2️⃣ fmt.Sprintf (String Formatting)**
✅ **Formats a string and returns it (without printing).**  

```go
package main
import "fmt"

func main() {
    name := "Alice"
    age := 30
    formatted := fmt.Sprintf("Name: %s, Age: %d", name, age)
    
    fmt.Println(formatted) // Output: Name: Alice, Age: 30
}
```
📌 **Use `fmt.Sprintf()`** when you need a formatted string **without printing it**.

---

## **3️⃣ fmt.Scan, fmt.Scanf, fmt.Scanln (User Input)**
✅ **Takes input from the user.**  

```go
package main
import "fmt"

func main() {
    var name string
    var age int

    fmt.Print("Enter name: ")
    fmt.Scan(&name) // Reads input until space or newline
    
    fmt.Print("Enter age: ")
    fmt.Scanf("%d", &age) // Reads formatted input

    fmt.Println("User:", name, "Age:", age)
}
```
📌 **`Scan()`** reads space-separated values, **`Scanf()`** reads formatted input, **`Scanln()`** reads until a newline.

---

## **4️⃣ Formatting Verbs in fmt.Printf**
✅ **Placeholders for different data types.**  

| Verb  | Meaning | Example |
|-------|---------|---------|
| `%v`  | Default value | `fmt.Printf("%v", 42)` → `42` |
| `%T`  | Type of variable | `fmt.Printf("%T", 42)` → `int` |
| `%d`  | Integer (base 10) | `fmt.Printf("%d", 42)` → `42` |
| `%b`  | Binary format | `fmt.Printf("%b", 5)` → `101` |
| `%x`  | Hexadecimal | `fmt.Printf("%x", 255)` → `ff` |
| `%f`  | Floating point | `fmt.Printf("%.2f", 3.14159)` → `3.14` |
| `%s`  | String | `fmt.Printf("%s", "hello")` → `hello` |
| `%q`  | Quoted string | `fmt.Printf("%q", "hello")` → `"hello"` |
| `%t`  | Boolean | `fmt.Printf("%t", true)` → `true` |
| `%p`  | Pointer address | `fmt.Printf("%p", &x)` → `0xc0000100f0` |

Example:
```go
package main
import "fmt"

func main() {
    x := 42
    pi := 3.14159
    str := "hello"
    
    fmt.Printf("Integer: %d, Binary: %b, Hex: %x\n", x, x, x)
    fmt.Printf("Float: %.2f\n", pi)
    fmt.Printf("String: %s, Quoted: %q\n", str, str)
}
```
📌 **Different format specifiers control how data is printed.**

---

## **5️⃣ fmt.Errorf (Custom Errors)**
✅ **Creates formatted error messages.**  

```go
package main
import (
    "fmt"
    "errors"
)

func validate(age int) error {
    if age < 18 {
        return fmt.Errorf("age %d is too young to register", age)
    }
    return nil
}

func main() {
    err := validate(16)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Registration successful")
    }
}
```
📌 **Use `fmt.Errorf()`** to format errors dynamically.

---

## **6️⃣ fmt.Fprint, fmt.Fprintln, fmt.Fprintf (Writing to Any Writer)**
✅ **Writes to files, network connections, or other destinations.**  

```go
package main
import (
    "fmt"
    "os"
)

func main() {
    file, _ := os.Create("output.txt")
    defer file.Close()

    fmt.Fprintln(file, "Hello, File!") // Writes to a file

    fmt.Println("Data written to file")
}
```
📌 **`Fprint()`** writes to any `io.Writer` (file, buffer, network, etc.).

---

### **🔹 Summary Table**
| `fmt` Function | Purpose |
|---------------|------------------------------------------------|
| `Print`, `Println`, `Printf` | Print to console with different formatting. |
| `Sprintf` | Format and return a string (without printing). |
| `Scan`, `Scanf`, `Scanln` | Read user input from the console. |
| `Errorf` | Create custom formatted errors. |
| `Fprint`, `Fprintln`, `Fprintf` | Write formatted output to files or other writers. |

Would you like **more practical examples** of any specific function? 🚀