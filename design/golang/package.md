### **Packages in Golang**  

In Go, a **package** is a collection of related Go files grouped together. Packages help organize code and enable **modularity and reusability**.  

---

## **1. What Can a Package Contain?**  
A Go package can contain:  
✅ **Functions** (e.g., `func Add(a, b int) int`)  
✅ **Variables** (e.g., `var Pi = 3.14`)  
✅ **Constants** (e.g., `const Version = "1.0"`)  
✅ **Structs** (e.g., `type Person struct { Name string }`)  
✅ **Interfaces** (e.g., `type Reader interface { Read() }`)  
✅ **Nested packages** (other packages inside a module)  

---

## **2. Creating and Using Custom Packages**  

### **Step 1: Create a Custom Package (`mathutil`)**
Inside your Go project, create a folder **`mathutil`** and a file **`mathutil.go`**:

```go
// mathutil/mathutil.go
package mathutil

// Add function (exported)
func Add(a, b int) int {
    return a + b
}

// subtract function (unexported)
func subtract(a, b int) int {
    return a - b
}
```
✅ **Exported (`Add`)** → Starts with a capital letter and is accessible outside the package.  
❌ **Unexported (`subtract`)** → Starts with a lowercase letter and is private to the package.  

---

### **Step 2: Use the Custom Package in `main.go`**  
Now, create a `main.go` file in the root folder:

```go
package main

import (
    "fmt"
    "myproject/mathutil" // Import custom package
)

func main() {
    sum := mathutil.Add(10, 5)
    fmt.Println("Sum:", sum) // Output: Sum: 15
}
```

---

## **3. How to Reuse Packages?**  

### **a) Reusing Standard Library Packages**  
Go provides built-in packages like `fmt`, `strings`, `math`, etc.  
Example using `math` package:
```go
import (
    "fmt"
    "math"
)

func main() {
    fmt.Println(math.Sqrt(16)) // Output: 4
}
```

### **b) Reusing Custom Packages in Other Projects**  
To reuse a package in different projects:  
1. Store it in a Git repository (e.g., GitHub).  
2. Use `go mod init <module-name>` to create a module.  
3. Import it using `go get <repo-url>`.  

Example (importing from GitHub):
```go
import "github.com/user/mathutil"
```

---

## **4. Package Initialization (`init` Function)**
A package can contain an **`init()`** function that runs automatically when imported.

```go
package mathutil

import "fmt"

func init() {
    fmt.Println("mathutil package initialized!")
}
```
Whenever `mathutil` is imported, the message **"mathutil package initialized!"** is printed.

---

## **5. Go Module System (`go.mod`)**
To manage dependencies, create a module:
```sh
go mod init myproject
```
This generates a `go.mod` file:
```
module myproject

go 1.20
```

---

### **Key Takeaways**
✅ **Go packages modularize code for reuse.**  
✅ **Exported identifiers** start with a capital letter.  
✅ **Use `go mod` for dependency management.**  
✅ **The `init()` function runs when a package is imported.**  

Would you like an example of **creating a package with interfaces?**