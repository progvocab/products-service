# **Golang `os` Package – File and System Operations**  

The **`os` package** in Go provides functions for **file handling, environment variables, system interactions, and process control**.

---

## **1️⃣ File Operations (`os.File`)**
### **📌 Creating and Writing to a File**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    file, err := os.Create("example.txt") // Create a new file
    if err != nil {
        fmt.Println("Error creating file:", err)
        return
    }
    defer file.Close()

    file.WriteString("Hello, Go!\n") // Write a string
    fmt.Println("File created successfully!")
}
```
📌 **`os.Create()`** creates a new file, and **`WriteString()`** writes data to it.

---

### **📌 Reading from a File**
```go
package main
import (
    "fmt"
    "os"
    "io"
)

func main() {
    file, err := os.Open("example.txt") // Open file
    if err != nil {
        fmt.Println("Error opening file:", err)
        return
    }
    defer file.Close()

    content, _ := io.ReadAll(file) // Read the entire file
    fmt.Println(string(content)) // Print content
}
```
📌 **`os.Open()`** opens a file for reading, and **`io.ReadAll()`** reads all content.

---

### **📌 Appending to a File**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    file, err := os.OpenFile("example.txt", os.O_APPEND|os.O_WRONLY, 0644)
    if err != nil {
        fmt.Println("Error opening file:", err)
        return
    }
    defer file.Close()

    file.WriteString("Appending text\n")
    fmt.Println("Data appended successfully!")
}
```
📌 **`os.OpenFile()`** opens a file with **append mode**.

---

## **2️⃣ Directory Operations (`os` Functions)**
### **📌 Creating a Directory**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    err := os.Mkdir("new_folder", 0755) // Create directory with permissions
    if err != nil {
        fmt.Println("Error creating directory:", err)
    } else {
        fmt.Println("Directory created successfully!")
    }
}
```
📌 **`os.Mkdir()`** creates a new directory.

---

### **📌 Removing a Directory**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    err := os.Remove("new_folder") // Remove directory or file
    if err != nil {
        fmt.Println("Error removing directory:", err)
    } else {
        fmt.Println("Directory removed successfully!")
    }
}
```
📌 **`os.Remove()`** deletes a file or directory.

---

## **3️⃣ Environment Variables (`os.Getenv` and `os.Setenv`)**
### **📌 Getting an Environment Variable**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    path := os.Getenv("PATH") // Get the PATH environment variable
    fmt.Println("System PATH:", path)
}
```
📌 **`os.Getenv()`** retrieves environment variables.

---

### **📌 Setting an Environment Variable**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    os.Setenv("MY_ENV", "Golang") // Set an environment variable
    fmt.Println("MY_ENV:", os.Getenv("MY_ENV"))
}
```
📌 **`os.Setenv()`** sets an environment variable.

---

## **4️⃣ Process Control (`os.Exit`, `os.Args`, `os.Getpid`)**
### **📌 Getting Command-Line Arguments**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    fmt.Println("Command-line args:", os.Args) // Print arguments
}
```
📌 **`os.Args`** stores command-line arguments.

---

### **📌 Exiting a Program with a Status Code**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    fmt.Println("Exiting program...")
    os.Exit(1) // Exit with status code 1 (error)
}
```
📌 **`os.Exit(1)`** terminates the program immediately.

---

## **5️⃣ Checking File Information (`os.Stat`)**
### **📌 Checking If a File Exists**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    _, err := os.Stat("example.txt")
    if os.IsNotExist(err) {
        fmt.Println("File does not exist!")
    } else {
        fmt.Println("File exists!")
    }
}
```
📌 **`os.Stat()`** checks file existence.

---

## **6️⃣ Summary Table – Important `os` Functions**
| **Function** | **Description** |
|-------------|----------------|
| `os.Create("file.txt")` | Create a file |
| `os.Open("file.txt")` | Open a file for reading |
| `os.Remove("file.txt")` | Delete a file |
| `os.Mkdir("folder", 0755)` | Create a directory |
| `os.Getenv("PATH")` | Get an environment variable |
| `os.Setenv("KEY", "VALUE")` | Set an environment variable |
| `os.Exit(1)` | Exit the program |
| `os.Args` | Get command-line arguments |
| `os.Stat("file.txt")` | Get file information |

Would you like **real-world use cases** for the `os` package? 🚀