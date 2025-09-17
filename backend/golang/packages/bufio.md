# **Golang `bufio` Package – Buffered I/O in Go**  

The **`bufio` package** in Go provides efficient buffered **reading and writing** operations. It is useful when dealing with large files, input streams, or network connections to **reduce the number of system calls** and improve performance.

---

## **1️⃣ bufio.Reader (Buffered Input)**
✅ **Reading text efficiently from standard input (`os.Stdin`) or a file.**

### **📌 Example: Reading Input from the Console**
```go
package main
import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    reader := bufio.NewReader(os.Stdin) // Create a buffered reader
    fmt.Print("Enter your name: ")
    name, _ := reader.ReadString('\n')  // Read until newline
    fmt.Println("Hello,", name)
}
```
📌 **`ReadString('\n')`** reads input **until a newline character**.

---

### **📌 Example: Reading from a File**
✅ **Read a file line-by-line using `bufio.Reader`.**
```go
package main
import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    file, err := os.Open("sample.txt")
    if err != nil {
        fmt.Println("Error opening file:", err)
        return
    }
    defer file.Close()

    reader := bufio.NewReader(file)
    for {
        line, err := reader.ReadString('\n')
        if err != nil {
            break
        }
        fmt.Print(line) // Print each line
    }
}
```
📌 **`ReadString('\n')`** reads a file **line-by-line**.

---

## **2️⃣ bufio.Scanner (Simple Line-by-Line Input)**
✅ **The `Scanner` is ideal for reading lines from standard input or files.**

### **📌 Example: Reading Console Input (Multiple Lines)**
```go
package main
import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    fmt.Println("Enter multiple lines (Ctrl+D to stop):")

    for scanner.Scan() { // Reads each line until EOF
        text := scanner.Text() // Get the scanned text
        fmt.Println("You entered:", text)
    }
}
```
📌 **`scanner.Text()`** retrieves the text from each line.

---

### **📌 Example: Reading a File Line-by-Line (Efficiently)**
```go
package main
import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    file, _ := os.Open("sample.txt")
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() { // Reads each line
        fmt.Println(scanner.Text())
    }
}
```
📌 **Using `scanner.Scan()` for line-by-line file reading is more memory efficient.**

---

## **3️⃣ bufio.Writer (Buffered Output)**
✅ **Buffered writing improves performance by reducing write system calls.**

### **📌 Example: Writing to a File Using `bufio.Writer`**
```go
package main
import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    file, _ := os.Create("output.txt") // Create a file
    defer file.Close()

    writer := bufio.NewWriter(file)
    writer.WriteString("Hello, Buffered Writer!\n") // Write a string
    writer.Flush() // Ensure all data is written to the file

    fmt.Println("Data written to file successfully")
}
```
📌 **Use `Flush()`** to ensure buffered data is written to the file.

---

## **4️⃣ Comparing bufio.Reader vs bufio.Scanner vs bufio.Writer**
| **Function** | **Use Case** | **Best For** |
|-------------|-------------|--------------|
| `bufio.NewReader()` | Read data **with custom delimiters** | **Efficient file reading** |
| `bufio.NewScanner()` | Read **line-by-line** | **User input or text processing** |
| `bufio.NewWriter()` | Write data **efficiently** | **Buffered writing to files** |

Would you like **real-world examples** using `bufio`? 🚀