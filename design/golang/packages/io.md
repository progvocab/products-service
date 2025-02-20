# **Golang `io` Package – Input/Output Operations**  

The **`io` package** in Go provides **interfaces and utilities** for reading, writing, copying, and managing input/output streams. It serves as the foundation for file handling, network communication, and buffered operations.

---

## **1️⃣ Key Interfaces in `io` Package**  

| **Interface**   | **Description** |
|----------------|---------------|
| `io.Reader`    | Reads data from a source (e.g., file, network) |
| `io.Writer`    | Writes data to a destination |
| `io.Closer`    | Closes a resource (like a file or network connection) |
| `io.Seeker`    | Seeks to a position in a file |
| `io.ReaderFrom` | Reads from another `io.Reader` |
| `io.WriterTo`   | Writes to another `io.Writer` |

---

## **2️⃣ `io.Reader` – Reading Data**
✅ `io.Reader` is an interface that wraps the `Read` method.

### **📌 Example: Reading from a File**
```go
package main
import (
    "fmt"
    "io"
    "os"
)

func main() {
    file, err := os.Open("example.txt")
    if err != nil {
        fmt.Println("Error opening file:", err)
        return
    }
    defer file.Close()

    buf := make([]byte, 100) // Buffer to store read data
    n, err := file.Read(buf) // Read into buffer
    if err != nil && err != io.EOF {
        fmt.Println("Error reading file:", err)
    }

    fmt.Println(string(buf[:n])) // Print the read data
}
```
📌 **`Read(buf []byte)`** reads data into the buffer and returns the number of bytes read.

---

## **3️⃣ `io.Writer` – Writing Data**
✅ `io.Writer` is an interface that wraps the `Write` method.

### **📌 Example: Writing to a File**
```go
package main
import (
    "fmt"
    "io"
    "os"
)

func main() {
    file, err := os.Create("output.txt")
    if err != nil {
        fmt.Println("Error creating file:", err)
        return
    }
    defer file.Close()

    writer := io.Writer(file)
    writer.Write([]byte("Hello, Golang!\n")) // Write data

    fmt.Println("Data written to file!")
}
```
📌 **`Write([]byte)`** writes data to the destination.

---

## **4️⃣ `io.Copy()` – Copying Data Between `Reader` and `Writer`**
✅ `io.Copy()` is used to transfer data between an `io.Reader` and an `io.Writer`.

### **📌 Example: Copying File Contents**
```go
package main
import (
    "fmt"
    "io"
    "os"
)

func main() {
    srcFile, _ := os.Open("source.txt")
    dstFile, _ := os.Create("destination.txt")
    defer srcFile.Close()
    defer dstFile.Close()

    bytesCopied, _ := io.Copy(dstFile, srcFile) // Copy data
    fmt.Println("Bytes copied:", bytesCopied)
}
```
📌 **`io.Copy(dst, src)`** efficiently copies all data from `src` to `dst`.

---

## **5️⃣ `io.TeeReader` – Reading and Writing Simultaneously**
✅ `io.TeeReader()` reads data and sends a copy to a writer.

### **📌 Example: Logging While Reading**
```go
package main
import (
    "fmt"
    "io"
    "os"
)

func main() {
    file, _ := os.Open("example.txt")
    defer file.Close()

    logFile, _ := os.Create("log.txt") // Log the read data
    defer logFile.Close()

    reader := io.TeeReader(file, logFile)
    buf := make([]byte, 100)
    reader.Read(buf) // Read and log simultaneously

    fmt.Println(string(buf))
}
```
📌 **Useful for logging input while processing it**.

---

## **6️⃣ `io.Pipe()` – In-Memory Stream Between Goroutines**
✅ `io.Pipe()` creates an in-memory connection between a reader and a writer.

### **📌 Example: Streaming Data Between Goroutines**
```go
package main
import (
    "fmt"
    "io"
    "strings"
)

func main() {
    reader, writer := io.Pipe()

    go func() {
        writer.Write([]byte("Hello from writer!"))
        writer.Close() // Close after writing
    }()

    buf := make([]byte, 50)
    n, _ := reader.Read(buf)
    fmt.Println(string(buf[:n])) // Output: Hello from writer!
}
```
📌 **`io.Pipe()` allows real-time data transfer between goroutines.**

---

## **7️⃣ `io.Seeker` – Seeking in a File**
✅ `io.Seeker` allows moving to a specific position in a file.

### **📌 Example: Seeking in a File**
```go
package main
import (
    "fmt"
    "os"
)

func main() {
    file, _ := os.Open("example.txt")
    defer file.Close()

    file.Seek(5, 0) // Move to 5th byte
    buf := make([]byte, 10)
    file.Read(buf) // Read next 10 bytes

    fmt.Println(string(buf)) // Print content from 5th byte
}
```
📌 **`Seek(offset, whence)`** moves to a specific position in the file.

---

## **8️⃣ Summary Table – Key Functions in `io` Package**
| **Function** | **Description** |
|-------------|----------------|
| `io.ReadFull(r, buf)` | Reads exactly `len(buf)` bytes into `buf` |
| `io.Copy(dst, src)` | Copies data from `src` to `dst` |
| `io.TeeReader(r, w)` | Reads data from `r` and writes it to `w` simultaneously |
| `io.Pipe()` | Creates a reader-writer in-memory stream |
| `io.Seek(offset, whence)` | Moves to a specific position in a file |

Would you like **real-world applications** using `io`? 🚀