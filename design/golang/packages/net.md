## **Golang `net` Package – Networking in Go**  

The **`net` package** in Golang provides networking functionality, including **TCP, UDP, and HTTP communication**.

---

## **1️⃣ TCP Server and Client**
### **📌 Creating a TCP Server**
✅ A **TCP server** listens on a port and accepts incoming client connections.

```go
package main
import (
    "fmt"
    "net"
)

func main() {
    listener, err := net.Listen("tcp", ":8080") // Listen on port 8080
    if err != nil {
        fmt.Println("Error starting server:", err)
        return
    }
    defer listener.Close()

    fmt.Println("Server is listening on port 8080...")
    
    for {
        conn, err := listener.Accept() // Accept client connection
        if err != nil {
            fmt.Println("Error accepting connection:", err)
            continue
        }
        go handleConnection(conn) // Handle each client in a Goroutine
    }
}

func handleConnection(conn net.Conn) {
    defer conn.Close()
    conn.Write([]byte("Hello, Client!\n")) // Send response
}
```
📌 The **server** listens on port `8080` and sends `"Hello, Client!"` when a connection is made.

---

### **📌 Creating a TCP Client**
✅ A **TCP client** connects to a server and reads the response.

```go
package main
import (
    "fmt"
    "net"
    "bufio"
)

func main() {
    conn, err := net.Dial("tcp", "localhost:8080") // Connect to server
    if err != nil {
        fmt.Println("Error connecting:", err)
        return
    }
    defer conn.Close()

    message, _ := bufio.NewReader(conn).ReadString('\n') // Read response
    fmt.Println("Server response:", message)
}
```
📌 The **client** connects to `localhost:8080` and prints the response from the server.

---

## **2️⃣ UDP Server and Client**
### **📌 Creating a UDP Server**
✅ A **UDP server** listens for incoming datagrams.

```go
package main
import (
    "fmt"
    "net"
)

func main() {
    addr, _ := net.ResolveUDPAddr("udp", ":8081") // UDP address
    conn, _ := net.ListenUDP("udp", addr)         // Listen for UDP
    defer conn.Close()

    fmt.Println("UDP server listening on port 8081...")

    buffer := make([]byte, 1024)
    for {
        n, clientAddr, _ := conn.ReadFromUDP(buffer) // Read data
        fmt.Printf("Received: %s from %s\n", string(buffer[:n]), clientAddr)
        conn.WriteToUDP([]byte("Message received"), clientAddr) // Send response
    }
}
```
📌 The **UDP server** listens on port `8081` and sends `"Message received"` when it gets a datagram.

---

### **📌 Creating a UDP Client**
✅ A **UDP client** sends a message to the server.

```go
package main
import (
    "fmt"
    "net"
)

func main() {
    addr, _ := net.ResolveUDPAddr("udp", "localhost:8081")
    conn, _ := net.DialUDP("udp", nil, addr)
    defer conn.Close()

    conn.Write([]byte("Hello, UDP Server!")) // Send message

    buffer := make([]byte, 1024)
    n, _, _ := conn.ReadFromUDP(buffer) // Read response
    fmt.Println("Server response:", string(buffer[:n]))
}
```
📌 The **client** sends `"Hello, UDP Server!"` and prints the server's response.

---

## **3️⃣ Lookup IP Address**
✅ **Find the IP address of a domain.**  

```go
package main
import (
    "fmt"
    "net"
)

func main() {
    ips, _ := net.LookupIP("google.com")
    for _, ip := range ips {
        fmt.Println("IP Address:", ip)
    }
}
```
📌 **`net.LookupIP()`** retrieves the IP addresses associated with a hostname.

---

## **4️⃣ Lookup Hostname from IP**
✅ **Reverse DNS lookup (find the hostname from an IP address).**  

```go
package main
import (
    "fmt"
    "net"
)

func main() {
    names, _ := net.LookupAddr("8.8.8.8")
    fmt.Println("Hostnames:", names)
}
```
📌 **`net.LookupAddr()`** performs a reverse DNS lookup.

---

## **5️⃣ Checking Network Interface Information**
✅ **List available network interfaces.**  

```go
package main
import (
    "fmt"
    "net"
)

func main() {
    interfaces, _ := net.Interfaces()
    for _, iface := range interfaces {
        fmt.Println("Interface Name:", iface.Name)
    }
}
```
📌 **`net.Interfaces()`** lists network interfaces like `eth0`, `wlan0`.

---

### **🔹 Summary Table**
| `net` Function | Purpose |
|---------------|-------------------------------------------|
| `net.Listen("tcp", ":8080")` | Start a **TCP server** on port 8080. |
| `net.Dial("tcp", "localhost:8080")` | Create a **TCP client** to connect to a server. |
| `net.ListenUDP("udp", addr)` | Start a **UDP server**. |
| `net.DialUDP("udp", nil, addr)` | Create a **UDP client**. |
| `net.LookupIP("domain.com")` | Get the **IP address** of a domain. |
| `net.LookupAddr("IP")` | Perform **reverse DNS lookup** (IP to domain). |
| `net.Interfaces()` | List **network interfaces** on the system. |

Would you like **real-world applications** using the `net` package? 🚀