### **Golang REPL (Read-Eval-Print Loop)**  

Unlike Python, Go does **not** have a built-in REPL (Read-Eval-Print Loop) by default. However, you can use third-party tools to get an interactive Go shell.

---

## **1️⃣ Using `gore` (Go REPL)**
[`gore`](https://github.com/motemen/gore) is a popular third-party interactive Go shell.

### **🔹 Install `gore`**
```bash
go install github.com/motemen/gore@latest
```
Ensure `$GOPATH/bin` is in your `PATH`:
```bash
export PATH=$PATH:$(go env GOPATH)/bin
```

### **🔹 Start Go REPL**
Run:
```bash
gore
```
Example usage inside `gore`:
```go
gore> fmt.Println("Hello, Go!")
Hello, Go!

gore> 10 + 20
30
```

---

## **2️⃣ Using `yaegi` (Go Interpreter)**
[`yaegi`](https://github.com/traefik/yaegi) is a Go interpreter that allows interactive execution.

### **🔹 Install `yaegi`**
```bash
go install github.com/traefik/yaegi/cmd/yaegi@latest
```
Run:
```bash
yaegi
```
Example:
```go
> fmt.Println("Go Interactive Shell")
Go Interactive Shell
```

---

## **3️⃣ Using `go run` for Quick Testing**
If you don't want a REPL but need quick execution, you can use:
```bash
echo 'package main; import "fmt"; func main() { fmt.Println("Hello, Go!") }' | go run -
```

---

### **🚀 Which One Should You Use?**
- ✅ **`gore`** → Best for Go REPL with package imports  
- ✅ **`yaegi`** → Best for scripting and interactive execution  
- ✅ **`go run`** → Quick testing without setting up a project  

Would you like examples of running Go snippets interactively? 🚀