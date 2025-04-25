### **Strings in Golang**  

In Go, a **string** is an immutable sequence of bytes (UTF-8 encoded characters). Unlike other languages, Go **does not have a character type**; instead, a character is represented as a **rune** (Unicode code point).  

---

## **1. Declaring Strings**
```go
package main
import "fmt"

func main() {
    var s1 string = "Hello, Go!"
    s2 := "Golang"
    fmt.Println(s1) // Output: Hello, Go!
    fmt.Println(s2) // Output: Golang
}
```

---

## **2. String Operations**  

### **a) String Length**
```go
s := "Hello"
fmt.Println(len(s)) // Output: 5
```

### **b) Accessing Characters**
```go
s := "GoLang"
fmt.Println(s[0]) // Output: 71 (ASCII of 'G')
fmt.Println(string(s[0])) // Output: G
```

### **c) Concatenation**
```go
s1 := "Hello"
s2 := " World"
result := s1 + s2
fmt.Println(result) // Output: Hello World
```

### **d) String Comparison**
```go
s1 := "Go"
s2 := "Go"
fmt.Println(s1 == s2) // Output: true
fmt.Println(s1 != "Java") // Output: true
```

### **e) Substring (Using Slicing)**
```go
s := "Golang"
fmt.Println(s[1:4]) // Output: ola
```

---

## **3. Strings Package Functions**
Go provides the `strings` package for advanced string operations.

```go
import "strings"
```

### **a) Contains**
```go
fmt.Println(strings.Contains("Hello, Go!", "Go")) // true
```

### **b) Split**
```go
words := strings.Split("Go is fun", " ")
fmt.Println(words) // Output: [Go is fun]
```

### **c) Replace**
```go
fmt.Println(strings.Replace("hello world", "world", "Golang", 1)) 
// Output: hello Golang
```

### **d) Trim (Removes spaces)**
```go
fmt.Println(strings.TrimSpace("  GoLang  ")) // Output: "GoLang"
```

### **e) ToUpper / ToLower**
```go
fmt.Println(strings.ToUpper("golang")) // Output: GOLANG
fmt.Println(strings.ToLower("GOLANG")) // Output: golang
```

---

## **4. String to Integer and Vice Versa**
### **a) Convert String to Integer (`strconv.Atoi`)**
```go
import "strconv"

num, err := strconv.Atoi("123")
if err == nil {
    fmt.Println(num) // Output: 123
}
```

### **b) Convert Integer to String (`strconv.Itoa`)**
```go
str := strconv.Itoa(456)
fmt.Println(str) // Output: "456"
```

---

## **5. String Iteration (Using Runes)**
Since Go strings are **byte slices**, iterating over them using `range` gives **rune values** (Unicode code points).
```go
s := "Go😊"
for _, ch := range s {
    fmt.Printf("%c ", ch)
}
// Output: G o 😊
```

---

### **Key Takeaways**
✅ **Strings are immutable in Go** (modifying them creates a new string).  
✅ **Use `strings` package** for advanced operations.  
✅ **Use `strconv` package** for conversions.  
✅ **Iterate with `range`** to handle Unicode characters properly.  

Would you like an example on **string builder for performance**?