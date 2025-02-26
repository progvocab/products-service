### **Types in Golang**  

Go is a **statically typed** language, meaning each variable has a specific type that is checked at compile-time. The types in Go can be categorized into several groups.

---

## **1. Basic Types**  
### **a) Boolean (`bool`)**  
- Holds **true** or **false** values.  
- Default value: `false`.  
```go
var flag bool = true
fmt.Println(flag) // Output: true
```

### **b) Numeric Types**  
#### **1. Integer Types**
| Type       | Size   | Range |
|------------|--------|------------|
| `int8`     | 8-bit  | -128 to 127 |
| `int16`    | 16-bit | -32,768 to 32,767 |
| `int32`    | 32-bit | -2,147,483,648 to 2,147,483,647 |
| `int64`    | 64-bit | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| `int`      | Platform-dependent (32-bit or 64-bit) |

```go
var num int32 = 100
fmt.Println(num) // Output: 100
```

#### **2. Unsigned Integer Types**
| Type       | Size   | Range |
|------------|--------|------------|
| `uint8`    | 8-bit  | 0 to 255 |
| `uint16`   | 16-bit | 0 to 65,535 |
| `uint32`   | 32-bit | 0 to 4,294,967,295 |
| `uint64`   | 64-bit | 0 to 18,446,744,073,709,551,615 |
| `uint`     | Platform-dependent (32-bit or 64-bit) |

```go
var uNum uint16 = 500
fmt.Println(uNum) // Output: 500
```

#### **3. Floating-Point Types**
| Type       | Size   | Precision |
|------------|--------|------------|
| `float32`  | 32-bit | ~6-7 decimal places |
| `float64`  | 64-bit | ~15 decimal places |

```go
var pi float64 = 3.14159
fmt.Println(pi) // Output: 3.14159
```

#### **4. Complex Number Types**
| Type       | Size   |
|------------|--------|
| `complex64`  | 64-bit (32-bit real + 32-bit imaginary) |
| `complex128` | 128-bit (64-bit real + 64-bit imaginary) |

```go
var c complex64 = complex(2, 3)
fmt.Println(c) // Output: (2+3i)
```

---

## **2. String Type (`string`)**  
- A **sequence of UTF-8 characters** (immutable).  
- Default value: `""` (empty string).  
```go
var name string = "Golang"
fmt.Println(name) // Output: Golang
```

- **Multiline string** (using backticks `` ` ``):
```go
var multiline = `This is 
a multiline string`
fmt.Println(multiline)
```

---

## **3. Derived (Composite) Types**  
### **a) Array (`[N]T`)**  
- Fixed-size collection of elements of the same type.
```go
var arr [3]int = [3]int{10, 20, 30}
fmt.Println(arr[1]) // Output: 20
```

### **b) Slice (`[]T`)**  
- **Dynamic-sized** version of an array.
```go
nums := []int{1, 2, 3}
nums = append(nums, 4) // Expands dynamically
fmt.Println(nums) // Output: [1 2 3 4]
```

### **c) Map (`map[K]V`)**  
- **Key-value** data structure.
```go
var user map[string]int = map[string]int{"Alice": 25, "Bob": 30}
fmt.Println(user["Alice"]) // Output: 25
```

### **d) Struct (`struct`)**  
- Collection of different fields (like an object).
```go
type Person struct {
    Name string
    Age  int
}

p := Person{Name: "John", Age: 30}
fmt.Println(p.Name) // Output: John
```

### **e) Pointer (`*T`)**  
- Holds the **memory address** of another variable.
```go
var num int = 10
var ptr *int = &num
fmt.Println(*ptr) // Output: 10
```

### **f) Function Type (`func`)**  
- Functions are **first-class citizens** and can be assigned to variables.
```go
func add(a, b int) int {
    return a + b
}

var sumFunc func(int, int) int = add
fmt.Println(sumFunc(2, 3)) // Output: 5
```

### **g) Interface (`interface{}`)**  
- Defines a **set of methods** a type must implement.
```go
type Shape interface {
    Area() float64
}

type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14 * c.Radius * c.Radius
}

var s Shape = Circle{5}
fmt.Println(s.Area()) // Output: 78.5
```

### **h) Channel (`chan T`)**  
- Used for **goroutine communication**.
```go
ch := make(chan int)
go func() { ch <- 10 }()
fmt.Println(<-ch) // Output: 10
```

---

## **4. Type Aliases and Custom Types**
### **a) Type Alias (`type NewName = OldType`)**
```go
type Age = int
var myAge Age = 25
fmt.Println(myAge) // Output: 25
```

### **b) Custom Type (`type NewType OldType`)**
```go
type Celsius float64
var temp Celsius = 36.5
fmt.Println(temp) // Output: 36.5
```

---

## **5. Empty Interface (`interface{}` or `any`)**
Holds **any type** (since all types implement `interface{}`).
```go
var data any = "Hello"
fmt.Println(data) // Output: Hello
```

---

## **Summary Table**
| Type Category | Example Types | Description |
|--------------|--------------|-------------|
| **Basic** | `bool`, `int`, `uint`, `float64`, `complex128`, `string` | Simple built-in types |
| **Composite** | `array`, `slice`, `map`, `struct`, `pointer` | Collections & objects |
| **Functional** | `func(int, int) int` | Function type |
| **Interface** | `interface{}` (or `any`) | Defines behavior |
| **Concurrency** | `chan`, `select` | Used for goroutines |

---

### **Key Takeaways**
✅ Go **does not support inheritance**, but **structs & interfaces** replace OOP.  
✅ **Use slices instead of arrays** for dynamic collections.  
✅ **Maps and structs** are useful for complex data structures.  
✅ **Channels** enable **goroutine communication**.  

Would you like a **comparison with C/C++/Java types**?