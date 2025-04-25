### **`reflect` Package in Go**
The `reflect` package in Go provides powerful tools for inspecting and manipulating variables at runtime. It allows you to:
- Determine the **type** of a variable.
- Get and modify the **value** of a variable dynamically.
- Work with **struct fields and methods** dynamically.

---

### **Key Concepts in `reflect`**
1. **`reflect.TypeOf()`** – Retrieves the type of a variable.
2. **`reflect.ValueOf()`** – Retrieves the value of a variable.
3. **Modify values using reflection (requires `reflect.Value` to be addressable).**
4. **Work with struct fields and methods dynamically.**

---

### **Example 1: Get Type and Value of a Variable**
```go
package main

import (
	"fmt"
	"reflect"
)

func main() {
	var x int = 42

	t := reflect.TypeOf(x)
	v := reflect.ValueOf(x)

	fmt.Println("Type:", t)   // Output: int
	fmt.Println("Value:", v)  // Output: 42
}
```

---

### **Example 2: Modify a Variable using Reflection**
Reflection allows modifying a variable only if it's **addressable** (`reflect.Value` must be a pointer).

```go
package main

import (
	"fmt"
	"reflect"
)

func main() {
	var x int = 42
	v := reflect.ValueOf(&x) // Pass address to modify value
	if v.Kind() == reflect.Ptr {
		v = v.Elem() // Dereference pointer to modify
	}
	v.SetInt(100) // Modify value dynamically

	fmt.Println("Modified value:", x) // Output: 100
}
```

---

### **Example 3: Inspecting Struct Fields and Tags**
```go
package main

import (
	"fmt"
	"reflect"
)

type Person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func main() {
	p := Person{Name: "Alice", Age: 30}

	t := reflect.TypeOf(p)

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		fmt.Printf("Field: %s, Type: %s, Tag: %s\n", field.Name, field.Type, field.Tag.Get("json"))
	}
}
```
**Output:**
```
Field: Name, Type: string, Tag: name
Field: Age, Type: int, Tag: age
```

---

### **Example 4: Calling Methods Dynamically**
```go
package main

import (
	"fmt"
	"reflect"
)

type User struct{}

func (u User) Greet(name string) {
	fmt.Println("Hello,", name)
}

func main() {
	u := User{}
	method := reflect.ValueOf(u).MethodByName("Greet")

	args := []reflect.Value{reflect.ValueOf("Bob")}
	method.Call(args) // Output: Hello, Bob
}
```

---

### **Summary**
- `reflect.TypeOf(x)` → Gets the type of `x`.
- `reflect.ValueOf(x)` → Gets the value of `x`.
- To modify a variable, use **a pointer** (`reflect.ValueOf(&x).Elem()`).
- `reflect.Struct` allows reading **fields** and **tags** dynamically.
- You can call **methods dynamically** using `MethodByName()`.

Reflection is powerful but should be used **cautiously** as it affects performance and readability.