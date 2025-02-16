## **OOP Concepts in Golang**  

Golang does not have traditional object-oriented programming (OOP) features like **classes** and **inheritance**, but it supports OOP principles through **structs**, **interfaces**, and **composition**.  

### **1️⃣ Encapsulation (Using Structs & Methods)**  
Encapsulation hides the internal details of a struct and exposes only necessary behavior through methods.

```go
package main
import "fmt"

// Define a struct with private and public fields
type Car struct {
	brand string  // private field
	Model string  // public field
}

// Constructor function
func NewCar(brand, model string) *Car {
	return &Car{brand: brand, Model: model}
}

// Public method
func (c *Car) GetBrand() string {
	return c.brand
}

func main() {
	car := NewCar("Toyota", "Corolla")
	fmt.Println(car.Model)      // Public field accessible
	fmt.Println(car.GetBrand()) // Access private field via method
}
```
**Key Points:**
- Fields that start with lowercase letters are **private**.
- Fields that start with uppercase letters are **public**.
- Methods allow controlled access to private fields.

---

### **2️⃣ Inheritance (Using Composition)**
Golang does not support **class-based inheritance** but uses **struct embedding (composition)** to achieve reusability.

```go
package main
import "fmt"

// Base struct
type Vehicle struct {
	Brand string
}

// Embedded struct (inherits Vehicle)
type Car struct {
	Vehicle
	Seats int
}

func main() {
	car := Car{Vehicle: Vehicle{Brand: "Honda"}, Seats: 5}
	fmt.Println(car.Brand) // Inherited from Vehicle
	fmt.Println(car.Seats) // Specific to Car
}
```
**Key Points:**
- `Car` embeds `Vehicle` to inherit its properties.
- Unlike classical inheritance, composition is **more flexible** and avoids deep hierarchies.

---

### **3️⃣ Polymorphism (Using Interfaces)**
Polymorphism allows different structs to implement the same interface.

```go
package main
import "fmt"

// Define an interface
type Animal interface {
	Speak() string
}

// Implement interface in Dog struct
type Dog struct{}
func (d Dog) Speak() string { return "Woof!" }

// Implement interface in Cat struct
type Cat struct{}
func (c Cat) Speak() string { return "Meow!" }

func makeSound(a Animal) {
	fmt.Println(a.Speak())
}

func main() {
	dog := Dog{}
	cat := Cat{}
	makeSound(dog) // "Woof!"
	makeSound(cat) // "Meow!"
}
```
**Key Points:**
- Interfaces define **common behavior** without enforcing struct relationships.
- **No explicit implementation** required; Go detects matching methods automatically.

---

### **4️⃣ Abstraction (Hiding Implementation Details)**
Abstraction allows exposing only relevant details via **interfaces**.

```go
package main
import "fmt"

type Shape interface {
	Area() float64
}

type Circle struct {
	Radius float64
}

func (c Circle) Area() float64 {
	return 3.14 * c.Radius * c.Radius
}

func printArea(s Shape) {
	fmt.Println("Area:", s.Area())
}

func main() {
	c := Circle{Radius: 5}
	printArea(c) // Calls the interface method
}
```
**Key Points:**
- The `Shape` interface hides how `Area()` is implemented.
- The user interacts only with `printArea(s Shape)`, not the internal details.

---

### **5️⃣ Method Overriding (Using Interfaces & Embedding)**
Golang **does not** support direct method overriding but achieves it using interfaces.

```go
package main
import "fmt"

type Vehicle struct{}

func (v Vehicle) Start() {
	fmt.Println("Vehicle is starting")
}

type Car struct {
	Vehicle
}

func (c Car) Start() { // Overrides Vehicle's Start()
	fmt.Println("Car is starting")
}

func main() {
	v := Vehicle{}
	v.Start() // "Vehicle is starting"

	c := Car{}
	c.Start() // "Car is starting"
}
```
**Key Points:**
- `Car` **overrides** `Start()` by redefining it.
- Calls `Car.Start()` instead of `Vehicle.Start()`.

---

### **6️⃣ Interfaces with Multiple Implementations**
A single interface can have **multiple implementations**.

```go
package main
import "fmt"

type Logger interface {
	Log(message string)
}

// Console logger
type ConsoleLogger struct{}
func (c ConsoleLogger) Log(message string) {
	fmt.Println("Console log:", message)
}

// File logger
type FileLogger struct{}
func (f FileLogger) Log(message string) {
	fmt.Println("File log:", message)
}

func writeLog(logger Logger, message string) {
	logger.Log(message)
}

func main() {
	cl := ConsoleLogger{}
	fl := FileLogger{}

	writeLog(cl, "Hello Console")
	writeLog(fl, "Hello File")
}
```
**Key Points:**
- `ConsoleLogger` and `FileLogger` **implement** the `Logger` interface.
- The `writeLog()` function works with **any logger type**.

---

### **7️⃣ Operator Overloading (Using Methods)**
Golang does not support **operator overloading**, but similar behavior can be achieved with **methods**.

```go
package main
import "fmt"

type Number struct {
	Value int
}

func (n Number) Add(other Number) Number {
	return Number{Value: n.Value + other.Value}
}

func main() {
	a := Number{10}
	b := Number{20}
	result := a.Add(b)
	fmt.Println(result.Value) // 30
}
```
**Key Points:**
- The `Add` method simulates `+` operator behavior.
- Golang avoids operator overloading for **simplicity**.

---

### **8️⃣ Multiple Inheritance (Using Interfaces)**
Golang does not support **multiple inheritance**, but you can achieve it via **multiple interfaces**.

```go
package main
import "fmt"

type Flyer interface {
	Fly()
}

type Swimmer interface {
	Swim()
}

type Bird struct{}

func (b Bird) Fly() {
	fmt.Println("Bird is flying")
}

func (b Bird) Swim() {
	fmt.Println("Bird is swimming")
}

func main() {
	b := Bird{}
	b.Fly()
	b.Swim()
}
```
**Key Points:**
- `Bird` implements both `Flyer` and `Swimmer` interfaces.
- This mimics **multiple inheritance** without class hierarchies.

---

### **9️⃣ Constructor Function (Simulating Constructors)**
Golang does not have constructors, but you can use factory functions.

```go
package main
import "fmt"

type User struct {
	Name string
	Age  int
}

// Constructor function
func NewUser(name string, age int) *User {
	return &User{Name: name, Age: age}
}

func main() {
	user := NewUser("Alice", 25)
	fmt.Println(user.Name, user.Age)
}
```
**Key Points:**
- Factory functions initialize structs.
- They replace constructors found in OOP languages.

---

## **🔹 Summary Table**
| OOP Concept     | Golang Equivalent |
|----------------|----------------|
| **Encapsulation** | Structs with public & private fields |
| **Inheritance** | Composition (Embedding structs) |
| **Polymorphism** | Interfaces |
| **Abstraction** | Interfaces hide implementation details |
| **Method Overriding** | Redefining methods in embedded structs |
| **Multiple Inheritance** | Implementing multiple interfaces |
| **Operator Overloading** | Not supported (use methods instead) |
| **Constructors** | Factory functions (`NewStruct()`) |

---

### **Final Thoughts**
Golang follows a **composition-over-inheritance** approach. Instead of deep class hierarchies, Go relies on **struct embedding** and **interfaces** for code reuse and polymorphism.

Would you like me to expand on any particular **OOP concept**? 🚀