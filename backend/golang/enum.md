### **Enums in Go**
Go does **not** have built-in enum types like other languages (e.g., C, Java). However, you can simulate enums using **constants and `iota`**, which is Go's way of defining sequential values.

---

## **1. Basic Enum Using `iota`**
Go uses `iota`, which auto-increments values within a `const` block.

```go
package main

import "fmt"

type Day int

const (
	Sunday Day = iota
	Monday
	Tuesday
	Wednesday
	Thursday
	Friday
	Saturday
)

func main() {
	fmt.Println(Sunday, Monday, Tuesday) // Output: 0 1 2
}
```
- `iota` starts at **0** and increments automatically.
- Each constant gets a unique integer value.

---

## **2. Enum with Custom Start Value**
You can set a custom start value.

```go
package main

import "fmt"

type Status int

const (
	Pending Status = iota + 1 // Starts from 1
	Approved
	Rejected
)

func main() {
	fmt.Println(Pending, Approved, Rejected) // Output: 1 2 3
}
```

---

## **3. Enum with Explicit Values**
You can manually assign values instead of `iota`.

```go
package main

import "fmt"

type LogLevel int

const (
	Debug   LogLevel = 10
	Info    LogLevel = 20
	Warning LogLevel = 30
	Error   LogLevel = 40
)

func main() {
	fmt.Println(Debug, Info, Warning, Error) // Output: 10 20 30 40
}
```

---

## **4. Enum with String Representation**
Since Go enums are integers, we can add a `String()` method for better readability.

```go
package main

import "fmt"

type Status int

const (
	Pending Status = iota + 1
	Approved
	Rejected
)

func (s Status) String() string {
	switch s {
	case Pending:
		return "Pending"
	case Approved:
		return "Approved"
	case Rejected:
		return "Rejected"
	default:
		return "Unknown"
	}
}

func main() {
	status := Approved
	fmt.Println(status) // Output: Approved
}
```

---

## **5. Enum with a `map` for String Lookup**
Another way to convert enums to strings is using a **map**.

```go
package main

import "fmt"

type Role int

const (
	User Role = iota + 1
	Admin
	Moderator
)

var roleNames = map[Role]string{
	User:      "User",
	Admin:     "Admin",
	Moderator: "Moderator",
}

func (r Role) String() string {
	return roleNames[r]
}

func main() {
	fmt.Println(User, Admin, Moderator) // Output: User Admin Moderator
}
```

---

## **6. Enum with Bit Flags (Using `iota` and Bitwise Operators)**
You can use **bitwise operations** to create flag-based enums.

```go
package main

import "fmt"

type Permission uint

const (
	Read Permission = 1 << iota  // 1 (0001)
	Write                        // 2 (0010)
	Execute                      // 4 (0100)
)

func hasPermission(userPerm Permission, check Permission) bool {
	return userPerm&check != 0
}

func main() {
	userPerm := Read | Execute // User has Read + Execute

	fmt.Println(hasPermission(userPerm, Read))   // Output: true
	fmt.Println(hasPermission(userPerm, Write))  // Output: false
	fmt.Println(hasPermission(userPerm, Execute)) // Output: true
}
```
- `1 << iota` creates **bit flags** for permission checks.
- **`&` (bitwise AND)** checks if a permission is set.

---

### **Summary**
| Enum Type | Example |
|-----------|---------|
| Basic Enum (`iota`) | `const (A = iota, B, C)` |
| Custom Start Value | `const (X = iota + 10, Y, Z)` |
| Explicit Values | `const (Red = 1, Green = 2, Blue = 3)` |
| String Representation | Implement `String()` method |
| Lookup Table | Use a `map[Enum]string` |
| Bit Flags | Use `1 << iota` with bitwise operations |

Enums in Go are implemented using `iota` and **constants**, making them simple and efficient.