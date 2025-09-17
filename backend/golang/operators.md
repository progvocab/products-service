### **Operators in Go (Golang)**
Go provides a variety of operators for performing operations on variables and values. These operators can be categorized into **arithmetic, relational, logical, bitwise, assignment, and miscellaneous operators**.

---

## **1. Arithmetic Operators**  
Used for performing mathematical operations.

| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `+` | Addition | `5 + 3` | `8` |
| `-` | Subtraction | `5 - 3` | `2` |
| `*` | Multiplication | `5 * 3` | `15` |
| `/` | Division | `5 / 2` | `2` (integer division) |
| `%` | Modulus (Remainder) | `5 % 2` | `1` |

### **Example**
```go
package main
import "fmt"

func main() {
    a, b := 10, 3
    fmt.Println("Addition:", a+b)
    fmt.Println("Subtraction:", a-b)
    fmt.Println("Multiplication:", a*b)
    fmt.Println("Division:", a/b)
    fmt.Println("Modulus:", a%b)
}
```

---

## **2. Relational (Comparison) Operators**  
Used for comparing values, returning `true` or `false`.

| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `==` | Equal to | `5 == 3` | `false` |
| `!=` | Not equal to | `5 != 3` | `true` |
| `>` | Greater than | `5 > 3` | `true` |
| `<` | Less than | `5 < 3` | `false` |
| `>=` | Greater than or equal to | `5 >= 5` | `true` |
| `<=` | Less than or equal to | `3 <= 5` | `true` |

### **Example**
```go
package main
import "fmt"

func main() {
    fmt.Println(10 == 10) // true
    fmt.Println(10 != 5)  // true
    fmt.Println(10 > 5)   // true
    fmt.Println(10 < 5)   // false
}
```

---

## **3. Logical Operators**  
Used for boolean operations.

| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `&&` | Logical AND | `true && false` | `false` |
| `||` | Logical OR | `true || false` | `true` |
| `!` | Logical NOT | `!true` | `false` |

### **Example**
```go
package main
import "fmt"

func main() {
    a, b := true, false
    fmt.Println(a && b) // false
    fmt.Println(a || b) // true
    fmt.Println(!a)     // false
}
```

---

## **4. Bitwise Operators**  
Used for bit-level operations.

| Operator | Description | Example |
|----------|------------|---------|
| `&` | Bitwise AND | `5 & 3` → `1` |
| `|` | Bitwise OR | `5 | 3` → `7` |
| `^` | Bitwise XOR | `5 ^ 3` → `6` |
| `&^` | Bitwise AND NOT (Clears bits) | `5 &^ 3` → `4` |
| `<<` | Left Shift | `5 << 1` → `10` |
| `>>` | Right Shift | `5 >> 1` → `2` |

### **Example**
```go
package main
import "fmt"

func main() {
    fmt.Println(5 & 3)  // 1
    fmt.Println(5 | 3)  // 7
    fmt.Println(5 ^ 3)  // 6
    fmt.Println(5 &^ 3) // 4
    fmt.Println(5 << 1) // 10
    fmt.Println(5 >> 1) // 2
}
```

---

## **5. Assignment Operators**  
Used for assigning values to variables.

| Operator | Description | Example |
|----------|------------|---------|
| `=` | Assign | `a = 5` |
| `+=` | Add and assign | `a += 3` (same as `a = a + 3`) |
| `-=` | Subtract and assign | `a -= 3` (same as `a = a - 3`) |
| `*=` | Multiply and assign | `a *= 3` (same as `a = a * 3`) |
| `/=` | Divide and assign | `a /= 3` (same as `a = a / 3`) |
| `%=` | Modulus and assign | `a %= 3` (same as `a = a % 3`) |
| `&=` | Bitwise AND and assign | `a &= 3` |
| `|=` | Bitwise OR and assign | `a |= 3` |
| `^=` | Bitwise XOR and assign | `a ^= 3` |
| `<<=` | Left shift and assign | `a <<= 1` |
| `>>=` | Right shift and assign | `a >>= 1` |

### **Example**
```go
package main
import "fmt"

func main() {
    a := 5
    a += 3  // a = a + 3
    fmt.Println(a) // 8
}
```

---

## **6. Miscellaneous Operators**  
| Operator | Description | Example |
|----------|------------|---------|
| `*` | Pointer dereferencing | `*ptr` |
| `&` | Address of a variable | `&x` |
| `:=` | Short variable declaration | `x := 5` |

### **Example**
```go
package main
import "fmt"

func main() {
    x := 10
    p := &x  // Pointer to x
    fmt.Println(*p)  // Dereference pointer (prints 10)
}
```

---

## **Summary Table**
| Category | Operators |
|----------|-----------|
| **Arithmetic** | `+`, `-`, `*`, `/`, `%` |
| **Relational** | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| **Logical** | `&&`, `||`, `!` |
| **Bitwise** | `&`, `|`, `^`, `&^`, `<<`, `>>` |
| **Assignment** | `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=` |
| **Miscellaneous** | `*`, `&`, `:=` |

---

Would you like an **example of a specific operator in a real-world scenario**?