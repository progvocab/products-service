### **Arrays in Golang**  

In Golang, an **array** is a **fixed-size** collection of elements of the same type. Arrays are **value types**, meaning they are copied when assigned or passed to functions.

---

## **1️⃣ Declaring an Array**
```go
package main
import "fmt"

func main() {
    var arr [5]int // An array of 5 integers (default values: 0)
    fmt.Println(arr) // Output: [0 0 0 0 0]
}
```
- The **size** of the array is **fixed** at **compile time**.
- Default values for numeric types are **0**, for booleans **false**, and for strings **""**.

---

## **2️⃣ Initializing an Array**
```go
package main
import "fmt"

func main() {
    arr := [5]int{10, 20, 30, 40, 50} 
    fmt.Println(arr) // Output: [10 20 30 40 50]
}
```

**Shorthand initialization with inferred size**:
```go
arr := [...]int{1, 2, 3, 4, 5} // Compiler infers size
fmt.Println(arr) // Output: [1 2 3 4 5]
```

---

## **3️⃣ Accessing and Modifying Elements**
```go
package main
import "fmt"

func main() {
    arr := [3]string{"Go", "Python", "Java"}
    fmt.Println(arr[1]) // Output: Python

    arr[2] = "C++"
    fmt.Println(arr) // Output: [Go Python C++]
}
```

---

## **4️⃣ Iterating Over an Array**
### **Using a `for` Loop**
```go
package main
import "fmt"

func main() {
    arr := [4]int{10, 20, 30, 40}

    for i := 0; i < len(arr); i++ {
        fmt.Println(arr[i])
    }
}
```

### **Using `range`**
```go
for index, value := range arr {
    fmt.Println("Index:", index, "Value:", value)
}
```
- `_` can be used to ignore the index:
```go
for _, value := range arr {
    fmt.Println(value) // Prints values only
}
```

---

## **5️⃣ Copying an Array (Value Type Behavior)**
```go
package main
import "fmt"

func main() {
    arr1 := [3]int{1, 2, 3}
    arr2 := arr1 // Creates a copy, not a reference
    arr2[0] = 100
    
    fmt.Println(arr1) // Output: [1 2 3]
    fmt.Println(arr2) // Output: [100 2 3]
}
```
- Changes to `arr2` **do not affect** `arr1` since arrays are **copied** by value.

---

## **6️⃣ Passing an Array to a Function**
```go
func modify(arr [3]int) {
    arr[0] = 100 // Modifies the copy, not the original
}

func main() {
    arr := [3]int{1, 2, 3}
    modify(arr) 
    fmt.Println(arr) // Output: [1 2 3] (unchanged)
}
```
✅ **Use slices (`[]int`) instead of arrays for better flexibility**.

---

## **7️⃣ Multidimensional Arrays**
```go
package main
import "fmt"

func main() {
    matrix := [2][3]int{
        {1, 2, 3},
        {4, 5, 6},
    }

    fmt.Println(matrix) // Output: [[1 2 3] [4 5 6]]
    fmt.Println(matrix[1][2]) // Output: 6
}
```

---

## **8️⃣ Array of Pointers**
```go
package main
import "fmt"

func main() {
    arr := [3]*int{}
    x, y, z := 10, 20, 30
    arr[0], arr[1], arr[2] = &x, &y, &z

    fmt.Println(*arr[0], *arr[1], *arr[2]) // Output: 10 20 30
}
```

---

## **9️⃣ When to Use Arrays?**
✅ When **size is fixed** and performance is critical.  
✅ When **memory layout and cache locality** are important.  
❌ **Prefer slices (`[]T`) for flexibility** in real-world applications.

Would you like a **comparison between arrays and slices**? 🚀