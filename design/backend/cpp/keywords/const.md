The `const` keyword in **C++** is used to define **immutable values**—values that cannot be changed after initialization. It enhances safety and readability by clearly indicating which variables or parameters should not be modified.

---

### **Common Uses of `const` in C++**

#### 1. **Const Variables**
```cpp
const int x = 10;
// x = 20; // Error: cannot modify a const variable
```

#### 2. **Const Pointers**
- **Pointer itself is constant**:
  ```cpp
  int a = 10;
  int* const ptr = &a; // can't change ptr, but can change *ptr
  ```
- **Data pointed to is constant**:
  ```cpp
  const int* ptr = &a; // can change ptr, not *ptr
  ```
- **Both pointer and data are constant**:
  ```cpp
  const int* const ptr = &a;
  ```

#### 3. **Const Function Parameters**
```cpp
void print(const string& name) {
    // name = "Bob"; // Error
    cout << name;
}
```
Used for passing by reference safely.

#### 4. **Const Functions (Methods)**
```cpp
class Employee {
    string name;
public:
    string getName() const { return name; } // cannot modify members
};
```
Used to mark member functions that do **not modify** the object state.

#### 5. **Const Return Type**
```cpp
const string& getName() const {
    return name;
}
```
Prevents the returned value from being modified by the caller.

---

### **Why Use `const`?**

- **Prevents accidental modification**
- **Enables compiler optimizations**
- **Improves code readability and safety**
- **Works with overloading**:
  ```cpp
  void display();         // non-const version
  void display() const;   // const version
  ```

---

Would you like a diagram or a code demo showing multiple const scenarios together?


`constexpr` in C++ is a keyword introduced in C++11 (enhanced in C++14 and later) that specifies that the **value of a variable or function can be evaluated at compile time**. It helps improve performance by moving computations from runtime to compile time.

---

### **Purpose of `constexpr`**
- Enable **compile-time evaluation** of constants.
- Define functions and variables whose values are **known at compile time**.

---

### **1. `constexpr` Variables**

```cpp
constexpr int max_users = 100;
```

- The value `100` is known at compile time.
- Similar to `const`, but guarantees compile-time evaluation.

---

### **2. `constexpr` Functions**

```cpp
constexpr int square(int x) {
    return x * x;
}

constexpr int result = square(5);  // Evaluated at compile time
```

- Function must contain only a single `return` statement in C++11 (relaxed in C++14).
- Can be used in constant expressions.

---

### **3. C++14+ Enhancements**

In C++14, `constexpr` functions can have:
- Local variables
- Loops
- Multiple return statements
- Conditional logic

```cpp
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}
```

---

### **4. `constexpr` with Classes**

```cpp
struct Point {
    int x, y;
    constexpr Point(int a, int b) : x(a), y(b) {}
    constexpr int getX() const { return x; }
};

constexpr Point p(10, 20);
constexpr int px = p.getX();
```

---

### **Use Cases**
- Array sizes
- Template parameters
- Optimizing runtime performance
- Replacing macros with type-safe constants

---

### **Comparison with `const`**
| Feature         | `const`             | `constexpr`             |
|----------------|---------------------|--------------------------|
| Const at runtime | Yes               | Yes                      |
| Const at compile-time | Not guaranteed | Guaranteed (if valid)   |
| Used in array sizes | Sometimes        | Always valid             |
| Can be a function?  | No               | Yes                      |

---

Would you like a real-world example comparing runtime vs `constexpr` performance in C++?