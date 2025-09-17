The `inline` keyword in C++ has **two major uses**, depending on context and C++ version:

---

### **1. Inline Functions (Since C++98)**

Used to **suggest** to the compiler to insert the function body directly at the call site, avoiding function call overhead.

```cpp
inline int add(int a, int b) {
    return a + b;
}
```

- Typically used for **short, performance-critical functions**.
- **Modern compilers ignore the hint** and perform inlining automatically based on optimization heuristics.

#### **Why use `inline` then?**
It primarily helps to **avoid multiple definition errors** when defining functions in header files.

```cpp
// math_utils.h
inline int square(int x) { return x * x; }
```

When included in multiple `.cpp` files, `inline` prevents linker errors due to multiple definitions.

---

### **2. Inline Variables (Since C++17)**

Allows defining **global constants or variables** in header files without violating the **One Definition Rule (ODR)**.

```cpp
inline constexpr int MAX_COUNT = 100;
```

- Ensures the variable can be defined in multiple translation units without linker errors.
- Commonly used with `constexpr` or `const`.

---

### **Summary Table**

| Use Case           | Purpose                                                  | C++ Version |
|--------------------|----------------------------------------------------------|-------------|
| Function Inlining  | Hint compiler to replace function call with body         | C++98       |
| ODR-safe Definition| Allow function/variable definition in headers            | C++98 (function), C++17 (variable) |
| Constant Sharing   | Define shared constants in headers (`inline constexpr`)  | C++17       |

---

Would you like a quick example of how `inline` prevents linker errors in multi-file projects?