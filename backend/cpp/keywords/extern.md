The `extern` keyword in C++ is used to **declare a variable or function that is defined in another translation unit or file**. It tells the compiler that the **definition exists elsewhere**, and **avoids multiple definitions** during linking.

---

### **1. `extern` with Variables**

```cpp
// file1.cpp
int count = 42;

// file2.cpp
extern int count; // declaration
void show() {
    std::cout << count; // OK, uses the variable from file1.cpp
}
```

- `extern int count;` means "count is declared here, but defined elsewhere."
- Used when sharing global variables across multiple files.

---

### **2. `extern` with Functions**

Functions are implicitly `extern` by default, but you can specify it explicitly:

```cpp
// file1.cpp
void greet() {
    std::cout << "Hello\n";
}

// file2.cpp
extern void greet(); // declaration
greet(); // OK
```

---

### **3. `extern "C"` for C/C++ Interoperability**

Tells C++ compiler to use **C-style linkage** (no name mangling), useful when linking with C libraries.

```cpp
extern "C" {
    void c_function(); // tells compiler this is a C function
}
```

---

### **Key Points**

- **Declares** but doesn’t define.
- Prevents **multiple definitions**.
- Required for **sharing globals/functions across files**.
- Used for **linking C libraries** in C++.

---

Would you like an example showing how to use `extern` with a header and multiple source files?