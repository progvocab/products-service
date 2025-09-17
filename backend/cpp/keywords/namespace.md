The `namespace` keyword in C++ is used to **organize code into logical groups** and to **avoid name conflicts**, especially in large projects or when using multiple libraries.

---

### **Why Use `namespace`?**

Imagine two libraries both define a function called `print()`. If you include both in your project, the compiler won't know which one to call. Namespaces solve this by **scoping** functions, variables, and classes.

---

### **Basic Syntax**

```cpp
namespace mylib {
    void print() {
        std::cout << "Hello from mylib" << std::endl;
    }
}
```

You can use it like this:

```cpp
mylib::print();  // using scope resolution operator
```

---

### **Using the `using` Directive**

```cpp
using namespace mylib;
print();  // no need for mylib::
```

> Caution: Avoid `using namespace` in headers or large scopes—can lead to name conflicts.

---

### **Nested Namespaces (C++17)**

```cpp
namespace company::project {
    void display() {
        std::cout << "Nested namespaces!" << std::endl;
    }
}
```

Call it using:

```cpp
company::project::display();
```

---

### **Anonymous Namespaces**

Used to limit scope **within a single translation unit (file)**:

```cpp
namespace {
    int internalValue = 42;
}
```

Equivalent to declaring something `static` in C.

---

### **Use Cases**

- Organizing library code
- Avoiding naming collisions
- Creating modular and maintainable code

---

### **Example**

```cpp
#include <iostream>

namespace math {
    int add(int a, int b) {
        return a + b;
    }
}

int main() {
    std::cout << math::add(3, 4);  // Output: 7
    return 0;
}
```

Would you like to see how namespaces can be split across multiple files or extended later?