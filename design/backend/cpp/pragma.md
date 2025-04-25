`#pragma once` is a **preprocessor directive** in C++ that ensures a **header file is included only once** in a single compilation, preventing **duplicate declarations** and **multiple inclusion issues**.

---

### **Why Use `#pragma once`?**

When a header file is included multiple times, like this:

```cpp
#include "my_header.h"
#include "my_header.h"
```

Without protection, this can cause redefinition errors. Traditionally, include guards are used:

```cpp
#ifndef MY_HEADER_H
#define MY_HEADER_H

// declarations

#endif
```

But `#pragma once` is a **simpler and less error-prone** alternative:

```cpp
#pragma once

// declarations
```

---

### **Advantages**
- Simpler and less prone to typos than include guards
- Supported by most modern compilers (GCC, Clang, MSVC)
- Slightly faster compilation (no macro checks)

---

### **Disadvantages**
- Not part of the official C++ standard (but widely supported)
- May fail with symbolic links or complex file systems (rare)

---

### **Example**

```cpp
// my_header.h
#pragma once

class MyClass {
public:
    void doSomething();
};
```

```cpp
// main.cpp
#include "my_header.h"
#include "my_header.h"  // Included twice, but won't cause issues due to #pragma once
```

---

Let me know if you want a comparison between `#pragma once` and include guards in table format.


`#pragma` is a **compiler directive** in C and C++ that provides **additional instructions to the compiler**. It's used for **compiler-specific features** that aren't part of the standard language.

---

### **What Is `#pragma`?**
- Tells the compiler to **use specific settings** or **enable/disable features**
- **Syntax:** `#pragma name [options]`
- Varies between compilers (e.g., GCC, Clang, MSVC)
- Often used for **optimization, warnings, alignment**, etc.

---

### **Common `#pragma` Directives and Their Use Cases**

| `#pragma` Directive         | Description / Use Case                                                      | Compiler Support       |
|----------------------------|------------------------------------------------------------------------------|------------------------|

| `#pragma once`             | Include a header file only once during compilation                           | GCC, Clang, MSVC       |

| `#pragma warning`          | Enable/disable specific compiler warnings                                     | MSVC                   |

| `#pragma GCC optimize`     | Specify optimization options                                                  | GCC                    |

| `#pragma pack(n)`          | Control data structure alignment (e.g., for memory layout)                    | All major compilers    |

| `#pragma region` / `#pragma endregion` | Create collapsible code blocks in IDEs (mostly for readability)                  | MSVC, some IDEs        |

| `#pragma message("text")`  | Print a custom message during compilation                                     | GCC, MSVC              |

| `#pragma omp`              | Used for OpenMP to enable parallelism                                         | With OpenMP support    |

| `#pragma ivdep`            | Vectorization hint to the compiler                                            | Intel, some GCC        |

| `#pragma deprecated`       | Mark functions/types as deprecated                                            | MSVC                   |

---

### **Examples**

**1. `#pragma once`:**
```cpp
#pragma once
// Prevents multiple inclusion of this header
```

**2. `#pragma pack`:**
```cpp
#pragma pack(1)
struct PackedStruct {
    char a;
    int b;
};
#pragma pack()
```

**3. `#pragma message`:**
```cpp
#pragma message("Compiling module X...")
```

**4. `#pragma warning`:**
```cpp
#pragma warning(disable: 4996) // Disable a specific warning in MSVC
```

**5. `#pragma omp`:**
```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    // parallel loop body
}
```

---

Would you like a downloadable or Markdown table version for reference?


Here's a Markdown version of the common `#pragma` directives in C/C++ for easy inclusion in README files or documentation:

---

  
### Common `#pragma` Directives in C/C++

| `#pragma` Directive             | Description / Use Case                                                   | Supported By           |
|--------------------------------|---------------------------------------------------------------------------|------------------------|
| `#pragma once`                 | Ensures the file is only included once per compilation                    | GCC, Clang, MSVC       |
| `#pragma warning`              | Enable/disable specific compiler warnings                                 | MSVC                   |
| `#pragma GCC optimize`         | Specify optimization levels or behaviors (e.g., `("O3")`)                 | GCC                    |
| `#pragma pack(n)`              | Controls memory alignment of structures (`n` is number of bytes)          | GCC, Clang, MSVC       |
| `#pragma region` / `#pragma endregion` | Used in IDEs to allow code folding                                      | MSVC, supported IDEs   |
| `#pragma message("text")`      | Print custom message during compilation                                   | GCC, Clang, MSVC       |
| `#pragma omp`                  | Enables parallel programming constructs using OpenMP                      | With OpenMP support    |
| `#pragma ivdep`                | Ignore vector dependencies (for vectorization)                            | Intel, some GCC builds |
| `#pragma deprecated`           | Marks functions or variables as deprecated                                | MSVC                   |

---

Would you like a cheat sheet PDF or need this integrated into a code documentation tool like Doxygen or Sphinx?