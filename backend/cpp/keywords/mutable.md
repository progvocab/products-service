The `mutable` keyword in **C++** is used to **allow modification** of a class member even if it is part of an object declared as `const`. It specifically overrides the `const` qualifier on data members.

---

### **Why `mutable` is Needed**

When a method is marked as `const`, it **cannot modify any member variables**—unless those variables are marked `mutable`.

---

### **Syntax & Example**

```cpp
#include <iostream>
using namespace std;

class Logger {
private:
    mutable int accessCount = 0;

public:
    string name;

    Logger(string n) : name(n) {}

    void printName() const {
        ++accessCount; // Allowed because accessCount is mutable
        cout << "Name: " << name << ", Accessed: " << accessCount << " times\n";
    }
};

int main() {
    const Logger logger("Alice");
    logger.printName();  // Works even though logger is const
}
```

---

### **Key Points**

- **Only applies to member variables** of classes/structs.
- Often used for **caching, logging, lazy evaluation, or statistics** inside const methods.
- Must be used **cautiously**, as it can **break const-correctness** if misused.

---

### **Typical Use Cases**

- **Caching** results inside a `const` function.
- **Logging** access even for logically `const` objects.
- **Thread-safe state tracking**, like a mutex or atomic counter.

---

Would you like a comparison between `mutable` and `const_cast` as well?