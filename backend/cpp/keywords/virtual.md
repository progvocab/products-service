The `virtual` keyword in **C++** is used to enable **runtime polymorphism**. It allows a **base class function** to be **overridden** in derived classes, and ensures that the **correct function** is called **via a base class pointer or reference**, depending on the actual object type.

---

### **Why Use `virtual`?**
Without `virtual`, the function call is **resolved at compile time** (static binding). With `virtual`, it is **resolved at runtime** (dynamic dispatch) via a **vtable (virtual table)** mechanism.

---

### **Basic Example**

```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual void show() {
        cout << "Base class\n";
    }
};

class Derived : public Base {
public:
    void show() override {
        cout << "Derived class\n";
    }
};

int main() {
    Base* bptr;
    Derived d;
    bptr = &d;

    bptr->show();  // Calls Derived::show() due to virtual
}
```

---

### **Key Concepts**

| Concept                     | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `virtual` function          | Enables overriding with dynamic dispatch                    |
| `override` keyword          | Ensures derived method overrides a base virtual method      |
| `final` keyword             | Prevents further overriding of a virtual method             |
| `virtual` destructor        | Ensures proper destruction in polymorphic base classes      |

---

### **Virtual Destructors Example**

```cpp
class Base {
public:
    virtual ~Base() {
        cout << "Base destroyed\n";
    }
};

class Derived : public Base {
public:
    ~Derived() {
        cout << "Derived destroyed\n";
    }
};

int main() {
    Base* obj = new Derived();
    delete obj;  // Both destructors called
}
```

Without a virtual destructor, only `Base`'s destructor would be called—causing a memory leak if `Derived` allocates resources.

---

### **When to Use `virtual`**

- When you expect derived classes to **override** base class behavior.
- When using **base class pointers or references**.
- When designing **polymorphic interfaces**.

---

Would you like to explore how `virtual` relates to abstract classes and interfaces in C++?