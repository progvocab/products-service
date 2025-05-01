Inheritance in **C++** is a fundamental object-oriented programming concept where a **class (child or derived)** inherits attributes and methods from another **class (parent or base)**. It promotes **code reusability**, **extensibility**, and supports **polymorphism**.

---

### **1. Basic Syntax**

```cpp
class Base {
public:
    void greet() {
        std::cout << "Hello from Base\n";
    }
};

class Derived : public Base {
    // Inherits greet() from Base
};
```

---

### **2. Example with Methods and Access Specifiers**

```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    void eat() {
        cout << "This animal eats food.\n";
    }

protected:
    void breathe() {
        cout << "This animal breathes air.\n";
    }

private:
    void sleep() {
        cout << "Sleeping...\n";
    }
};

class Dog : public Animal {
public:
    void bark() {
        cout << "Dog barks.\n";
        // breathe();  // Allowed: protected member
        // sleep();    // Not allowed: private in base class
    }
};

int main() {
    Dog d;
    d.eat();   // Inherited public method
    d.bark();  // Child method
}
```

---

### **3. Types of Inheritance**

| Type           | Syntax                        | Description                                |
|----------------|-------------------------------|--------------------------------------------|
| Single         | `class A : public B`           | One base and one derived class             |
| Multiple       | `class C : public A, public B` | Inherit from multiple base classes         |
| Multilevel     | A → B → C                      | Chain of inheritance                       |
| Hierarchical   | A → B, A → C                   | One base class, multiple derived classes   |
| Hybrid         | Combination of above           | Requires care (e.g., Diamond problem)      |

---

### **4. Virtual Inheritance (Diamond Problem)**

```cpp
class A {
public:
    void say() { cout << "A\n"; }
};

class B : virtual public A {};
class C : virtual public A {};
class D : public B, public C {};

int main() {
    D obj;
    obj.say(); // Only one instance of A
}
```

---

### **Use Cases**

- Reuse code across multiple classes.
- Specialize generic behavior (polymorphism).
- Extend existing systems without modifying base code.

---

Would you like examples for polymorphism or virtual functions with inheritance?