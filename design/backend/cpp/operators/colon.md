In C++, the **single colon (`:`)** and **double colon (`::`)** are completely different and serve distinct purposes.

---

### **1. Single Colon (`:`)**

Used mainly in two contexts:

#### a) **Constructor Initialization List**

```cpp
class MyClass {
    int x;
public:
    MyClass(int a) : x(a) {}  // ':' initializes x with a
};
```

- **Purpose**: Initializes class members before the constructor body executes.
- **Why use it?** Essential for initializing `const`, reference, or base class members.

#### b) **Access Specifiers in Inheritance**

```cpp
class Base {};
class Derived : public Base {};  // ':' used to inherit
```

- **Purpose**: Indicates **inheritance** type (`public`, `protected`, or `private`).

---

### **2. Double Colon (`::`) — Scope Resolution Operator**

Used to access members (variables, functions, classes) **in a specific scope**.

#### a) **Accessing Global Variables**

```cpp
int x = 10;
void func() {
    int x = 20;
    cout << ::x;  // prints 10, accesses global x
}
```

#### b) **Defining Methods Outside Class**

```cpp
class MyClass {
public:
    void show();
};

void MyClass::show() {
    cout << "Hello";
}
```

#### c) **Accessing Static Class Members**

```cpp
class MyClass {
public:
    static int count;
};

int MyClass::count = 0;  // using ::
```

#### d) **Namespaces**

```cpp
namespace myspace {
    int val = 5;
}

int main() {
    cout << myspace::val;  // access namespace variable
}
```

---

### **Summary Table**

| Operator | Name                   | Purpose                                  | Common Use Cases                         |
|----------|------------------------|------------------------------------------|-------------------------------------------|
| `:`      | Single colon           | Constructor init, inheritance            | `: public`, `: x(a)`                      |
| `::`     | Scope resolution       | Access members in scope or class/namespace | `MyClass::`, `::global_var`, `namespace::` |

Would you like example code comparing both in a single program?