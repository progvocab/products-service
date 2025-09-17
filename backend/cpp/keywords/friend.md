In **C++**, the `friend` keyword allows a **non-member function**, **another class**, or a **member function of another class** to **access the private and protected members** of a class.

This is used when two or more classes or functions need **close cooperation** but you don't want to expose internal members globally.

---

### **Types of Friendships**

| Type                           | Explanation                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| Friend function                | A non-member function with access to private/protected members.             |
| Friend class                   | All methods of the friend class get access to the private members.          |
| Friend member function         | A specific member function of another class is granted access.              |

---

### **1. Friend Function Example**

```cpp
#include <iostream>
using namespace std;

class Box {
private:
    int width;
public:
    Box(int w) : width(w) {}
    friend void printWidth(Box b);  // Friend function
};

void printWidth(Box b) {
    cout << "Width is: " << b.width << endl;  // Can access private member
}
```

---

### **2. Friend Class Example**

```cpp
class Box;

class Printer {
public:
    void print(Box& b);
};

class Box {
private:
    int height = 10;
    friend class Printer;  // All Printer methods can access private members
};

void Printer::print(Box& b) {
    cout << "Height is: " << b.height << endl;
}
```

---

### **3. Friend Member Function of Another Class**

```cpp
class A;

class B {
public:
    void showA(A& a);  // Only this method is a friend
};

class A {
private:
    int x = 100;
    friend void B::showA(A&);  // Only showA() gets access
};

void B::showA(A& a) {
    cout << "A::x = " << a.x << endl;
}
```

---

### **Key Points**

- Friendship is **not mutual**: If A is a friend of B, B is not automatically a friend of A.
- Friendship is **not inherited**.
- Use sparingly: It breaks **encapsulation** and should only be used when **tight coupling** is justified.

---

Would you like to see how `friend` compares with `public` setters/getters or use cases in operator overloading?