The `override` keyword in **C++** is used to indicate that a member function is intended to **override** a **virtual function** in a base class. It was introduced in **C++11** to help catch errors at compile time, such as:

- Misspelling the function name
- Mismatching the signature
- Forgetting `virtual` in the base class

---

### **Basic Syntax and Example**

```cpp
class Base {
public:
    virtual void display() const {
        std::cout << "Base display\n";
    }
};

class Derived : public Base {
public:
    void display() const override {  // Correctly overrides Base::display
        std::cout << "Derived display\n";
    }
};
```

---

### **Why `override` Is Useful**

If you mistakenly don't match the base function signature, the compiler will give an error:

```cpp
class Derived : public Base {
public:
    void display() override { // Error: base function is const, this is not
        std::cout << "Derived display\n";
    }
};
```

Without `override`, this would silently create a new method instead of overriding.

---

### **Common Mistakes Caught by `override`**

| Mistake                       | Without `override` | With `override`     |
|------------------------------|--------------------|---------------------|
| Wrong function name          | Compiles silently  | Compile-time error  |
| Wrong parameter types        | Compiles silently  | Compile-time error  |
| Missing `const` qualifier    | Compiles silently  | Compile-time error  |

---

### **Best Practice**

- Always use `override` for overriding virtual methods.
- Combine with `final` if you want to prevent further overrides:
  
```cpp
void display() const override final;
```

---

Would you like examples with `virtual`, `final`, and `override` together?