In **C++**, **templates** are a powerful feature that allow you to write **generic and reusable code**. They let you define **functions** or **classes** with a placeholder for a data type, which is specified later when the function or class is instantiated.

---

### **Types of Templates in C++**

1. **Function Templates**  
2. **Class Templates**  
3. **Variadic Templates** (C++11 and later)

---

### **1. Function Template**

```cpp
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    cout << max(3, 7);       // int
    cout << max(3.5, 2.5);   // double
}
```

- `T` is a placeholder for any data type.
- Compiler generates the function when it's used with a specific type.

---

### **2. Class Template**

```cpp
template <typename T>
class Box {
private:
    T value;
public:
    void set(T val) { value = val; }
    T get() { return value; }
};

int main() {
    Box<int> intBox;
    intBox.set(100);

    Box<string> strBox;
    strBox.set("Hello");
}
```

- A single class can be used for multiple types.

---

### **3. Variadic Template (Advanced)**

Used to accept a variable number of template arguments:

```cpp
template<typename T, typename... Args>
void printAll(T first, Args... rest) {
    cout << first << endl;
    if constexpr (sizeof...(rest) > 0)
        printAll(rest...);
}
```

---

### **Use Cases**

- Writing generic libraries (like STL: `vector<T>`, `map<K, V>`)
- Avoiding code duplication
- Compile-time type checking with flexibility

---

### **Design Pattern Inspiration**

Templates implement **Generic Programming**, which is closely related to **Policy-based Design**, a form of the **Strategy Pattern** at compile time.

---

Would you like examples with template specialization or SFINAE (Substitution Failure Is Not An Error)?