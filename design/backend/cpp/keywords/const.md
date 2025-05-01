The `const` keyword in **C++** is used to define **immutable values**—values that cannot be changed after initialization. It enhances safety and readability by clearly indicating which variables or parameters should not be modified.

---

### **Common Uses of `const` in C++**

#### 1. **Const Variables**
```cpp
const int x = 10;
// x = 20; // Error: cannot modify a const variable
```

#### 2. **Const Pointers**
- **Pointer itself is constant**:
  ```cpp
  int a = 10;
  int* const ptr = &a; // can't change ptr, but can change *ptr
  ```
- **Data pointed to is constant**:
  ```cpp
  const int* ptr = &a; // can change ptr, not *ptr
  ```
- **Both pointer and data are constant**:
  ```cpp
  const int* const ptr = &a;
  ```

#### 3. **Const Function Parameters**
```cpp
void print(const string& name) {
    // name = "Bob"; // Error
    cout << name;
}
```
Used for passing by reference safely.

#### 4. **Const Functions (Methods)**
```cpp
class Employee {
    string name;
public:
    string getName() const { return name; } // cannot modify members
};
```
Used to mark member functions that do **not modify** the object state.

#### 5. **Const Return Type**
```cpp
const string& getName() const {
    return name;
}
```
Prevents the returned value from being modified by the caller.

---

### **Why Use `const`?**

- **Prevents accidental modification**
- **Enables compiler optimizations**
- **Improves code readability and safety**
- **Works with overloading**:
  ```cpp
  void display();         // non-const version
  void display() const;   // const version
  ```

---

Would you like a diagram or a code demo showing multiple const scenarios together?