In C++, the term **"vector"** is not a keyword but rather a class template provided by the Standard Template Library (STL) in the `<vector>` header. The `std::vector` is a dynamic array that can grow or shrink in size at runtime, offering a flexible and efficient way to store a sequence of elements. Unlike built-in arrays, `std::vector` manages its own memory and provides many useful member functions for manipulating the data.

Below, I'll explain `std::vector`, its key features, and provide code examples to demonstrate its usage.

---

### Key Features of `std::vector`
1. **Dynamic Size**: Automatically resizes as elements are added or removed.
2. **Contiguous Storage**: Elements are stored in contiguous memory, allowing random access in O(1) time.
3. **Memory Management**: Handles memory allocation and deallocation automatically (RAII).
4. **Type Safety**: Template-based, so it works with any data type (e.g., `int`, `string`, custom classes).
5. **Rich Interface**: Provides methods for insertion, deletion, iteration, and more.
6. **Capacity vs. Size**: Maintains a distinction between the number of elements (`size`) and allocated memory (`capacity`).

---

### Declaration and Syntax
To use `std::vector`, include the `<vector>` header:
```cpp
#include <vector>
```
Declare a vector:
```cpp
std::vector<Type> name;            // Empty vector
std::vector<Type> name(size);      // Vector with size elements, default-initialized
std::vector<Type> name(size, value); // Vector with size elements, initialized to value
std::vector<Type> name{init_list}; // Initialize with initializer list
```

---

### Common Member Functions
Here are some key methods of `std::vector`:
- **Size and Capacity**:
  - `size()`: Returns the number of elements.
  - `capacity()`: Returns the current allocated storage.
  - `empty()`: Checks if the vector is empty.
  - `reserve(n)`: Preallocates space for `n` elements.
  - `resize(n)`: Changes size to `n`, adding default elements or removing excess.
- **Element Access**:
  - `operator[]`: Accesses element at index (no bounds checking).
  - `at(index)`: Accesses element with bounds checking (throws `std::out_of_range` if invalid).
  - `front()`: Returns the first element.
  - `back()`: Returns the last element.
- **Modification**:
  - `push_back(value)`: Adds an element to the end.
  - `pop_back()`: Removes the last element.
  - `insert(pos, value)`: Inserts an element at the specified position.
  - `erase(pos)`: Removes an element or range.
  - `clear()`: Removes all elements.
- **Iterators**:
  - `begin()`, `end()`: Returns iterators to the start and past-the-end.
  - `rbegin()`, `rend()`: Reverse iterators.

---

### Code Examples

#### Example 1: Basic Vector Operations
```cpp
#include <iostream>
#include <vector>

int main() {
    // Declare a vector
    std::vector<int> vec;

    // Add elements
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    // Access elements
    std::cout << "First element: " << vec[0] << "\n";  // Output: 10
    std::cout << "Last element: " << vec.back() << "\n";  // Output: 30

    // Size and capacity
    std::cout << "Size: " << vec.size() << "\n";  // Output: 3
    std::cout << "Capacity: " << vec.capacity() << "\n";  // Output: >= 3

    // Iterate over elements
    std::cout << "Elements: ";
    for (int x : vec) {
        std::cout << x << " ";  // Output: 10 20 30
    }
    std::cout << "\n";

    return 0;
}
```

#### Example 2: Initialization and Modification
```cpp
#include <iostream>
#include <vector>

int main() {
    // Initialize with size and value
    std::vector<int> vec1(5, 100);  // 5 elements, all 100
    std::cout << "vec1: ";
    for (int x : vec1) std::cout << x << " ";  // Output: 100 100 100 100  делают
    std::cout << "\n";

    // Initialize with initializer list
    std::vector<int> vec2{1, 2, 3, 4, 5};
    std::cout << "vec2: ";
    for (int x : vec2) std::cout << x << " ";  // Output: 1 2 3 4 5
    std::cout << "\n";

    // Modify vector
    vec2.push_back(6);  // Add 6 to the end
    vec2.pop_back();    // Remove 6
    vec2.insert(vec2.begin() + 1, 10);  // Insert 10 at index 1
    std::cout << "Modified vec2: ";
    for (int x : vec2) std::cout << x << " ";  // Output: 1 10 2 3 4 5
    std::cout << "\n";

    return 0;
}
```

#### Example 3: Using Iterators
```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<std::string> words{"hello", "world", "cpp"};

    // Using iterators
    std::cout << "Words: ";
    for (auto it = words.begin(); it != words.end(); ++it) {
        std::cout << *it << " ";  // Output: hello world cpp
    }
    std::cout << "\n";

    // Reverse iteration
    std::cout << "Reverse: ";
    for (auto it = words.rbegin(); it != words.rend(); ++it) {
        std::cout << *it << " ";  // Output: cpp world hello
    }
    std::cout << "\n";

    return 0;
}
```

#### Example 4: Memory Management
```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec;

    // Reserve space to avoid reallocations
    vec.reserve(10);  // Allocates space for 10 elements
    std::cout << "Capacity after reserve: " << vec.capacity() << "\n";  // Output: >= 10
    std::cout << "Size: " << vec.size() << "\n";  // Output: 0

    // Add elements
    for (int i = 1; i <= 5; ++i) {
        vec.push_back(i);
    }

    // Resize
    vec.resize(3);  // Keep only first 3 elements
    std::cout << "After resize(3): ";
    for (int x : vec) std::cout << x << " ";  // Output: 1 2 3
    std::cout << "\n";

    // Clear
    vec.clear();
    std::cout << "After clear, size: " << vec.size() << "\n";  // Output: 0

    return 0;
}
```

#### Example 5: Custom Objects in Vector
```cpp
#include <iostream>
#include <vector>
#include <string>

class Person {
public:
    std::string name;
    int age;
    Person(std::string n, int a) : name(n), age(a) {}
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << "\n";
    }
};

int main() {
    std::vector<Person> people;
    people.emplace_back("Alice", 30);  // Constructs Person directly in vector
    people.emplace_back("Bob", 25);

    std::cout << "People:\n";
    for (const auto& p : people) {
        p.display();
    }
    // Output:
    // Name: Alice, Age: 30
    // Name: Bob, Age: 25

    return 0;
}
```

---

### Key Considerations
1. **Performance**:
   - Accessing elements (`[]`, `at`) is O(1).
   - `push_back` is amortized O(1) but may trigger reallocation if capacity is exceeded.
   - Insertion or deletion in the middle is O(n) due to shifting elements.
2. **Memory**:
   - When size exceeds capacity, `std::vector` reallocates memory (typically doubling capacity), which can be costly.
   - Use `reserve()` to preallocate memory if the size is known.
3. **Safety**:
   - Use `at()` instead of `[]` for bounds checking.
   - Iterators may become invalid after operations like `push_back` or `insert` if reallocation occurs.
4. **Alternatives**:
   - Use `std::array` for fixed-size arrays.
   - Use `std::list` or `std::deque` for frequent insertions/deletions in the middle.

---

### Why Use `std::vector`?
- **Flexibility**: Grows dynamically unlike fixed-size arrays.
- **Ease of Use**: Simplifies memory management and provides a rich API.
- **Compatibility**: Works seamlessly with STL algorithms (e.g., `std::sort`, `std::find`).
- **Type Safety**: Template-based, supports any type with proper copy/move semantics.

If you have a specific use case or need more advanced examples (e.g., with STL algorithms or custom allocators), let me know!