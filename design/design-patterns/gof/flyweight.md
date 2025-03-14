### **Flyweight Pattern (Structural Design Pattern)**
The **Flyweight Pattern** is a **memory optimization technique** used to minimize resource usage when creating a large number of objects. Instead of **storing duplicate data in multiple objects**, it **shares common data between multiple instances**.

This pattern is useful when:
- A system needs to create **a large number of similar objects**.
- Object creation and memory consumption **are expensive**.
- There is a **lot of shared intrinsic state**.

---

## **🔹 Key Concepts in Flyweight Pattern**
1. **Intrinsic State** – The part of the object that **can be shared** (e.g., common properties).  
2. **Extrinsic State** – The part of the object that **varies per instance** and is provided externally.  
3. **Flyweight Factory** – Manages a pool of shared objects and **returns existing instances** instead of creating new ones.  

---

## **🔹 Example: Flyweight Pattern in Python**
Let's say we are developing a **text editor** that needs to handle **millions of characters** efficiently.

### **Step 1: Define the Flyweight Class**
```python
class Character:
    """Flyweight: Stores shared (intrinsic) state."""
    
    _instances = {}  # Flyweight Factory: Stores shared objects

    def __new__(cls, char):
        if char not in cls._instances:
            cls._instances[char] = super(Character, cls).__new__(cls)
            cls._instances[char].char = char  # Intrinsic state (shared)
        return cls._instances[char]

    def display(self, font_size, color):
        """Extrinsic state (provided externally)."""
        print(f"Displaying '{self.char}' with size {font_size} and color {color}")


# Step 2: Create Characters (Shared Instances)
char_a1 = Character("A")
char_a2 = Character("A")
char_b = Character("B")

# Step 3: Check if "A" instances are shared
print(char_a1 is char_a2)  # ✅ True (Same object shared)

# Step 4: Display characters with different external properties
char_a1.display(12, "red")  
char_a2.display(18, "blue")  
char_b.display(14, "green")  
```

### **🔹 Explanation:**
- **Character instances are shared** (char `"A"` is created only once).
- **Extrinsic properties (`font_size`, `color`) are provided dynamically**.
- **Memory usage is reduced** because duplicate objects are avoided.

---

## **🔹 Real-World Use Cases**
✅ **Text editors** (sharing font, style, and characters).  
✅ **Game development** (storing similar sprites, trees, enemies).  
✅ **Database connection pooling** (reusing database connections).  
✅ **Caching systems** (avoiding duplicate objects in memory).  

---

## **🔹 When to Use the Flyweight Pattern?**
| **Scenario** | **Use Flyweight?** |
|-------------|------------------|
| Large number of similar objects | ✅ Yes |
| Objects contain **shared** data | ✅ Yes |
| High memory consumption | ✅ Yes |
| Objects have **unique, independent data** | ❌ No |

Would you like an example in **Java, C++, or another language**? 🚀