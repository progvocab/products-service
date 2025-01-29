### **Java 17 Features Not Available in Java 11**  
Java 17 is a **long-term support (LTS) release** that introduces several improvements over Java 11. Below are the key **new features** added in Java 17 that were **not available** in Java 11.  

---

## **1. Sealed Classes (`sealed` Keyword)**
- **Sealed classes** restrict which classes can extend them.  
- Helps **enforce inheritance rules** and improves code maintainability.  
- **Java 11:** No way to restrict which classes extend a given class.  
- **Java 17:** Use `sealed`, `permits`, and `non-sealed`.

### **Example:**
```java
// Sealed class with permitted subclasses
public sealed class Shape permits Circle, Rectangle { }

final class Circle extends Shape { }  // Allowed
non-sealed class Rectangle extends Shape { }  // Can be extended further
```
- `sealed`: Only specific subclasses can extend it.  
- `non-sealed`: Allows further extension.  
- `final`: Prevents further subclassing.  

---

## **2. Pattern Matching for `switch` (Preview Feature)**
- **Improves readability** by allowing pattern matching in `switch`.  
- Eliminates need for **explicit type casting**.  
- **Java 11:** Requires `if-else` checks and manual type casting.  
- **Java 17:** Matches patterns directly in `switch`.

### **Example:**
```java
static void process(Object obj) {
    switch (obj) {
        case String s -> System.out.println("String: " + s);
        case Integer i -> System.out.println("Integer: " + i);
        default -> System.out.println("Other: " + obj);
    }
}
```
- **No need** for `instanceof` and explicit casting.  
- **More concise and readable** than `if-else` chains.  

---

## **3. New `Record` Enhancements**  
- **Records** were introduced in Java 14 and became stable in Java 16.  
- Java 17 allows **records to implement interfaces** and **use sealed types**.

### **Example:**
```java
sealed interface Animal permits Dog, Cat { }

record Dog(String name) implements Animal { }
record Cat(String name) implements Animal { }
```
- **Simplifies data classes** with **immutable fields**.  

---

## **4. New Random Number Generator API**
- Introduces `RandomGenerator` interface for **better control** over random numbers.  
- **Java 11:** Used `java.util.Random`.  
- **Java 17:** Provides multiple **new algorithms**.

### **Example:**
```java
import java.util.random.RandomGenerator;

public class RandomExample {
    public static void main(String[] args) {
        RandomGenerator generator = RandomGenerator.of("L128X1024MixRandom");
        System.out.println(generator.nextInt(100)); // Random number between 0-99
    }
}
```
- More **efficient and customizable** random number generation.  

---

## **5. Strongly Encapsulated `java.lang.SecurityManager` (Deprecated)**
- **Java 17 removed `SecurityManager`** (it was present in Java 11).  
- Applications relying on `SecurityManager` must **find alternative security measures**.

---

## **6. Foreign Function & Memory API (Incubating)**
- Provides **direct access** to native memory without JNI.  
- **Java 11:** Needed `Unsafe` or JNI for native memory access.  
- **Java 17:** Introduces **safer, faster** ways to interact with native code.

### **Example:**
```java
try (MemorySegment segment = MemorySegment.allocateNative(100)) {
    MemoryAccess.setInt(segment, 42); // Store value 42
}
```
- **Faster than JNI** and provides better **memory safety**.  

---

## **7. Deprecation and Removal of Legacy Features**
### **7.1. Removal of Applet API**
- **Java 11:** Had `java.applet.Applet` (mostly unused).  
- **Java 17:** Completely removed Applets.  

### **7.2. Deprecation of `finalize()` Method**
- `finalize()` is now deprecated in Java 17.  
- Encourages using **try-with-resources** or **`Cleaner` API**.

---

## **8. Performance Improvements**
- **ZGC (Z Garbage Collector) & G1 Enhancements:**  
  - **Lower latency and better memory management**.  
  - **Better JIT optimizations** in JVM.  

---

## **Comparison: Java 11 vs Java 17**
| Feature | Java 11 | Java 17 |
|---------|--------|--------|
| **Sealed Classes** | ❌ | ✅ |
| **Pattern Matching in `switch`** | ❌ | ✅ |
| **Enhanced Records** | ❌ | ✅ |
| **New Random API** | ❌ | ✅ |
| **Removed `SecurityManager`** | ✅ | ❌ |
| **Foreign Function & Memory API** | ❌ | ✅ |
| **Better Garbage Collection (ZGC, G1)** | ✅ | ✅ (Improved) |
| **Removed Applet API** | ✅ | ❌ |
| **Deprecated `finalize()`** | ❌ | ✅ |

---

## **Conclusion**
- **Java 17 is a major improvement** over Java 11.  
- **Sealed classes, pattern matching, and record updates** enhance code maintainability.  
- **Better performance with new Random API and improved Garbage Collection**.  
- **Deprecated SecurityManager and removed Applet API** help modernize Java.  

Would you like help with **migrating from Java 11 to Java 17**? Let me know!