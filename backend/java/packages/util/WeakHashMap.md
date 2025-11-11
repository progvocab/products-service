`WeakHashMap` in Java is a **HashMap-like collection** where the **keys are stored as weak references** — meaning they **do not prevent garbage collection**.

---

### 🔍 **Key Concept**

If a key object in a `WeakHashMap` **has no strong references elsewhere** in the program, the **garbage collector can reclaim it**.
When that happens, the corresponding **map entry is automatically removed**.

---

### ⚙️ **Internal Working**

* Internally, `WeakHashMap` wraps each key in a `WeakReference<Key>`.
* The JVM’s **ReferenceQueue** tracks which weak references have been cleared.
* During GC or next access, cleared entries are removed from the map.

---

### 🧩 **Example**

```java
import java.util.WeakHashMap;

public class WeakHashMapExample {
    public static void main(String[] args) {
        WeakHashMap<Object, String> map = new WeakHashMap<>();
        Object key = new Object();

        map.put(key, "Weak Value");
        System.out.println("Before GC: " + map);

        key = null;  // Remove strong reference
        System.gc(); // Request GC

        // Wait a bit for GC to process
        try { Thread.sleep(100); } catch (InterruptedException e) {}

        System.out.println("After GC: " + map);
    }
}
```

**Output:**

```
Before GC: {java.lang.Object@5e91993f=Weak Value}
After GC: {}
```

---

### ⚡ **When to Use**

* Caching data that can be safely discarded when memory is low.
* Storing metadata for objects (like images, classloaders, etc.) where entries should vanish when the keys are no longer in use.

---

### 🚫 **Difference from HashMap**

| Feature        | HashMap                  | WeakHashMap             |
| -------------- | ------------------------ | ----------------------- |
| Key Reference  | Strong                   | Weak                    |
| Entry Lifetime | Until explicitly removed | Until key is GC’ed      |
| Prevents GC    | ✅ Yes                    | ❌ No                    |
| Use Case       | Regular storage          | Memory-sensitive caches |

---

💡 **In short:**

> `WeakHashMap` automatically removes entries when keys are garbage-collected, making it ideal for **memory-sensitive caching and weakly referenced data structures**.
