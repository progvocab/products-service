 # Metaspace

**PermGen was removed in Java 8 and replaced by Metaspace**


---

**Metaspace** is the memory region introduced in **Java 8** to replace the old **PermGen (Permanent Generation)**. It is used to store **class metadata** (information about classes, methods, fields, bytecode, etc.). Unlike PermGen, which was part of the JVM heap, Metaspace resides in **native memory (outside the Java heap)** and grows dynamically as needed, limited only by the available system memory unless capped.

* **Introduced in**: Java 8 (PermGen removed)
* **Purpose**: Store class metadata
* **Location**: Native memory (not part of heap)
* **Configuration**: Controlled with `-XX:MetaspaceSize` (initial) and `-XX:MaxMetaspaceSize` (maximum)
* **Error on exhaustion**: `java.lang.OutOfMemoryError: Metaspace`

## Main advantage:
 Developers don’t need to guess/tune class metadata size as strictly as with PermGen; Metaspace grows automatically.


---

# 🔹 1. What They Are

| Aspect               | **PermGen (Permanent Generation)**                                                                                 | **Metaspace**                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Location             | Inside the **Java Heap** (part of JVM heap memory, managed by GC)                                                  | Outside the Java Heap, in **native memory**                                               |
| Purpose              | Stores metadata about classes: class structure, methods, fields, constant pool, interned Strings, static variables | Stores the same kind of **class metadata** and runtime constant pool, but not in the heap |
| Introduced / Removed | Existed until Java 7                                                                                               | Introduced in **Java 8** (PermGen removed)                                                |

---

# 🔹 2. Problems with PermGen

1. **Fixed maximum size**

   * Controlled by `-XX:PermSize` and `-XX:MaxPermSize`.
   * If too small → `java.lang.OutOfMemoryError: PermGen space`.

2. **Hard to tune**

   * Developers often had to guess the right size depending on number of classes loaded.

3. **Classloader leaks**

   * In web apps (Tomcat, JBoss, etc.), redeploying could leave classes stuck in PermGen (classloader references not garbage-collected).

---

# 🔹 3. Why Metaspace is Better

* **No fixed max size by default** → it grows dynamically, limited only by **native memory (OS limits)**.
* **Class metadata storage** is decoupled from the heap → reduces GC pressure on the Java heap.
* Easier for developers → no more `PermGen` OOM unless native memory runs out.

---

# 🔹 4. Tuning Parameters

| Parameter    | PermGen (Java 7 and below)                  | Metaspace (Java 8+)                                                                         |
| ------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Initial size | `-XX:PermSize`                              | `-XX:MetaspaceSize`                                                                         |
| Max size     | `-XX:MaxPermSize`                           | `-XX:MaxMetaspaceSize`                                                                      |
| GC trigger   | Based on usage vs. `PermSize`/`MaxPermSize` | When class metadata exceeds `MetaspaceSize`, GC is triggered; size may increase dynamically |

---

# 🔹 5. What Goes Inside

### PermGen contents:

* Class metadata (methods, fields, bytecode)
* Method metadata
* Interned Strings (`String.intern()`)
* Static variables

### Metaspace contents:

* Class metadata (methods, fields, bytecode)
* Method metadata
* **No interned Strings** (moved to Java Heap in Java 7 update 6+)
* **No static variables** (stored in the heap with their classes)

---

# 🔹 6. GC Behavior

* **PermGen**: Collected during full GC, but limited by heap space.
* **Metaspace**: Collected when classloaders become unreachable. GC can free associated metadata, and memory can be returned to the OS.

---

# 🔹 7. Example Errors

* **Java 7 and below**:

  ```
  java.lang.OutOfMemoryError: PermGen space
  ```
* **Java 8+ (Metaspace)**:

  ```
  java.lang.OutOfMemoryError: Metaspace
  ```

---

# 🔹 8. Summary Table

| Feature         | **PermGen (≤ Java 7)**                        | **Metaspace (≥ Java 8)**                                   |
| --------------- | --------------------------------------------- | ---------------------------------------------------------- |
| Location        | Heap memory (inside JVM)                      | Native memory (outside JVM heap)                           |
| Default Max     | Fixed, small (e.g., 64–85 MB)                 | Unlimited (depends on system memory)                       |
| Config Options  | `-XX:PermSize`, `-XX:MaxPermSize`             | `-XX:MetaspaceSize`, `-XX:MaxMetaspaceSize`                |
| Stores          | Class metadata, interned Strings, static vars | Class metadata only (Strings + statics moved to heap)      |
| Common Error    | `OutOfMemoryError: PermGen space`             | `OutOfMemoryError: Metaspace`                              |
| Redeploy Issues | Frequent leaks due to classloaders            | Reduced, but classloader leaks can still consume Metaspace |

---

✅ **In short**:

* **PermGen** was rigid, inside the heap, and prone to OOM.
* **Metaspace** is flexible, native memory-based, and easier to manage.

---

Would you like me to also **draw a diagram comparing JVM memory areas (Heap, PermGen, Metaspace, Native memory, Stack)** so it’s visually clear?
