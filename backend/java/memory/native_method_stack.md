The **Native Method Stack** is often less talked about compared to Heap or Thread Stack, but it’s very important. Let’s break it down:

---

## 🔹 What is the Native Method Stack?

* In Java, most methods are **Java methods** and use the **Java Thread Stack**.
* But some methods are written in **native languages** (like C or C++), often for **JVM internals or system-level operations**.
* When a **native method** is invoked, the JVM uses the **Native Method Stack**.

---

## 🔹 Key Points

1. **Storage**

   * Holds local variables, parameters, return values, and intermediate computations for **native (non-Java) methods**.
   * Similar to the Java thread stack but for **native code execution**.

2. **When It’s Used**

   * Whenever Java calls a method via **JNI (Java Native Interface)** or other JVM native hooks.
   * Example: `System.loadLibrary("xyz")` loads a native library.

3. **Error Handling**

   * If it overflows → `StackOverflowError`.
   * If memory can’t be allocated → `OutOfMemoryError`.

4. **Relation with Threads**

   * Each thread has **its own native method stack**, just like it has its own Java stack.

---

## 🔹 Example

```java
public class NativeExample {
    static {
        System.loadLibrary("nativeLib");
    }

    // Declaration of a native method
    public native void nativePrint();

    public static void main(String[] args) {
        new NativeExample().nativePrint();
    }
}
```

Here:

* `nativePrint()` is declared in Java.
* The implementation exists in C/C++ via JNI.
* When called, execution moves from the **Java Thread Stack → Native Method Stack → Native code**.

---

## 🔹 Mermaid Diagram: Native Method Stack in Action

```mermaid
flowchart TD
    A[Java Code] --> B[Java Thread Stack]
    B --> C{Native Method Call?}
    C -- Yes --> D[Native Method Stack]
    D --> E[Native Code (C/C++) via JNI]
    C -- No --> F[Continue in Java]
```

---

✅ **In short**: The **Native Method Stack** is a special stack used by threads when executing **native (non-Java) methods**, typically via JNI. It keeps track of native calls, local variables, and return values just like the Java stack does for Java methods.

---

Would you like me to also compare **Java Thread Stack vs Native Method Stack** in a **side-by-side table** for interview prep?
