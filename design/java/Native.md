Great question! Identifying the **native method implementations** in OpenJDK involves a mix of reading the **Java API source** and then tracing into the **HotSpot JVM's C++ source**. Here’s a step-by-step guide to help you locate where a native method (like `hashCode`, `wait`, `notify`, etc.) is implemented:

---

### **1. Find the Java declaration**

Let’s use `hashCode()` from `java.lang.Object` as an example:

- GitHub: [`java/lang/Object.java`](https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/lang/Object.java)

You’ll see:

```java
@IntrinsicCandidate
public native int hashCode();
```

This tells you:
- It's a **native** method (implemented outside Java),
- Possibly has **JVM-level optimizations** (`@IntrinsicCandidate`),
- Implemented somewhere in **HotSpot (C++)**.

---

### **2. Look for the JNI mapping or intrinsic name**

For standard classes like `java.lang.Object`, the JVM handles certain methods **specially** and does not use standard JNI. These are built-in **intrinsics**.

- For non-intrinsic native methods (like `System.currentTimeMillis()`), you’ll find their mappings in `jni.cpp` or `native*` files.
- For intrinsic or tightly bound native methods, like `Object.hashCode`, you’ll find the C++ implementation in:

```
src/hotspot/share/runtime/
src/hotspot/share/classfile/
src/hotspot/share/oops/
```

For example:
- `Object.hashCode()` → [ObjectSynchronizer::FastHashCode()](https://github.com/openjdk/jdk/blob/master/src/hotspot/share/runtime/synchronizer.cpp)
- `Thread.start()` → [JavaThread::start()](https://github.com/openjdk/jdk/blob/master/src/hotspot/share/runtime/thread.cpp)

---

### **3. Search the JVM source code**

#### GitHub Search:

Search in the [OpenJDK GitHub repository](https://github.com/openjdk/jdk) for something like:

```
FastHashCode
```

or

```
JVM_HashCode
```

You’ll find entries like:

```cpp
JVM_ENTRY(jint, JVM_HashCode(JNIEnv* env, jobject handle))
```

File: [`jvm.cpp`](https://github.com/openjdk/jdk/blob/master/src/hotspot/share/prims/jvm.cpp)

---

### **4. Understand JVM_* naming**

In many cases, native methods are implemented in C++ using a naming convention like:

```cpp
JVM_MethodName()
```

Examples:

| Java Method | C++ Native Symbol |
|-------------|--------------------|
| `Object.hashCode()` | `JVM_HashCode()` |
| `System.gc()` | `JVM_GC()` |
| `Thread.start()` | `JVM_StartThread()` |

Search for `JVM_` in `src/hotspot/share/prims/jvm.cpp`.

---

### **5. Special Case: Intrinsics**

For `@IntrinsicCandidate` methods, they're optimized by HotSpot during compilation. You can see which methods are intrinsic by checking:

- File: [`vmSymbols.hpp`](https://github.com/openjdk/jdk/blob/master/src/hotspot/share/classfile/vmSymbols.hpp)
- Look for `VM_SYMBOLS_DO` and search for `hashCode`, `equals`, etc.

These help the compiler recognize these methods and apply intrinsic replacements.

---

### **6. Bonus: JNI-registered native methods**

For non-intrinsic methods, check:

- `src/java.base/share/native/libjava/`
- `src/java.base/unix/native/libjava/`
- `src/java.base/windows/native/libjava/`

Example: `System.currentTimeMillis()` is in `System.c`.

---

### TL;DR Summary

| Step | What to Do |
|------|------------|
| 1. | Locate the native method in Java (look for `native` keyword) |
| 2. | Check if it’s `@IntrinsicCandidate` (JVM handles directly) |
| 3. | Look in `hotspot/share/prims/jvm.cpp` for `JVM_*` functions |
| 4. | Search the repo with method name or `JVM_MethodName` |
| 5. | Explore `vmSymbols.hpp` for intrinsics |
| 6. | For JNI-style natives, explore `libjava` C source directories |

---

If you want, I can help you trace a specific method step-by-step in the OpenJDK source. Just name it!

Great question!

The `@IntrinsicCandidate` annotation and the method:

```java
@IntrinsicCandidate
public native int hashCode();
```

refer to **methods whose implementations are provided by the JVM itself**, typically in **native code written in C/C++**, not in Java source code. Let's walk through what this means.

---

### **1. `@IntrinsicCandidate` Annotation**

This is an **internal JDK annotation** used to mark methods that:
- Are **"intrinsic"**, meaning they may be **optimized by the JIT compiler** (like C2 in HotSpot).
- Often correspond to **native methods**, and their bodies are implemented **directly by the JVM** or through **native libraries**.

You’ll find this used in core classes like `Object`, `String`, etc.

---

### **2. `public native int hashCode()`**

This is in `java.lang.Object`:

```java
@IntrinsicCandidate
public native int hashCode();
```

- `native` keyword = This method is **not implemented in Java**, but in **native code** (C/C++).
- This means you won’t find a `.java` implementation of `hashCode()` inside the JDK source.
- Instead, the implementation is inside the **HotSpot JVM**, typically in the **C++ files** under OpenJDK.

---

### **3. Where is the native implementation?**

In OpenJDK, the native implementation lives in HotSpot:

- File: [`object.cpp`](https://github.com/openjdk/jdk/blob/master/src/hotspot/share/oops/oop.cpp)  
- Function: It usually uses the object's **identity hash**, stored in its header or generated.

Look for `ObjectSynchronizer::FastHashCode()` and related code in:
```
src/hotspot/share/runtime/synchronizer.cpp
```

Also relevant:
```cpp
intptr_t ObjectSynchronizer::FastHashCode(Thread* current, oop obj);
```

---

### **4. Why this matters**

- You can’t override or rewrite the behavior of `Object.hashCode()` because it's **tied deeply to object identity and memory layout**.
- The JVM may **inline or intrinsify** this method for performance.

---

### Summary

| Item | Description |
|------|-------------|
| `@IntrinsicCandidate` | Marks methods for potential intrinsic optimization by the JVM. |
| `native int hashCode()` | Implemented in C++ inside the JVM (not in Java source). |
| Actual implementation | Found in HotSpot source (`synchronizer.cpp`, `oop.cpp`). |
| Purpose | Provides a fast, identity-based hash for objects. |

Would you like a walkthrough of how to trace this in the OpenJDK GitHub source or even compile a minimal OpenJDK to inspect it?
