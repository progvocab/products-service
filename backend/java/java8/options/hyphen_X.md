You want the **most important `-X` JVM options** — the ones you’ll see in production configs or troubleshooting.

The `-X` flags are **non-standard JVM options** (not guaranteed across all JVMs, but stable in Oracle/OpenJDK).

---

# 🔹 Top 10 `java -X` Options

### 1. **`-Xmx<size>` → Maximum Heap Size**

Sets the **maximum Java heap memory**.

```bash
java -Xmx1024m MyApp
```

* Example: `-Xmx2g` (2 GB max heap)
* Important for avoiding **OutOfMemoryError**.

---

### 2. **`-Xms<size>` → Initial Heap Size**

Sets the **starting heap memory**.

```bash
java -Xms512m MyApp
```

* Example: `-Xms512m` (start with 512 MB heap).
* Usually set **equal to `-Xmx`** in production for stability.

---

### 3. **`-Xss<size>` → Thread Stack Size**

Defines memory per thread stack.

```bash
java -Xss256k MyApp
```

* Default \~1 MB. Lowering allows **more threads** but risks `StackOverflowError`.

---

### 4. **`-XshowSettings` → Show JVM Settings**

Prints useful info: classpath, memory, properties.

```bash
java -XshowSettings:all -version
```

* Modes: `all`, `vm`, `system`, `properties`.
* Handy for debugging environment issues.

---

### 5. **`-Xint` → Interpreted Mode Only**

Runs JVM in **interpreter-only** mode (no JIT).

```bash
java -Xint MyApp
```

* Very slow, but useful for debugging JIT issues.

---

### 6. **`-Xcomp` → Compile All Methods Immediately**

Forces **JIT compilation** at startup.

```bash
java -Xcomp MyApp
```

* Stress test for JIT compiler.

---

### 7. **`-Xmixed` → Mixed Mode (Default)**

Runs with both **interpreter + JIT compiler**.

```bash
java -Xmixed MyApp
```

* Default mode: balance of startup time + performance.

---

### 8. **`-Xloggc:<file>` → GC Log File**

Sends **Garbage Collection logs** to a file.

```bash
java -Xloggc:/var/log/myapp_gc.log MyApp
```

* Useful for performance tuning and memory troubleshooting.

---

### 9. **`-Xnoclassgc` → Disable Class Garbage Collection**

Prevents unloading of unused classes.

```bash
java -Xnoclassgc MyApp
```

* Rarely used; can cause **metaspace leaks**.

---

### 10. **`-Xcheck:jni` → JNI Error Checking**

Checks for **JNI (Java Native Interface) errors**.

```bash
java -Xcheck:jni MyApp
```

* Slows execution but catches **native integration bugs**.

---

# 🔹 Quick Reference Table

| Option           | Purpose                   |
| ---------------- | ------------------------- |
| `-Xmx<size>`     | Max heap size             |
| `-Xms<size>`     | Initial heap size         |
| `-Xss<size>`     | Thread stack size         |
| `-XshowSettings` | Show JVM settings         |
| `-Xint`          | Interpreter-only mode     |
| `-Xcomp`         | Compile all methods (JIT) |
| `-Xmixed`        | Mixed mode (default)      |
| `-Xloggc:file`   | Log GC events             |
| `-Xnoclassgc`    | Disable class GC          |
| `-Xcheck:jni`    | Extra JNI checks          |

---

👉 In **production**, the most common ones are:

* `-Xmx` / `-Xms` (heap size tuning)
* `-Xss` (stack size for many threads)
* `-Xloggc` (GC logging)
* `-XshowSettings` (debug startup issues)

---

Do you want me to also make a **cheat sheet for top 10 `-XX` options** (GC tuning, heap dumps, performance)?
