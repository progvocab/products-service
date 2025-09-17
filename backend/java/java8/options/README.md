

When you run a Java program with the `java` command, you can pass **JVM options** that change how the Java Virtual Machine behaves. These options fall into three big groups:

---

## 🔹 1. **Standard Options**

* Always available, documented, and stable.
* Examples:

```bash
java -version              # show Java version
java -classpath mylib:. MyApp   # set classpath
java -Dkey=value           # define a system property
java -jar myapp.jar        # run from JAR
```

---

## 🔹 2. **-X Options (Non-standard)**

* Non-standard but **commonly used** JVM options.
* Not guaranteed to be supported on all JVM versions.

Examples:

```bash
java -Xmx1024m   # set maximum heap size to 1024 MB
java -Xms512m    # set initial heap size to 512 MB
java -Xss256k    # set thread stack size
java -XshowSettings:all   # show JVM settings
```

👉 Use `java -X` to see available ones on your JVM.

---

## 🔹 3. **-XX Options (Advanced)**

* **Tuning and debugging** parameters.
* Internal JVM knobs, may change between versions.
* Two types:

  * `-XX:+Option` → enable
  * `-XX:-Option` → disable
  * `-XX:Option=value` → set a value

Examples:

```bash
java -XX:+UseG1GC             # use G1 garbage collector
java -XX:+PrintGCDetails      # print GC logs
java -XX:MaxPermSize=256m     # (Java 7) max permanent generation size
java -XX:MetaspaceSize=128m   # (Java 8+) metaspace size
java -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/dump.hprof
```

---

## 🔹 4. **Special Debug/Diagnostic Options**

Some `-XX` options require unlocking:

```bash
java -XX:+UnlockDiagnosticVMOptions -XX:+PrintFlagsFinal
```

This prints **all available JVM flags** and their values.

---

## 🔹 Summary Table

| Option Type  | Example                     | Purpose                              |
| ------------ | --------------------------- | ------------------------------------ |
| **Standard** | `-classpath`, `-Dkey=value` | Stable, common                       |
| **-X**       | `-Xmx1g`, `-Xms512m`        | Memory, performance tuning           |
| **-XX**      | `-XX:+UseG1GC`              | Advanced GC, debugging, experimental |

---

👉 In practice:

* Use `-X` options for **heap/stack size tuning**.
* Use `-XX` options for **GC tuning, debugging, advanced monitoring**.
* Use **standard options** for classpath, system properties, jar execution.

---

Would you like me to prepare a **cheat sheet of the most useful -X and -XX options for production Java apps** (memory, GC, troubleshooting)?
