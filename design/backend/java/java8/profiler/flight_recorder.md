Let’s walk through the **same profiling demo using Java Flight Recorder (JFR)**, which is **low-overhead** and suitable for production.

---

# 🔹 Step 1: Write the Same Demo Program

We’ll use the same program as before:

```java
public class ProfilerDemoJFR {
    public static void main(String[] args) {
        while (true) {
            slowMethod();
            fastMethod();
        }
    }

    public static void slowMethod() {
        for (int i = 0; i < 1000000; i++) {
            Math.sqrt(i);
        }
    }

    public static void fastMethod() {
        String s = "";
        for (int i = 0; i < 1000; i++) {
            s += "x";
        }
    }
}
```

Compile:

```bash
javac ProfilerDemoJFR.java
```

---

# 🔹 Step 2: Run with JFR Enabled

Start the program with **JFR recording** enabled for 30 seconds:

```bash
java -XX:StartFlightRecording=duration=30s,filename=recording.jfr -jar ProfilerDemoJFR.jar
```

* `duration=30s` → record for 30 seconds.
* `filename=recording.jfr` → output file.

👉 While running, JFR captures **CPU usage, memory allocation, GC, threads, exceptions, I/O**.

---

# 🔹 Step 3: Analyze the Recording

When the program ends (or after 30s), you’ll have `recording.jfr`.

Now you can analyze it:

### Option 1: Java Mission Control (GUI)

1. Install **Java Mission Control (JMC)**: [https://www.oracle.com/java/technologies/javamissioncontrol.html](https://www.oracle.com/java/technologies/javamissioncontrol.html).
2. Open `recording.jfr` in JMC.
3. Navigate through tabs:

   * **Method Profiling** → See `slowMethod()` consuming CPU.
   * **Memory → Allocations** → See `String` and `char[]` from `fastMethod()`.
   * **GC & Heap** → Monitor garbage collections.

### Option 2: Command-line (Quick Check)

JFR also supports summaries via CLI (Java 14+):

```bash
jfr summary recording.jfr
```

You’ll see top events, CPU, allocations, and GC stats.

---

# 🔹 Step 4: Optimization Insight

* **CPU bottleneck** → `slowMethod()` dominates.
* **Memory inefficiency** → `fastMethod()` shows repeated string concatenations.
* JFR highlights **GC pressure** due to unnecessary allocations.

✅ Replace string concatenation with `StringBuilder`.
✅ Reduce heavy computation in `slowMethod`.

---

📌 **Summary**:

* **VisualVM** → great for dev, interactive debugging.
* **JFR + JMC** → better for production (low overhead, detailed recording).

---

👉 Do you want me to also show how to **continuously run JFR in the background on a production app** (like always recording last 5 minutes for troubleshooting)?
