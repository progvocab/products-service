Here’s a concise list of **important JVM memory heuristics** (rules/strategies the JVM uses to manage memory) — with examples to clarify each:

---

### 🧠 **1. GC Trigger Heuristics**

* **What it means:** JVM decides *when* to trigger garbage collection based on heap usage.
* **Example:**
  If `Eden space` (young generation) fills up to 100%, a **Minor GC** is triggered.
  If **Old Generation** usage exceeds a threshold (e.g., 70–80%), a **Major GC** or **Full GC** may run.

---

### ⚖️ **2. Adaptive Sizing Heuristic**

* **What it means:** JVM automatically adjusts sizes of memory regions (Eden, Survivor spaces) for optimal GC performance.
* **Example:**
  `-XX:+UseAdaptiveSizePolicy` allows JVM to increase the Eden space if objects are short-lived or shrink it if too many survive GC.

---

### 🔁 **3. Tenuring Threshold Heuristic**

* **What it means:** Determines how many GC cycles an object survives before being promoted to the Old Generation.
* **Example:**
  `-XX:MaxTenuringThreshold=10` — an object surviving 10 Minor GCs gets promoted.
  JVM may lower this dynamically if the survivor space overflows.

---

### 🧹 **4. GC Algorithm Heuristic**

* **What it means:** JVM may change GC behavior or parallelism based on CPU cores and heap size.
* **Example:**
  G1 GC divides the heap into regions and prioritizes regions with most reclaimable garbage;
  Shenandoah or ZGC use concurrent heuristics to minimize pause times.

---

### 💾 **5. Heap Expansion/Contraction Heuristic**

* **What it means:** JVM grows or shrinks the heap based on GC performance.
* **Example:**
  If frequent GCs reclaim little memory, heap size grows (up to `-Xmx`).
  If heap is underutilized, JVM shrinks it (down to `-Xms`).

---

### 📊 **6. Pause-Time Goal Heuristic**

* **What it means:** GC tries to balance throughput and latency based on user goal.
* **Example:**
  `-XX:MaxGCPauseMillis=200` — JVM adjusts region size, GC frequency, and concurrency to keep GC pauses below 200 ms.

---

### 🧩 **7. Allocation Rate Heuristic**

* **What it means:** JVM observes how fast the application allocates objects and adjusts GC scheduling accordingly.
* **Example:**
  If allocation rate spikes, JVM may trigger GC earlier to prevent sudden heap exhaustion.

---

### 🧠 **8. Ergonomics Heuristic**

* **What it means:** JVM chooses default heap sizes, GC algorithms, and thread counts based on system hardware.
* **Example:**
  A server with 8 cores and 16 GB RAM → JVM may select G1GC and allocate `-Xmx` ≈ ¼ of total RAM automatically.

---

Would you like me to include a **diagram showing how these heuristics interact during GC cycles** (like a memory lifecycle view)?
