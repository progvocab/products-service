## **G1 Garbage Collector (G1 GC)**
- has been the **default collector since Java 9**.

---

# 🔹 1. Why G1 GC?

Before G1, Java mostly used:

* **Parallel GC (Throughput collector)** → Fast, but had **long Stop-the-World (STW) pauses**.
* **CMS (Concurrent Mark-Sweep)** → Lower pauses, but suffered from **fragmentation** and unpredictable compaction.

👉 G1 GC was designed to **combine the best of both**:

* **Predictable pause times** (low-latency, configurable).
* **Parallel + concurrent work**.
* **Region-based design** to avoid fragmentation.

---

# 🔹 2. Heap Layout (Partitioning into Regions)

Unlike traditional collectors (Young vs Old continuous spaces), G1 divides the heap into many **equal-sized regions** (1MB–32MB each, depending on heap size).

```
Heap
+----+----+----+----+----+----+----+
| R1 | R2 | R3 | R4 | R5 | R6 | R7 | ...
+----+----+----+----+----+----+----+

Regions can be:
- Eden (new objects)
- Survivor (surviving young objects)
- Old (long-lived objects)
- Humongous (large objects spanning regions)
```

This **partitioning avoids fragmentation** since objects can be relocated between regions.

---

# 🔹 3. G1 GC Phases

### **(A) Young GC (Minor GC)**

* Happens when Eden fills.
* **Stop-the-world (but parallel):**

  * Live objects copied from Eden → Survivor or Old regions.
  * Dead objects discarded.
* Multiple threads evacuate regions in parallel.

---

### **(B) Concurrent Marking**

* Runs **concurrently** with application threads.
* Identifies live objects in Old regions.
* Uses SATB (Snapshot-At-The-Beginning) algorithm → sees all objects reachable at start of marking.

---

### **(C) Remark (short STW)**

* Fixes up references changed during concurrent marking.
* Short pause compared to CMS.

---

### **(D) Mixed GC**

* After marking, G1 knows which Old regions have most garbage.
* Performs **Mixed Collections**:

  * Collects Eden + some Old regions in the same cycle.
  * Prioritizes regions with **most garbage** → hence the name **Garbage First (G1)**.

---

### **(E) Compaction**

* G1 **always evacuates live objects into new regions**, so compaction is **built-in**.
* No fragmentation problem like CMS.

---

# 🔹 4. Pause Time Goal

You can **tune pause times** with:

```bash
-XX:MaxGCPauseMillis=200
```

* G1 tries to meet this pause goal by selecting the right number of regions to collect.
* Predictable latency, even for **large heaps (tens of GBs)**.

---

# 🔹 5. Humongous Objects

* Objects larger than **50% of a region size** are stored in **Humongous regions** (spanning multiple contiguous regions).
* G1 tries to clean them, but very large objects can still cause allocation pressure.

---

# 🔹 6. G1 vs Others

| Feature            | Parallel GC            | CMS                     | **G1 GC**                  |
| ------------------ | ---------------------- | ----------------------- | -------------------------- |
| Heap layout        | Young + Old contiguous | Young + Old contiguous  | Region-based               |
| Compaction         | STW compaction         | No (fragmentation risk) | Always (region evacuation) |
| Concurrency        | No                     | Yes (mark+sweep)        | Yes (mark + mixed GC)      |
| Pause time control | No                     | Limited                 | Yes (`MaxGCPauseMillis`)   |
| Best for           | Throughput             | Lower pauses            | Predictable latency        |

---

# 🔹 7. Example JVM Options

```bash
java -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -Xmx4g MyApp
```

* Enables G1 GC.
* Sets target max pause time to 200ms.
* Heap size = 4GB.

---

# 🔹 8. Analogy

Think of G1 as a **city garbage service**:

* City = Heap.
* Divided into **districts (regions)**.
* Garbage trucks (GC threads) **first clean the dirtiest districts** (regions with most garbage).
* This keeps the city livable without shutting it down completely.

---

✅ **Summary**

* G1 GC = **Region-based, concurrent, compacting, low-pause collector**.
* Always avoids fragmentation.
* Balances **throughput + latency**.
* Default since Java 9.

---

Would you like me to also draw a **timeline diagram of G1 GC phases** (showing where mutator threads run vs where STW pauses happen)?


Great 👍 Let’s visualize the **timeline of G1 GC phases** showing how **application threads (mutators)** and **GC threads** interleave.

---

# 🔹 G1 GC Timeline Diagram

```
Time ──────────────────────────────────────────────────────────────────────────►

   Application (Mutator Threads)
   ──────────────────────────────██████████████████████████████████████████
                                   ▲         ▲         ▲           ▲
                                   |         |         |           |
                               short STW  short STW  short STW   short STW
                               (Young GC) (Remark)   (Young GC)  (Mixed GC)

   Garbage Collector (GC Threads)
   ───────────────────────────────────────────────────────────────
       Young GC (STW, parallel)
                   │
                   └──> Concurrent Marking (runs while app continues)
                             │
                             └──> Remark (short STW)
                                     │
                                     └──> Cleanup + Mixed GC (STW, parallel)
```

---

# 🔹 Explanation of Timeline

1. **Young GC (Minor GC)**

   * Stop-the-world (STW), parallel.
   * Live objects copied from Eden → Survivor/Old.
   * Pause is **short** since regions are small.

2. **Concurrent Marking**

   * Runs **alongside the application**.
   * GC threads walk the heap to mark live objects in Old Gen.
   * Mutator threads keep running.

3. **Remark**

   * Short STW pause to catch up on changes made during marking.
   * Much shorter than CMS’s remark phase.

4. **Mixed GC**

   * Collects both Young regions + some Old regions with most garbage.
   * Still STW, but parallel and short.
   * Compaction happens automatically during evacuation → no fragmentation.

---

# 🔹 Key Points in Diagram

* Application threads mostly run **uninterrupted**.
* STW pauses happen but are **short and predictable**.
* Concurrent work (marking) reduces long pauses.
* G1 selects “garbage-first” regions → keeps pauses within `-XX:MaxGCPauseMillis`.

---

✅ So the timeline shows **mutators and GC interleaving**:

* **STW phases are short and infrequent.**
* **Concurrent phases happen in the background.**

---

Would you like me to also draw a **heap diagram with regions (Eden, Survivor, Old, Humongous)** to show how G1 moves objects around during these phases?
