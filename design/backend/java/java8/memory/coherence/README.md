Good question 👍 — **coherence in Java Garbage Collection (GC)** is not always explicitly called out in JVM docs, but the idea comes from **memory coherence and correctness** when the **mutator (application code)** and the **collector (GC)** both interact with the heap.

Let me unpack this for you.

---

# 🔹 What is Coherence in GC?

**Coherence** in garbage collection means:

> Ensuring that **mutator threads see a consistent view of memory** while the **collector may be moving, reclaiming, or updating objects**.

Without coherence, you could have:

* Dangling pointers (mutator using reclaimed object).
* Stale references (mutator accessing old location after object moved).
* Race conditions (mutator updates during GC move).

So **coherence = correctness of heap view across mutators and collector**.

---

# 🔹 How JVM Ensures Coherence

### 1. **Stop-the-World (STW) GC**

* Simplest case:

  * JVM **pauses all mutator threads**, runs GC, compacts/moves objects, updates references, then resumes mutators.
* Coherence guaranteed because only one party (GC) touches heap at a time.

---

### 2. **Concurrent/Incremental GC (e.g., G1, ZGC, Shenandoah)**

* Mutator and collector run **at the same time**.
* JVM must maintain coherence with special techniques:

#### 🔸 **Write Barriers**

* When mutator updates a reference, a **barrier** ensures GC sees the change.
* Example: card marking in G1 (mutator marks a memory card dirty if it updates an object reference).

#### 🔸 **Read Barriers**

* When mutator reads a reference, barrier ensures it sees the **current location** (especially in moving GCs like ZGC, Shenandoah).
* Example: if GC moves object, read barrier forwards pointer to new location.

#### 🔸 **Snapshot-at-the-Beginning (SATB)**

* Collector assumes the heap is frozen at GC start.
* Write barriers record old values when mutator overwrites references.
* Ensures GC doesn’t “miss” reachable objects.

#### 🔸 **Load-Value Barriers (LVB)**

* Used in **ZGC** and **Shenandoah**.
* Every load goes through a barrier that fixes references if object was relocated.

---

# 🔹 Analogy

Imagine you’re renovating a library (GC) while people are reading books (mutator):

* **STW GC**: Close the library → move shelves/books → reopen.
* **Concurrent GC**: Let people keep reading while moving shelves.

  * Need helpers (barriers) to ensure readers always find books in the right spot.

This “always find the right book” is **coherence**.

---

# 🔹 Code Example (for intuition)

```java
public class GCDemo {
    public static void main(String[] args) {
        Object obj = new Object();  // allocated in Eden (mutator action)

        // Hint GC
        System.gc();                // collector may move 'obj' to Survivor or Old

        // Coherence: even if 'obj' is moved, reference still points correctly
        System.out.println(obj.hashCode()); 
    }
}
```

Here:

* The **mutator** uses `obj`.
* The **collector** may relocate `obj`.
* JVM’s barriers ensure **coherent access** → the reference remains valid.

---

# 🔹 Summary

**Coherence in Java GC** means:

* Mutators always see a **consistent, correct view of objects**, even while GC runs.
* JVM ensures this via:

  * **Stop-the-World** (trivial coherence).
  * **Barriers (write/read/load)** in concurrent GCs.
  * **SATB snapshotting**.
  * **Forwarding pointers** for moved objects.

Without coherence, the program would crash or misbehave after GC moves or frees objects.

---

👉 Do you want me to **contrast coherence vs consistency vs safety** in GC? These are often confused but mean slightly different things in JVM memory management.
