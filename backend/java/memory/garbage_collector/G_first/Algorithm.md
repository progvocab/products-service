
G1GC is the **default collector** since Java 9, designed to handle **large heaps** (multi-GB) with predictable pause times.

---

## 🔹 G1 Garbage Collector Algorithm – Step by Step

### 1. **Heap Layout**

* The Java heap is divided into **regions** (not fixed Eden/Survivor/Old).
* Each region is typically **1MB–32MB**, chosen based on heap size.
* Roles:

  * **Eden** (new objects)
  * **Survivor** (objects surviving young collections)
  * **Old** (long-lived objects)
  * **Humongous** (large objects, >50% of a region)

---

### 2. **Phases of G1 GC Algorithm**

#### **(a) Young Collection (Minor GC)**

* New objects allocated in **Eden regions**.
* When Eden fills:
.
  1. Pause application threads (STW – Stop The World).
  2. Surviving objects copied → **Survivor regions**.
  3. If objects survive multiple cycles → promoted to **Old regions**.
* Uses **copying collection** (fast, compacts memory).

---

#### **(b) Concurrent Marking (Old Generation)**

* Triggered when **heap occupancy threshold** is crossed.
* Goal: Identify regions with **most garbage**.
* Steps:

  1. **Initial Mark** – mark roots (STW pause).
  2. **Concurrent Marking** – traverse object graph (parallel, no pause).
  3. **Remark** – catch missed references (short STW pause).
  4. **Cleanup** – calculate live data % in each region.

---

#### **(c) Mixed Collection**

* After marking, G1 does **mixed GCs**:

  * Collects **Eden + some Old regions** (those with high garbage %).
  * Prioritizes regions with **highest garbage-to-live ratio** (hence “Garbage-First”).
* Reclaims space efficiently, while meeting pause-time goals.

---

#### **(d) Humongous Object Handling**

* Large objects are allocated directly in **Humongous regions**.
* Collected during **Full GC** or **Mixed GC**.

---

#### **(e) Full GC (Fallback)**

* If concurrent marking cannot keep up (heap pressure too high), G1 falls back to a **Stop-the-World Full GC**.
* Uses **single-threaded mark-sweep-compact** (very costly, should be avoided).

---

## 🔹 G1GC Algorithm Flow (Mermaid)

```mermaid
flowchart TD
    A[Object Allocation in Eden] -->|Eden fills| B[Young GC]
    B --> C[Survivors moved to Survivor region]
    C --> D[Promotion to Old if age threshold met]
    D -->|Heap threshold crossed| E[Concurrent Marking]
    E --> F[Calculate Garbage % per region]
    F --> G[Mixed GC: Collect Eden + Old regions with most garbage]
    G --> H{Heap pressure too high?}
    H -- Yes --> I[Full GC (Stop-the-World)]
    H -- No --> A
```

---

## 🔑 Key Features of G1GC

* **Region-based** instead of contiguous generations.
* **Predictable pause times** (configurable via `-XX:MaxGCPauseMillis`).
* **Concurrent global marking** to avoid long STW pauses.
* Collects **"Garbage-First" regions** (most space reclaimed per pause).
* Avoids fragmentation with **copying compaction**.

---

⚡ So in short:

* Young collections → Copying GC.
* Old collections → Concurrent marking + region-based cleanup.
* Mixed collections → Garbage-first, reclaiming most garbage-heavy regions.

Would you like me to also show a **memory movement diagram (Eden → Survivor → Old with G1)** to complement this algorithm explanation?
