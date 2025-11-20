

# Java Garbage Collector 

| Java Version        | Default Garbage Collector | New Garbage Collectors Introduced                    |
| ------------------- | ------------------------- | ---------------------------------------------------- |
| **Java 1.2 (1998)** | Serial GC                 | First introduction of **Generational GC** (Serial)   |
| **Java 1.3**        | Serial GC                 | –                                                    |
| **Java 1.4**        | Serial GC                 | **Parallel GC** (young gen parallelization)          |
| **Java 5 (1.5)**    | Parallel GC               | **Concurrent Mark-Sweep (CMS) GC**                   |
| **Java 6**          | Parallel GC               | Parallel Old GC (parallel compaction for old gen)    |
| **Java 7**          | Parallel GC               | **G1 GC** (experimental)                             |
| **Java 8**          | Parallel GC               | G1 officially supported & stable                     |
| **Java 9**          | G1 GC                     | –                                                    |
| **Java 10**         | G1 GC                     | **Experimental ZGC**                                 |
| **Java 11 (LTS)**   | G1 GC                     | **ZGC (improved)**, **Epsilon GC** (no-op collector) |
| **Java 12**         | G1 GC                     | **Shenandoah GC** (experimental, Red Hat)            |
| **Java 13**         | G1 GC                     | Improvements to ZGC (support for macOS & Windows)    |
| **Java 14**         | G1 GC                     | ZGC goes **production-ready**                        |
| **Java 15**         | G1 GC                     | **Shenandoah GC production-ready**                   |
| **Java 16**         | G1 GC                     | ZGC & Shenandoah fully optimized                     |
| **Java 17 (LTS)**   | G1 GC                     | Nothing new (ZGC & Shenandoah stable)                |
| **Java 18**         | G1 GC                     | Minor improvements in ZGC                            |
| **Java 19**         | G1 GC                     | **Generational ZGC (experimental)**                  |
| **Java 20**         | G1 GC                     | Improvements to Generational ZGC                     |
| **Java 21 (LTS)**   | G1 GC                     | **Generational ZGC (production-ready)**              |
| **Java 22**         | G1 GC                     | Refinements in ZGC, Shenandoah                       |
| **Java 23**         | G1 GC                     | Ongoing refinements only                             |



* **Serial GC** → earliest, single-threaded.
* **Parallel GC** → throughput-focused (multi-threaded stop-the-world).
* **CMS** → first low-latency concurrent collector.
* **G1 GC** → default since Java 9 (balanced throughput + low pause).
* **ZGC** & **Shenandoah** → ultra-low latency, scalable to **multi-TB heaps**.
* **Epsilon GC** → "no-op" GC for testing / benchmarking.
* **Generational ZGC** (Java 21) → combines generational model + ZGC’s low-latency design.


## GC Algorithms



| Garbage Collector                       | Algorithms Used                                                      | Notes                                       |
| --------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------- |
| Serial GC                               | Mark-Sweep-Compact                                                   | Single-threaded, for small heaps            |
| Parallel GC (Throughput GC)             | Parallel Mark-Sweep-Compact                                          | Multi-threaded young and old generation GC  |
| CMS (Concurrent Mark Sweep, deprecated) | Concurrent Mark-Sweep                                                | Low-pause but fragmentation issues          |
| G1 GC                                   | Region-based + Concurrent Marking + Copying + Predictive Pause Model | Default since Java 9                        |
| ZGC                                     | Region-based + Load Barriers + Colored Pointers + Concurrent Marking | Ultra-low pause (<10ms), scalable           |
| Shenandoah GC                           | Region-based + Brooks Pointers + Concurrent Evacuation               | Low pause with concurrent compaction        |
| Epsilon GC                              | No-op (no GC)                                                        | For performance testing or short-lived apps |



Stop-the-World (STW) pauses are moments when the JVM suspends all application threads so that the Garbage Collector can safely perform operations that cannot tolerate changes in the heap or thread stacks.

Here is the concise explanation:

What is an STW pause?

An STW pause is a point during garbage collection when the JVM halts every running Java thread—including user threads, worker threads, schedulers, and internal JVM threads—so the GC can perform tasks that require a consistent, unchanging memory state.

Why does GC need STW pauses?

Some GC operations cannot run concurrently, such as:

Finding GC roots (stack, registers, thread-local references)

Ensuring reference graphs don't change during marking

Compacting or relocating objects

Updating pointers after moves

Verifying heap correctness


These actions require a moment when no thread is modifying memory, hence the need for STW.

Which collectors use STW?

Parallel GC: Long, full STW during compaction.

G1 GC: Shorter, more predictable STW evacuation pauses.

CMS: Still has STW Initial Mark and Final Remark.

ZGC / Shenandoah: Very short STW (<1 ms) but still required for root marking.


In one line

STW pauses are mandatory JVM freeze points that allow the GC to safely analyze or modify memory without risk of concurrent changes.


CMS does not compact by default, but it still needs pause times because several critical operations cannot be done concurrently and must stop all application threads for correctness and safety.


1. CMS needs STW pauses for “Initial Mark” and “Final Remark”

Component: Concurrent Mark-Sweep (CMS) Collector

Although CMS marks and sweeps concurrently, it cannot start marking without first knowing all GC roots are stable.

Initial Mark (STW):

Identifies direct GC roots (stack, registers, class metadata).

Must be STW to avoid missing references.


Final Remark (STW):

Captures any references that changed while CMS was marking concurrently.

Ensures accurate liveness before sweeping.



These are short but mandatory pauses.


---

2. CMS sweeping is concurrent, but metadata cleanup is not

Even without compaction, CMS must occasionally perform:

cleaning dead class metadata

updating free lists

managing fragmentation tracking


Parts of this require brief STW coordination.


---

3. CMS fails without STW pauses because it is not moving objects

Since CMS does not compact, it cannot handle:

objects being relocated

reference updates during movement


Therefore it relies on STW phases to ensure:

The heap view is safe

No references are missed

The mark phase is consistent


This is a correctness requirement.


---

4. CMS fragmentation increases the need for Full GC (which is STW and compacts)

When fragmentation becomes too high:

CMS triggers Concurrent Mode Failure

JVM falls back to Parallel Old Full GC

This is a long, blocking compaction


So even though CMS’s design avoids compaction, the system still needs pauses under pressure.


---

In one line

CMS still needs pause time because accurate root marking and reference stability cannot be done concurrently, even though compaction is not performed.
### Predictive Pause Model in G1 GC

G1 GC uses a **Predictive Pause Model** to meet a user-defined pause time target (for example, `-XX:MaxGCPauseMillis=200`). Instead of stopping the entire heap, the **G1 Garbage Collector** divides the heap into many small **regions** and uses runtime statistics to *predict* how long collecting each region will take. The **G1 GC Policy component** then selects just enough regions for the next GC cycle so that the pause stays within the requested time.

It continuously updates cost models from previous collections (copy cost, scan cost, remembered set cost) and uses these predictions to schedule incremental evacuation work. This allows G1 to offer **soft real-time behavior**, avoiding long unpredictable pauses typical in older garbage collectors.




### Load Barriers

A **load barrier** is a small piece of code inserted by the **JVM** when a reference is *read* from memory. It allows concurrent garbage collectors (like **ZGC** and **Shenandoah**) to intercept every pointer load and apply relocation, coloring checks, or pointer fixing. When a thread loads an object reference, the load barrier checks whether the object has been moved, needs color adjustment, or requires remediation. This enables **fully concurrent compaction** without stopping application threads.



### Colored Pointers

**Colored pointers** are used by **ZGC** to store GC metadata *inside the pointer bits* rather than in object headers. ZGC uses the unused upper bits of 64-bit pointers to encode color flags such as:

* **Mark bit** (is the object marked?)
* **Remap bit** (does the pointer point to the relocated address?)
* **Finalizable bit** (is the object awaiting finalization?)

When a pointer is loaded, the **ZGC load barrier** reads its color and decides whether to fix the pointer or perform relocation. This allows ZGC to maintain **concurrent marking and compaction** with extremely low pauses.


### Brooks Pointers

**Brooks pointers** are used by **Shenandoah GC**. Instead of encoding metadata inside pointer bits, Shenandoah keeps a **forwarding pointer inside each object** itself. Each object contains a hidden header field called the *Brooks forwarding pointer* that initially points to the object’s own address.

When an object is moved, the GC updates this forwarding pointer to point to the new location. The **Shenandoah write/load barrier** always dereferences the Brooks pointer first, ensuring the application threads always see the latest object location. This allows **concurrent evacuation** without Stop-the-World compaction.





| Concept          | Used By         | Purpose                                           | How It Works                                    |
| ---------------- | --------------- | ------------------------------------------------- | ----------------------------------------------- |
| Load Barriers    | ZGC, Shenandoah | Intercept every pointer load and apply relocation | JVM inserts code at reference loads             |
| Colored Pointers | ZGC             | Encode GC metadata in pointer bits                | Pointer bits → mark/remap state                 |
| Brooks Pointers  | Shenandoah      | Forwarding pointer stored in object header        | All pointer loads go through forwarding pointer |



### Parallel Garbage Collector

Also called **Throughput Collector** (`-XX:+UseParallelGC`).

* **Goal:** Maximize **throughput** (application work / total time).
* Works with **multiple GC threads** to clean the heap **in parallel**.
* But still **Stop-the-World (STW)**:

* When GC runs, all application threads (mutators) are paused.

### Working

1. Heap is divided into **Young Generation** (Eden + Survivor) and **Old Generation**.
2. When Eden fills → **Minor GC** occurs.

   * Multiple GC threads clear Eden and move survivors.
   * Mutator threads are stopped until collection finishes.
3. When Old Gen fills → **Major GC / Full GC** occurs.

   * Again, all threads stop.
   * Multiple GC threads compact Old Gen.

### Characteristics

* **Parallelism:** GC itself is parallelized (uses multiple CPU cores).
* **Not concurrent:** mutator threads are paused during GC.
* **Best for batch jobs / high-throughput systems** where pauses are tolerable.

 

 

```bash
java -XX:+UseParallelGC -XX:ParallelGCThreads=4 MyApp
```

* Uses 4 GC threads.
* Still STW during GC, but faster collection compared to Serial GC.
 

## Concurrent Garbage Collectors

Examples: **CMS (Concurrent Mark-Sweep, deprecated in Java 9+)**, **G1 GC (default since Java 9)**, **ZGC**, **Shenandoah**.

* **Goal:** Minimize **pause times** (latency).
* Work **concurrently with mutator threads**, so application keeps running during most of the GC.

 

###   CMS (Concurrent Mark-Sweep) – classic concurrent collector

* Runs alongside application threads.
* Process:

  1. **Initial Mark (STW):** Mark roots.
  2. **Concurrent Mark:** Mark reachable objects while app runs.
  3. **Remark (STW):** Catch up on changes missed during concurrent mark.
  4. **Concurrent Sweep:** Reclaim dead objects.
* Low pause times, but causes **fragmentation** (no compaction).

 

###   G1 GC (Garbage-First GC, default in Java 9+)

* Heap divided into **many small regions**.
* Collects regions with most garbage first.
* Uses both **parallelism** (multi-threaded GC) and **concurrency** (runs alongside mutators).
* Process:

  * Young GC (parallel, stop-the-world but fast).
  * Concurrent marking of Old Gen.
  * Mixed GCs (collect both Young + some Old regions).
* Predictable pauses (configurable with `-XX:MaxGCPauseMillis`).

 

###   ZGC & Shenandoah

* **Ultra-low pause GCs**.
* Pause times are in **milliseconds, independent of heap size**.
* Achieved with:

  * **Colored pointers / forwarding pointers.**
  * **Read barriers** (on object access, fix reference if object moved).
  * Fully concurrent compaction (no long STW pauses).
* Suitable for **large heaps (TBs)** and **real-time systems**.

 

###   Difference between Parallel and  Concurrent

| Feature           | Parallel GC (Throughput GC)    | Concurrent GCs (CMS, G1, ZGC, Shenandoah) |
| ----------------- | ------------------------------ | ----------------------------------------- |
| **Parallelism**   | ✅ Yes (GC uses many threads)   | ✅ Yes (GC uses many threads)              |
| **Concurrency**   | ❌ No (mutators stop during GC) | ✅ Yes (GC runs alongside mutators)        |
| **STW Pauses**    | Long (but faster than Serial)  | Short (few ms for concurrent GCs)         |
| **Goal**          | Maximize throughput            | Minimize pause times (low latency)        |
| **Heap Size**     | Small to medium heaps          | Medium to very large heaps (up to TBs)    |
| **Best Use Case** | Batch jobs, background tasks   | Interactive apps, low-latency services    |




### Latency vs Throughput

* **Parallel GC**: Multiple GC threads, but **STW** → good for throughput.
* **Concurrent GC (CMS, G1, ZGC, Shenandoah)**: GC + mutators run **together**, small pauses → good for low-latency.
  
| Aspect         | Parallel GC                                                             | Concurrent GC (CMS, G1)                                                |
| -------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Pause time** | Long pauses (STW), but done faster using multiple cores                 | Shorter pauses, since many tasks run alongside the app                 |
| **CPU usage**  | Uses multiple cores, but **only during STW pauses**                     | Uses CPU cores concurrently, which can slightly reduce app throughput  |
| **Throughput** | High (good if pause time doesn’t matter much)                           | Slightly lower (because GC runs alongside app threads)                 |
| **Best for**   | Batch jobs, backend services where throughput matters more than latency | Low-latency apps, interactive systems, real-time response requirements |


Great! Here's a clean, interview-ready version of your answer:


---

In Java garbage collectors, concurrent phases allow the GC to perform work—such as marking reachable objects—while the application threads (mutators) continue running. In contrast, a Stop-the-World (STW) pause halts all application threads so the GC can safely perform operations that require exclusive access, such as certain marking, cleanup, or reference processing steps. In collectors like Parallel GC, most work (including marking) happens during STW pauses, which can cause noticeable latency.

Modern collectors like G1 reduce these pause times by breaking the heap into regions and focusing GC cycles only on selected regions rather than the entire heap. G1 performs much of the marking concurrently and provides configurable pause-time goals to keep latency predictable. Even more advanced collectors such as ZGC and Shenandoah push this further by performing both marking and compaction concurrently. They use techniques like colored pointers or load barriers so that mutators always access the correct, moved object without needing long pauses.







