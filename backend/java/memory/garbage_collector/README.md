

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







