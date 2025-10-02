

---

# Key Multithreading Concepts in Java

| Concept                            | Explanation                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Thread**                         | A lightweight unit of execution. In Java, created by extending `Thread` or implementing `Runnable`.           |
| **Thread Lifecycle**               | States: `New → Runnable → Running → Waiting/Blocked → Terminated`. Managed by JVM and OS.                     |
| **Thread Priorities**              | Each thread has a priority (`1` to `10`). JVM/OS uses it for scheduling (but not guaranteed).                 |
| **Runnable Interface**             | Defines the `run()` method. Preferred way to create threads (decouples task from thread).                     |
| **Callable & Future**              | `Callable<V>` returns a result and can throw exceptions. `Future` represents the result of async computation. |
| **Executor Framework**             | Provides a high-level API (`ExecutorService`, `ThreadPoolExecutor`) for managing thread pools.                |
| **Synchronization**                | Mechanism to control access to shared resources. Done with `synchronized` keyword or locks.                   |
| **Lock Interface (ReentrantLock)** | Provides explicit locking/unlocking, more flexible than `synchronized`.                                       |
| **Volatile Keyword**               | Ensures visibility of variable updates across threads. Prevents caching in thread-local memory.               |
| **Atomic Variables**               | (`AtomicInteger`, `AtomicLong`, etc.) provide lock-free thread-safe operations.                               |
| **ThreadLocal**                    | Each thread has its own isolated copy of a variable. Useful for user sessions, request context.               |
| **Wait, Notify, NotifyAll**        | Methods used for inter-thread communication on objects’ monitors.                                             |
| **Deadlock**                       | Situation where two or more threads wait indefinitely for each other’s resources.                             |
| **Livelock**                       | Threads keep changing state in response to each other but never make progress.                                |
| **Starvation**                     | A thread never gets CPU time/resources due to low priority or scheduling.                                     |
| **Thread Pools**                   | Reuse fixed number of threads to improve performance (`Executors.newFixedThreadPool()`).                      |
| **Fork/Join Framework**            | Splits tasks into smaller subtasks (divide & conquer) and merges results (`ForkJoinPool`).                    |
| **Parallel Streams**               | High-level API for parallel data processing using the Fork/Join framework internally.                         |
| **CompletableFuture**              | Asynchronous, non-blocking computations with chaining (`thenApply`, `thenCombine`).                           |
| **Concurrency Utilities**          | Classes in `java.util.concurrent`: `CountDownLatch`, `CyclicBarrier`, `Semaphore`, `BlockingQueue`.           |

---

#  Quick Notes

* **Thread Safety**: Avoid race conditions using synchronization, locks, atomics.
* **Executors**: Prefer over manual `Thread` creation for better scalability.
* **Avoid Deadlocks**: Use consistent lock ordering, timeouts, or try-lock.
* **Best Practice**: Use high-level concurrency utilities (Executor, CompletableFuture) instead of low-level `wait/notify`.

---


# **List of Locking mechanism**:

---

## 🔑 Ways to Lock in Java

1. **Object Monitor (Intrinsic Lock)**

   * Obtained via `synchronized` methods or blocks.
   * Every object has a built-in monitor lock.
   * Example:

     ```java
     synchronized (obj) { ... }
     ```

2. **ReentrantLock (java.util.concurrent.locks.Lock)**

   * More flexible than `synchronized`.
   * Allows **tryLock()**, **lockInterruptibly()**, and **fair locking**.
   * Example:

     ```java
     Lock lock = new ReentrantLock();
     lock.lock();
     try { ... } finally { lock.unlock(); }
     ```

3. **ReentrantReadWriteLock**

   * Provides **read lock** (shared, multiple readers allowed) and **write lock** (exclusive).
   * Good for scenarios with more reads than writes.

4. **StampedLock (Java 8+)**

   * Optimized for read-mostly use cases.
   * Supports **optimistic reads** (no blocking).
   * More scalable than `ReentrantReadWriteLock`.

5. **Semaphore (java.util.concurrent)**

   * Controls access to a resource pool with a set number of permits.
   * Example: limit to N concurrent threads.

6. **CountDownLatch**

   * Not a lock in the strict sense, but **blocks threads until a count reaches zero**.
   * Often used for **one-time synchronization**.

7. **CyclicBarrier**

   * Synchronizes a set of threads at a barrier point before continuing execution.
   * Useful for **multi-threaded algorithms**.

8. **Phaser (Java 7+)**

   * More flexible version of `CyclicBarrier` for **multi-phase synchronization**.

9. **Atomic Variables (java.util.concurrent.atomic)**

   * Lock-free thread-safe operations (CAS — Compare-And-Swap).
   * Example: `AtomicInteger`, `AtomicLong`.

10. **Volatile Keyword**

* Not a lock, but ensures **visibility** of changes across threads.
* Used to prevent caching issues in multi-threading.

---

 **Summary**:

* **Basic locking** → `synchronized` (object monitors).
* **Advanced locking** → `ReentrantLock`, `ReadWriteLock`, `StampedLock`.
* **Coordination tools** → `Semaphore`, `CountDownLatch`, `CyclicBarrier`, `Phaser`.
* **Lock-free** → `Atomic` classes + `volatile`.

---

Great question 👍
When you start a plain Java application (say you just run `java MyApp` with a simple `main` method), the JVM itself creates **a set of system threads** even before your code runs. So you’ll usually see **much more than just your `main` thread**.

---

### 🔹 Typical default threads in a fresh JVM

At startup, you will generally see:

1. **Main thread**

   * Runs your `public static void main(String[] args)` method.

2. **Reference Handler**

   * Cleans up references like `PhantomReference`, `WeakReference`, `SoftReference`.

3. **Finalizer**

   * Runs finalizers on objects that are about to be garbage-collected (deprecated in recent JDKs, but still present in some versions).

4. **Signal Dispatcher**

   * Handles signals sent to the JVM process (e.g., CTRL+C, or OS signals).

5. **GC threads**

   * Depends on the garbage collector chosen by the JVM.
   * For example:

     * **G1GC** (default since JDK 9) → multiple GC worker threads + concurrent marking threads.
     * **Parallel GC** → several GC threads.
     * **Serial GC** → just one GC thread.

6. **JIT compiler threads**

   * The JVM’s Just-In-Time compiler (C2, Tiered compilation, Graal) spins up background threads to compile bytecode into machine code.

7. **Attach Listener**

   * Used when tools (like VisualVM, JConsole, profilers) connect to the JVM.

---

### 🔹 How many exactly?

It **depends on:**

* **JVM version** (Java 8, 11, 17, 21… may differ).
* **GC algorithm** in use.
* **JVM options** (`-XX:+UseParallelGC`, `-XX:CICompilerCount=2`, etc.).
* **OS & hardware** (more CPUs → more GC/JIT threads).

👉 But on a typical **Java 11+ with default G1GC**, you’ll often see around **10–20 threads** immediately at startup.

---

### 🔹 Quick check yourself

You can print all threads at JVM startup with:

```java
public class ThreadCheck {
    public static void main(String[] args) {
        Thread.getAllStackTraces().keySet()
              .forEach(t -> System.out.println(t.getName()));
    }
}
```

On a simple run, you’ll usually see something like:

```
main
Reference Handler
Finalizer
Signal Dispatcher
Common-Cleaner
Attach Listener
GC Thread#0
GC Thread#1
...
C2 CompilerThread0
C2 CompilerThread1
```

---

**Summary:**
By default, a Java application **starts with ~10–20 threads** (not just `main`), depending on the JVM and GC.


