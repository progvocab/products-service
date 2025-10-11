# Executor Service 

- **Problem** : Creating a new Thread is costly.
- **Solution** : Resuse Threads using Executor Service.

Lets say you a number of tasks to be executed concurrently,  Instead of creating a new for each task , maintain a pool of Threads , and reuse them. This can be done using Executor Service.

### Why Thread creation is costly 








---

## 1️⃣ What happens when you create a new Thread

When you do:

```java
Thread t = new Thread(() -> {
    // some work
});
t.start();
```

The JVM has to:

1. **Allocate memory** for:

   * Thread object (Java object in heap)
   * Native thread stack (typically 512 KB to 1 MB per thread by default)
   * Thread-local storage, JVM thread control structures
2. **Register the thread** with the OS (each Java thread maps to a native OS thread in HotSpot).
3. **Schedule the thread** with the OS kernel.
4. **Start execution** (context switches, CPU scheduling).

All these steps are **much heavier than simply reusing an existing thread** from a thread pool.

---

## 2️⃣ Why creating threads is costly

| Cost        | Reason                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------- |
| Memory      | Each thread needs a **stack**, JVM structures, and thread-local memory                             |
| CPU         | OS has to schedule the thread, context-switch between threads                                      |
| Latency     | Startup time is noticeable if creating threads frequently (~100–500 µs per thread, sometimes more) |
| Scalability | Thousands of threads can exhaust memory and degrade system performance                             |

**Example:**
Creating 10,000 new threads simultaneously can easily **consume multiple GB of RAM** just for stacks, and cause thrashing due to excessive context switching.

---

## 3️⃣ Why thread pools help

Instead of creating a new thread per task:

* Threads are **created once** and reused.
* Idle threads **wait** for new tasks.
* Reduces:

  * Memory allocation overhead
  * OS scheduling overhead
  * GC pressure from frequent thread objects

---

## 4️⃣ Empirical numbers (approximate)

| Operation                             | Time / Cost                       |
| ------------------------------------- | --------------------------------- |
| Creating new thread                   | 100–500 µs (or more)              |
| Starting task on existing thread pool | 1–10 µs                           |
| Context switch between threads        | ~1–10 µs                          |
| Thread stack memory                   | 512 KB–1 MB per thread by default |

> So using a thread pool can be **10x–100x faster per task** in high-throughput workloads.

---

## 5️⃣ Recommendations

1. **Use thread pools (`ExecutorService`)** instead of new `Thread()` for tasks.
2. **Use CachedThreadPool** for bursty loads — threads get reused, idle threads die after 60s.
3. **Use FixedThreadPool** when you want predictable concurrency and memory usage.
4. **Avoid creating threads in tight loops** or per-request in high-traffic servers.

---

✅ **Bottom line:**
Creating a new thread is **expensive in both memory and CPU**, and can limit scalability. **Thread pools exist specifically to avoid this overhead by reusing threads.**

---

If you want, I can also **show a side-by-side timing comparison of new Thread vs ThreadPool execution** with Java code — it’s a very practical demonstration of why pools are better.

Do you want me to show that?


---

## 🧩 1. What is `ExecutorService`?

* `ExecutorService` is a **high-level framework** for managing and controlling threads.
* Instead of creating and managing threads manually, you **submit tasks** to an executor and let it handle **thread pooling, scheduling, and lifecycle**.

---

### ✅ Why use it?

Without it:

```java
new Thread(() -> doSomething()).start();
```

* You manually create and start threads.
* No control over reusing threads or limiting concurrency.

With `ExecutorService`:

```java
ExecutorService executor = Executors.newFixedThreadPool(5);
executor.submit(() -> doSomething());
```

* Efficient **thread reuse**.
* Built-in **thread pool management**.
* Supports **task cancellation, scheduling, and graceful shutdown**.

---

## ⚙️ 2. Hierarchy Overview

```mermaid
classDiagram
    Executor <|-- ExecutorService
    ExecutorService <|-- ScheduledExecutorService
    Executors --> ExecutorService : creates instances

    class Executor {
      <<interface>>
      +execute(Runnable command)
    }

    class ExecutorService {
      <<interface>>
      +submit(Runnable task)
      +submit(Callable<T> task)
      +invokeAll(Collection<? extends Callable<T>> tasks)
      +invokeAny(Collection<? extends Callable<T>> tasks)
      +shutdown()
      +shutdownNow()
      +awaitTermination(long timeout, TimeUnit unit)
      +isShutdown()
      +isTerminated()
    }

    class ScheduledExecutorService {
      <<interface>>
      +schedule(Runnable command, long delay, TimeUnit unit)
      +scheduleAtFixedRate(Runnable command, long initDelay, long period, TimeUnit unit)
      +scheduleWithFixedDelay(Runnable command, long initDelay, long delay, TimeUnit unit)
    }
```

---

## 🧠 3. Common Implementations

| Executor Type             | Created Using                         | Description                                        |
| ------------------------- | ------------------------------------- | -------------------------------------------------- |
| **Single-threaded**       | `Executors.newSingleThreadExecutor()` | One thread executes tasks sequentially.            |
| **Fixed thread pool**     | `Executors.newFixedThreadPool(n)`     | Reuses a fixed number of threads.                  |
| **Cached thread pool**    | `Executors.newCachedThreadPool()`     | Creates threads as needed, reuses idle ones.       |
| **Scheduled thread pool** | `Executors.newScheduledThreadPool(n)` | Schedules tasks for future or periodic execution.  |
| **Work-stealing pool**    | `Executors.newWorkStealingPool()`     | Uses multiple queues; best for parallel workloads. |

---

## 🔹 4. Key Methods in `ExecutorService`

| Method                                                                   | Description                                                            |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `void execute(Runnable task)`                                            | Runs a Runnable task (no return).                                      |
| `<T> Future<T> submit(Callable<T> task)`                                 | Submits a task that returns a result via `Future`.                     |
| `Future<?> submit(Runnable task)`                                        | Submits a Runnable (returns Future for tracking completion).           |
| `<T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks)` | Executes all tasks and waits for all to complete.                      |
| `<T> T invokeAny(Collection<? extends Callable<T>> tasks)`               | Executes all, returns the result of the first successful task.         |
| `void shutdown()`                                                        | Gracefully shuts down (no new tasks accepted, waits for running ones). |
| `List<Runnable> shutdownNow()`                                           | Immediately stops tasks (returns unstarted ones).                      |
| `boolean awaitTermination(long timeout, TimeUnit unit)`                  | Waits for termination after shutdown.                                  |
| `boolean isShutdown()`                                                   | True if shutdown initiated.                                            |
| `boolean isTerminated()`                                                 | True if all tasks completed after shutdown.                            |

---

## 💡 5. Example — Fixed Thread Pool

```java
import java.util.concurrent.*;

public class ExecutorExample {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(3);

        for (int i = 1; i < 6; i++) {
            final int taskId = i;
            executor.submit(() -> {
                System.out.println("Running Task " + taskId + " on " + Thread.currentThread().getName());
                try { Thread.sleep(1000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            });
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}
```

---

### Output (example)

```
Running Task 1 on pool-1-thread-1
Running Task 2 on pool-1-thread-2
Running Task 3 on pool-1-thread-3
Running Task 4 on pool-1-thread-1
Running Task 5 on pool-1-thread-2
```

➡️ Notice how only **3 threads** are reused — not one per task.

---

## 🧭 6. When to Use Each Type

| Executor Type               | Use Case                                         |
| --------------------------- | ------------------------------------------------ |
| `newFixedThreadPool()`      | CPU-bound tasks (limit concurrency).             |
| `newCachedThreadPool()`     | Short-lived, many small async tasks (I/O heavy). |
| `newSingleThreadExecutor()` | Sequential execution with task isolation.        |
| `newScheduledThreadPool()`  | Periodic jobs (e.g., heartbeat, cleanup).        |
| `newWorkStealingPool()`     | Divide & conquer parallelism (ForkJoin).         |

---

## 🔚 7. Graceful Shutdown Sequence

```java
executor.shutdown();              // stop accepting new tasks
executor.awaitTermination(...);   // wait for tasks to finish
if (!executor.isTerminated()) {
    executor.shutdownNow();       // force terminate
}
```

---

✅ **In summary:**
`ExecutorService` gives you:

* Thread pooling
* Task scheduling
* Graceful shutdown
* Future-based results
* Automatic management of concurrency

---

Would you like me to also show how **ExecutorService**, **ThreadPoolExecutor**, and **ForkJoinPool** relate internally (with Mermaid diagram and explanation of thread reuse)?
