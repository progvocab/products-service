

## 🔹 Semaphore in Threads (Java context)

A **semaphore** is a synchronization mechanism that **controls access to a shared resource** by maintaining a set number of permits.

Think of it like a counter for how many threads can access a resource at the same time.

* **Binary Semaphore (mutex-like):** Only 1 permit (acts like a lock).
* **Counting Semaphore:** Multiple permits (e.g., at most 3 threads can access a resource at once).

---

## 🔹 How it Works

1. A semaphore is initialized with a number of **permits**.
2. A thread calls `acquire()` to get a permit:

   * If a permit is available → thread continues.
   * If not → thread blocks until a permit is released.
3. After the thread finishes using the resource, it calls `release()` to give the permit back.

---

## 🔹 Example 1 – Binary Semaphore (1 permit)

```java
import java.util.concurrent.Semaphore;

public class BinarySemaphoreExample {
    private static final Semaphore semaphore = new Semaphore(1); // only 1 permit

    public static void main(String[] args) {
        Runnable task = () -> {
            try {
                System.out.println(Thread.currentThread().getName() + " waiting...");
                semaphore.acquire(); // request permit
                System.out.println(Thread.currentThread().getName() + " acquired lock!");

                Thread.sleep(1000); // simulate work

            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                semaphore.release(); // release permit
                System.out.println(Thread.currentThread().getName() + " released lock!");
            }
        };

        new Thread(task, "Thread-1").start();
        new Thread(task, "Thread-2").start();
    }
}
```

### Output (example)

```
Thread-1 waiting...
Thread-1 acquired lock!
Thread-2 waiting...
Thread-1 released lock!
Thread-2 acquired lock!
Thread-2 released lock!
```

Here only **one thread runs at a time** → acts like a mutex.

---

## 🔹 Example 2 – Counting Semaphore (3 permits)

```java
import java.util.concurrent.Semaphore;

public class CountingSemaphoreExample {
    private static final Semaphore semaphore = new Semaphore(3); // 3 permits

    public static void main(String[] args) {
        Runnable task = () -> {
            try {
                semaphore.acquire();
                System.out.println(Thread.currentThread().getName() + " entered.");
                Thread.sleep(2000); // simulate work
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                System.out.println(Thread.currentThread().getName() + " exiting.");
                semaphore.release();
            }
        };

        for (int i = 1; i <= 6; i++) {
            new Thread(task, "Thread-" + i).start();
        }
    }
}
```

### Output (example)

```
Thread-1 entered.
Thread-2 entered.
Thread-3 entered.
Thread-1 exiting.
Thread-4 entered.
Thread-2 exiting.
Thread-5 entered.
Thread-3 exiting.
Thread-6 entered.
```

At most **3 threads run in parallel**.

---

## 🔹 When to Use Semaphore

* To **limit concurrent access** (e.g., only 10 database connections allowed).
* To **implement resource pools** (like connection pool, thread pool).
* As a **mutex substitute** (using binary semaphore).

---

✅ **Summary:**

* A **semaphore** is a counter of permits that regulates how many threads can access a resource simultaneously.
* `acquire()` → take permit (may block).
* `release()` → give permit back.
* Binary = lock, Counting = allow N threads.

---

Do you want me to also compare **Semaphore vs Lock vs Monitor (synchronized keyword)** so you clearly see when to use which?


Perfect 👍, let’s compare **Semaphore vs Lock vs Monitor (`synchronized`)**.

---

# 🔹 1. Monitor (`synchronized`)

* Built-in synchronization mechanism in Java.
* Every object in Java has an **intrinsic lock (monitor)**.
* A thread must acquire this lock before executing a synchronized block/method.
* Only **one thread** can hold the monitor at a time.

✅ **Good for:** simple critical sections, ensuring only one thread accesses code.
❌ **Limitations:**

* Always exclusive (no multiple permits).
* No fairness guarantee.
* No advanced control (e.g., tryLock with timeout).

---

### Example

```java
public class MonitorExample {
    private int count = 0;

    public synchronized void increment() {
        count++; // only one thread at a time
    }
}
```

---

# 🔹 2. Lock (from `java.util.concurrent.locks`)

* More flexible than `synchronized`.
* Explicitly lock/unlock using `lock()` and `unlock()`.
* Provides features like:

  * `tryLock()` → try acquiring without blocking.
  * `lockInterruptibly()` → can be interrupted while waiting.
  * Fairness policy (`ReentrantLock(true)`).
  * Supports multiple condition variables.

✅ **Good for:** fine-grained control, complex synchronization.
❌ **Limitation:** Only one thread can hold the lock at a time (like monitor).

---

### Example

```java
import java.util.concurrent.locks.ReentrantLock;

public class LockExample {
    private final ReentrantLock lock = new ReentrantLock();
    private int count = 0;

    public void increment() {
        lock.lock(); // acquire lock
        try {
            count++;
        } finally {
            lock.unlock(); // always release
        }
    }
}
```

---

# 🔹 3. Semaphore

* Works with **permits** instead of exclusive lock.
* Can allow **multiple threads** to access a resource at the same time.
* `acquire()` decreases permits, `release()` increases permits.
* **Binary Semaphore** = behaves like a lock.
* **Counting Semaphore** = multiple permits → useful for throttling or resource pools.

✅ **Good for:** limiting concurrency (e.g., only 5 threads can download at once).
❌ **Limitation:** doesn’t enforce mutual exclusion unless it’s binary.

---

### Example

```java
import java.util.concurrent.Semaphore;

public class SemaphoreExample {
    private final Semaphore semaphore = new Semaphore(3); // 3 permits

    public void accessResource() {
        try {
            semaphore.acquire(); // request permit
            System.out.println(Thread.currentThread().getName() + " using resource");
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            semaphore.release(); // release permit
        }
    }
}
```

---

# 🔹 Side-by-Side Comparison

| Feature                  | `synchronized` (Monitor) | Lock (ReentrantLock)   | Semaphore                                  |
| ------------------------ | ------------------------ | ---------------------- | ------------------------------------------ |
| Mutual Exclusion         | ✅ Yes (1 thread only)    | ✅ Yes (1 thread only)  | ✅ Binary semaphore; ❌ Counting allows many |
| Multiple Permits         | ❌ No                     | ❌ No                   | ✅ Yes (Counting Semaphore)                 |
| Explicit Acquire/Release | ❌ Hidden by JVM          | ✅ Yes (`lock/unlock`)  | ✅ Yes (`acquire/release`)                  |
| Try with Timeout         | ❌ No                     | ✅ Yes                  | ✅ Yes (`tryAcquire`)                       |
| Fairness Option          | ❌ No                     | ✅ Yes                  | ✅ Yes                                      |
| Advanced Features        | ❌ No                     | ✅ Conditions, fairness | ✅ Resource throttling                      |
| Simplicity               | ✅ Very simple            | ⚠️ Medium complexity   | ⚠️ Medium complexity                       |

---

✅ **Rule of Thumb:**

* Use **`synchronized`** for **simple exclusive locking**.
* Use **`Lock`** if you need **advanced control** (fairness, tryLock, interruptibility).
* Use **`Semaphore`** when you need to **limit the number of concurrent threads** (resource pool, throttling).

---

👉 Do you want me to also show a **real-world analogy** (like parking lot, ATM, etc.) for these three so it’s super intuitive?
