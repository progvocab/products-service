
---

##   `ReentrantLock`

* A **mutual exclusion lock** provided in `java.util.concurrent.locks`.
* "Reentrant" means **a thread holding the lock can acquire it again without deadlocking**.
* More **flexible** than the `synchronized` keyword:

  * Supports **fair locking**
  * Provides **tryLock()** (non-blocking)
  * Supports **lockInterruptibly()**
  * Requires **explicit unlock()**.

---

##  Key Methods

1. `lock()` → Acquires the lock (waits if unavailable).
2. `unlock()` → Releases the lock (must be called in `finally`).
3. `tryLock()` → Tries to acquire lock **without waiting**.
4. `tryLock(long time, TimeUnit unit)` → Waits for a max time before giving up.
5. `lockInterruptibly()` → Acquires lock but allows interruption while waiting.
6. `isLocked()` → Checks if lock is held.
7. `isHeldByCurrentThread()` → Checks if the current thread holds it.

---

##  Basic Example

```java
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class SharedResource {
    private final Lock lock = new ReentrantLock();
    private int counter = 0;

    public void increment() {
        lock.lock(); // acquire lock
        try {
            counter++;
            System.out.println(Thread.currentThread().getName() + " -> " + counter);
        } finally {
            lock.unlock(); // always release
        }
    }

    public static void main(String[] args) {
        SharedResource resource = new SharedResource();

        Runnable task = resource::increment;

        Thread t1 = new Thread(task, "Thread-1");
        Thread t2 = new Thread(task, "Thread-2");

        t1.start();
        t2.start();
    }
}
```

* Ensures **only one thread increments at a time**.
* Without `lock()`, both threads could modify `counter` simultaneously (race condition).

---

## Example with `tryLock()`

```java
import java.util.concurrent.locks.ReentrantLock;

public class TryLockExample {
    private final ReentrantLock lock = new ReentrantLock();

    public void access() {
        if (lock.tryLock()) {  // won't wait, just check
            try {
                System.out.println(Thread.currentThread().getName() + " got the lock");
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock();
            }
        } else {
            System.out.println(Thread.currentThread().getName() + " could not get the lock");
        }
    }

    public static void main(String[] args) {
        TryLockExample example = new TryLockExample();
        Runnable task = example::access;

        new Thread(task, "T1").start();
        new Thread(task, "T2").start();
    }
}
```

---

##  Fair vs Unfair Lock

```java
ReentrantLock fairLock = new ReentrantLock(true);  // fair = true
ReentrantLock unfairLock = new ReentrantLock();    // default unfair
```

* **Fair lock** → Longest waiting thread gets the lock first (like a queue).
* **Unfair lock** (default) → Can allow lock stealing → higher throughput but less fairness.

---

##  When to Use `ReentrantLock`

* Need **tryLock()** or **lockInterruptibly()**.
* Want **fair locking**.
* Complex locking scenarios (multiple conditions, reentrant behavior).
* Explicit control over lock acquisition/release.

---

