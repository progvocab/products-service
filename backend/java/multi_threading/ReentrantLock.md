
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

ReentrantLock is often introduced as “a replacement for `synchronized`”, but its real strength comes out in situations where **you need finer control over locking**.

Here’s a **simple but important use case**:

---

## **Use Case: Bounded Buffer (Producer-Consumer Problem)**

Imagine a **bounded buffer** (like a queue with limited capacity). Multiple **producer threads** put items in, and multiple **consumer threads** take items out.
We need to ensure:

* Only one thread can modify the buffer at a time.
* Producers wait if the buffer is full.
* Consumers wait if the buffer is empty.

This is where **ReentrantLock + Condition variables** shine, because they give us **fine-grained control** beyond what `synchronized` can do.

---

### **Code Example**

```java
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.LinkedList;
import java.util.Queue;

class BoundedBuffer<T> {
    private final Queue<T> buffer = new LinkedList<>();
    private final int capacity;

    private final Lock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();

    public BoundedBuffer(int capacity) {
        this.capacity = capacity;
    }

    public void put(T item) throws InterruptedException {
        lock.lock();
        try {
            while (buffer.size() == capacity) {
                notFull.await(); // wait if buffer is full
            }
            buffer.add(item);
            System.out.println("Produced: " + item);
            notEmpty.signal(); // signal consumer
        } finally {
            lock.unlock();
        }
    }

    public T take() throws InterruptedException {
        lock.lock();
        try {
            while (buffer.isEmpty()) {
                notEmpty.await(); // wait if buffer is empty
            }
            T item = buffer.poll();
            System.out.println("Consumed: " + item);
            notFull.signal(); // signal producer
            return item;
        } finally {
            lock.unlock();
        }
    }
}

public class ProducerConsumerDemo {
    public static void main(String[] args) {
        BoundedBuffer<Integer> buffer = new BoundedBuffer<>(3);

        // Producer
        new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    buffer.put(i);
                    Thread.sleep(200);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();

        // Consumer
        new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    buffer.take();
                    Thread.sleep(500);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }
}
```

---

### **Why ReentrantLock is Important Here**

* **Conditions** (`notFull`, `notEmpty`) allow us to explicitly control *when* threads wait and *who* gets signaled.
* `synchronized` + `wait/notify` could do something similar, but `Condition` variables give **multiple waiting queues**, which makes logic much cleaner.
* `ReentrantLock` allows **timeout locks**, `tryLock()`, and fair ordering (`new ReentrantLock(true)`), which `synchronized` doesn’t provide.

---

✅ This use case is **simple** (bounded buffer) but **important**, because it demonstrates why `ReentrantLock` is more powerful than `synchronized` in concurrency-heavy systems like message queues, job schedulers, and thread pools.

---

Would you like me to also show a **real-world enterprise scenario** (like logging systems or database connection pools) where ReentrantLock is used in production?


Excellent follow-up! 👌 Yes, we **can implement the same bounded buffer (producer-consumer)** using `synchronized`, `wait()`, and `notifyAll()`. But there are **important differences** compared to using `ReentrantLock + Condition`.

---

## **1️⃣ Example with `synchronized + wait/notifyAll`**

```java
import java.util.LinkedList;
import java.util.Queue;

class BoundedBufferSync<T> {
    private final Queue<T> buffer = new LinkedList<>();
    private final int capacity;

    public BoundedBufferSync(int capacity) {
        this.capacity = capacity;
    }

    public synchronized void put(T item) throws InterruptedException {
        while (buffer.size() == capacity) {
            wait(); // wait if buffer is full
        }
        buffer.add(item);
        System.out.println("Produced: " + item);
        notifyAll(); // wake up waiting consumers
    }

    public synchronized T take() throws InterruptedException {
        while (buffer.isEmpty()) {
            wait(); // wait if buffer is empty
        }
        T item = buffer.poll();
        System.out.println("Consumed: " + item);
        notifyAll(); // wake up waiting producers
        return item;
    }
}

public class ProducerConsumerSyncDemo {
    public static void main(String[] args) {
        BoundedBufferSync<Integer> buffer = new BoundedBufferSync<>(3);

        // Producer
        new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    buffer.put(i);
                    Thread.sleep(200);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();

        // Consumer
        new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    buffer.take();
                    Thread.sleep(500);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }
}
```

---

## **2️⃣ Differences Between `synchronized` and `ReentrantLock`**

| Feature                          | `synchronized + wait/notify`                    | `ReentrantLock + Condition`                                  |
| -------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| **Lock Type**                    | Implicit, JVM-managed monitor lock              | Explicit, programmer-controlled                              |
| **Fairness**                     | No fairness guarantee                           | Can create *fair locks* (`new ReentrantLock(true)`)          |
| **Multiple Condition Queues**    | Only **one wait set per object**                | Multiple `Condition` objects allowed (`notFull`, `notEmpty`) |
| **Interruptible Locking**        | No (once waiting, only notify can wake)         | Yes (`lock.lockInterruptibly()`)                             |
| **Try Lock with Timeout**        | Not possible                                    | `tryLock()` and `tryLock(timeout)` available                 |
| **Readability in Complex Cases** | Harder to manage (lots of `wait()/notifyAll()`) | Cleaner with multiple `Condition`s                           |
| **Performance**                  | Simpler, less overhead                          | Slightly more overhead, but more flexible                    |

---

## **3️⃣ Key Takeaway**

* For **simple cases** like a basic producer-consumer buffer, `synchronized + wait/notifyAll` works fine.
* For **complex concurrency control** (e.g., multiple conditions, fairness, timed locks, or interruption), **`ReentrantLock + Condition` is much more powerful**.

Think of it like this:

* `synchronized` = **basic lock with 1 waiting room**.
* `ReentrantLock + Conditions` = **advanced lock with multiple waiting rooms, timeouts, fairness, and emergency exits**.

---

👉 Would you like me to also **visualize this difference with a diagram (Mermaid)** showing **1 wait set vs multiple condition queues**? That way, it’s super clear why ReentrantLock is better in certain cases.


Great question 👍 — yes, we can absolutely implement the **Producer–Consumer (Bounded Buffer)** problem using a **Semaphore**. In fact, semaphore-based solutions are a **classic** way to demonstrate synchronization in operating systems courses.

---

## 🔑 Idea with Semaphores

We use **three things**:

1. `Semaphore empty` → counts how many empty slots are available in the buffer (initial = capacity).
2. `Semaphore full` → counts how many items are available to consume (initial = 0).
3. A **mutex (lock)** → ensures only one thread modifies the buffer at a time.

---

## 📝 Example Code (Java, using `Semaphore`)

```java
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.Semaphore;

class BoundedBufferSemaphore<T> {
    private final Queue<T> buffer = new LinkedList<>();
    private final int capacity;

    private final Semaphore empty; // available slots
    private final Semaphore full;  // available items
    private final Semaphore mutex = new Semaphore(1); // mutual exclusion

    public BoundedBufferSemaphore(int capacity) {
        this.capacity = capacity;
        this.empty = new Semaphore(capacity); // initially all slots empty
        this.full = new Semaphore(0);         // initially no items
    }

    public void put(T item) throws InterruptedException {
        empty.acquire();   // wait if no empty slot
        mutex.acquire();   // lock buffer
        buffer.add(item);
        System.out.println("Produced: " + item);
        mutex.release();   // unlock buffer
        full.release();    // signal an item is available
    }

    public T take() throws InterruptedException {
        full.acquire();    // wait if no item
        mutex.acquire();   // lock buffer
        T item = buffer.poll();
        System.out.println("Consumed: " + item);
        mutex.release();   // unlock buffer
        empty.release();   // signal a slot is free
        return item;
    }
}

public class ProducerConsumerSemaphoreDemo {
    public static void main(String[] args) {
        BoundedBufferSemaphore<Integer> buffer = new BoundedBufferSemaphore<>(3);

        // Producer
        new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    buffer.put(i);
                    Thread.sleep(200);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();

        // Consumer
        new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    buffer.take();
                    Thread.sleep(500);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }
}
```

---

## ⚖️ Comparison: `Semaphore` vs `synchronized` / `ReentrantLock`

| Feature              | `Semaphore`                                     | `synchronized` / `ReentrantLock`            |
| -------------------- | ----------------------------------------------- | ------------------------------------------- |
| **Primary use**      | Signaling and counting resources                | Mutual exclusion (locks)                    |
| **Multiple permits** | Yes (can allow multiple threads at once)        | No (1 thread only)                          |
| **Fairness option**  | Yes (`new Semaphore(permits, true)`)            | Yes (`new ReentrantLock(true)`)             |
| **Buffer slots**     | Naturally modeled with permits (`empty`/`full`) | Requires explicit wait/notify or Conditions |
| **Complexity**       | Clear for producer-consumer                     | Can get verbose with wait/notify            |
| **Interruptible**    | Yes (`acquire()` is interruptible)              | Yes with `lockInterruptibly()`              |

---

✅ So, using a **Semaphore** is often the **cleanest and most intuitive** way to solve bounded buffer problems, because the permits directly map to **slots/items**.

---

👉 Do you want me to also **draw a Mermaid diagram** showing how **empty/full/mutex semaphores interact** between producers and consumers?


