Perfect 👍 The trio `wait()`, `notify()`, and `notifyAll()` are the **classic thread communication tools** in Java.
They let **threads coordinate execution** when working on shared resources.
Let’s go through the **most common use cases with code examples**.

---

# 🔑 Key Rules

1. These methods are defined in **Object** class (because every object has a monitor lock).
2. They can only be called inside a `synchronized` block/method.
3. `wait()` → releases the lock and pauses thread until `notify()`/`notifyAll()` is called.
4. `notify()` → wakes **one waiting thread** (chosen randomly).
5. `notifyAll()` → wakes **all waiting threads** (they compete for the lock).

---

# 🧵 Use Case 1: **Producer–Consumer Problem (Bounded Buffer)**

Classic problem where one thread produces data, another consumes it.

```java
import java.util.LinkedList;
import java.util.Queue;

class SharedBuffer {
    private final Queue<Integer> queue = new LinkedList<>();
    private final int CAPACITY = 5;

    public synchronized void produce(int value) throws InterruptedException {
        while (queue.size() == CAPACITY) {
            wait(); // buffer full, wait for consumer
        }
        queue.add(value);
        System.out.println("Produced: " + value);
        notifyAll(); // notify consumers waiting
    }

    public synchronized int consume() throws InterruptedException {
        while (queue.isEmpty()) {
            wait(); // buffer empty, wait for producer
        }
        int val = queue.poll();
        System.out.println("Consumed: " + val);
        notifyAll(); // notify producers waiting
        return val;
    }
}

public class ProducerConsumerExample {
    public static void main(String[] args) {
        SharedBuffer buffer = new SharedBuffer();

        // Producer thread
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 10; i++) {
                try {
                    buffer.produce(i);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        // Consumer thread
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 10; i++) {
                try {
                    buffer.consume();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        producer.start();
        consumer.start();
    }
}
```

👉 Here:

* `wait()` suspends producers if buffer is full, or consumers if empty.
* `notifyAll()` wakes up threads when buffer state changes.

---

# 🧵 Use Case 2: **Thread Sequencing (Ordering Execution)**

Ensure threads execute in a particular order.

```java
class Task {
    private boolean firstDone = false;

    public synchronized void first() {
        System.out.println("First");
        firstDone = true;
        notifyAll();
    }

    public synchronized void second() throws InterruptedException {
        while (!firstDone) {
            wait(); // wait until first() is done
        }
        System.out.println("Second");
    }
}

public class SequencingExample {
    public static void main(String[] args) {
        Task task = new Task();

        Thread t2 = new Thread(() -> {
            try {
                task.second();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        Thread t1 = new Thread(task::first);

        t2.start();
        t1.start();
    }
}
```

👉 Here, `wait()` ensures `second()` runs **after** `first()`.

---

# 🧵 Use Case 3: **Worker Threads Waiting for a Signal**

Threads wait until a main thread signals them.

```java
class Signal {
    private boolean ready = false;

    public synchronized void waitForSignal() throws InterruptedException {
        while (!ready) {
            wait(); // wait for signal
        }
        System.out.println(Thread.currentThread().getName() + " received signal!");
    }

    public synchronized void sendSignal() {
        ready = true;
        notifyAll(); // wake all waiting threads
    }
}

public class SignalExample {
    public static void main(String[] args) throws InterruptedException {
        Signal signal = new Signal();

        for (int i = 1; i <= 3; i++) {
            new Thread(() -> {
                try {
                    signal.waitForSignal();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }, "Worker-" + i).start();
        }

        Thread.sleep(2000);
        System.out.println("Main thread sending signal...");
        signal.sendSignal();
    }
}
```

👉 Here:

* All worker threads **wait**.
* When `sendSignal()` is called, `notifyAll()` wakes them up.

---

# 📊 Summary Table

| Use Case                    | Method Used             | Explanation                                   |
| --------------------------- | ----------------------- | --------------------------------------------- |
| **Producer–Consumer**       | `wait()`, `notifyAll()` | Coordinate producers & consumers using buffer |
| **Thread Sequencing**       | `wait()`, `notifyAll()` | Ensure one thread runs after another          |
| **Worker Threads / Signal** | `wait()`, `notifyAll()` | Many workers wait until a signal is sent      |

---

✅ **Best Practices**:

* Always use `while` with `wait()` (not `if`) to avoid **spurious wakeups**.
* Prefer `notifyAll()` over `notify()` unless you’re sure only one waiting thread is enough.
* In modern Java, higher-level constructs (`BlockingQueue`, `CountDownLatch`, `Semaphore`, `CyclicBarrier`) are often preferred.

---

Would you like me to also **show how these same use cases look using `BlockingQueue` and `CountDownLatch`** (the modern replacements for wait/notify)?
Excellent question 👍 — exceptions in `wait()`, `notify()`, and `notifyAll()` are a very **important but often overlooked** part of Java concurrency. Let’s break it down carefully.

---

## ⚠️ Exceptions in `wait()`, `notify()`, `notifyAll()`

### 1. **IllegalMonitorStateException**

* Thrown when a thread calls `wait()`, `notify()`, or `notifyAll()` **without owning the object’s monitor lock**.
* In other words: they must be called inside a `synchronized` block/method on the same object.

**Example:**

```java
public class IllegalMonitorExample {
    public static void main(String[] args) {
        Object lock = new Object();

        // Wrong usage -> not inside synchronized(lock)
        try {
            lock.wait();
        } catch (InterruptedException e) {
            System.out.println("Interrupted!");
        }
    }
}
```

👉 Output:

```
Exception in thread "main" java.lang.IllegalMonitorStateException
```

---

### 2. **InterruptedException**

* Thrown when a thread is **waiting** (inside `wait()`) and another thread interrupts it (`thread.interrupt()`).
* This allows responsive cancellation of waiting threads.

**Example:**

```java
class Shared {
    public synchronized void doWait() {
        try {
            System.out.println(Thread.currentThread().getName() + " is waiting...");
            wait();  // will be interrupted
            System.out.println(Thread.currentThread().getName() + " resumed!");
        } catch (InterruptedException e) {
            System.out.println(Thread.currentThread().getName() + " was interrupted while waiting!");
        }
    }
}

public class InterruptedWaitExample {
    public static void main(String[] args) throws InterruptedException {
        Shared shared = new Shared();

        Thread t1 = new Thread(shared::doWait, "Worker");

        t1.start();
        Thread.sleep(2000);

        System.out.println("Main thread interrupting Worker...");
        t1.interrupt(); // causes InterruptedException in doWait()
    }
}
```

👉 Output:

```
Worker is waiting...
Main thread interrupting Worker...
Worker was interrupted while waiting!
```

---

### 3. **Spurious Wakeups** (not exception but related)

* A thread waiting can **wake up without notify/notifyAll being called**.
* That’s why we **always use `while` loop with wait()** instead of `if`.

**Correct usage:**

```java
synchronized(lock) {
    while (!condition) {
        lock.wait();
    }
    // safe to proceed
}
```

---

## 📊 Summary of Exceptions

| Method        | Possible Exception             | Scenario                         |
| ------------- | ------------------------------ | -------------------------------- |
| `wait()`      | `IllegalMonitorStateException` | Called without owning the lock   |
|               | `InterruptedException`         | Thread interrupted while waiting |
| `notify()`    | `IllegalMonitorStateException` | Called without owning the lock   |
| `notifyAll()` | `IllegalMonitorStateException` | Called without owning the lock   |

---

## ✅ Best Practices

* Always call `wait()` / `notify()` / `notifyAll()` **inside `synchronized(object)`**.
* Always handle `InterruptedException` properly → either clean up and exit, or rethrow.
* Always use `while (condition == false)` instead of `if` to guard against spurious wakeups.

---

👉 Now here’s a real-world **scenario of intercepted exception**:

* Imagine a worker thread is waiting for new tasks in a buffer (`wait()`).
* The system is shutting down, so the main thread **interrupts** all workers.
* Workers immediately throw `InterruptedException`, exit their loop, and shut down gracefully.

Would you like me to **extend the Producer-Consumer example** I gave earlier to include this shutdown-interrupt scenario? That way you’ll see how `InterruptedException` fits in a real design.
