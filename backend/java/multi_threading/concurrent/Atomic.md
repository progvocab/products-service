# Atomic 

One of the **classic and most important use cases** of `AtomicInteger` or `AtomicLong` is **thread-safe counters without using locks**.

### Use Cases

* Counting requests per second (RPS)
* Tracking retries in a background job
* Generating unique IDs in multi-threaded systems


##   Use Case: Request Counter in a Web Server

Imagine a Spring Boot or plain Java server where multiple threads handle incoming requests.
We want to **count total requests** served.

If we use a normal `int` or `long`, we would need synchronization → slower performance.
Instead, `AtomicInteger` / `AtomicLong` allows **lock-free thread-safe increments**.

###  Non-Atomic Counter - Race Condition 

###  Problem
```java
public class NonAtomicCounter {
    private int counter = 0;

    public void increment() {
        counter++; // Not atomic!
    }

    public int getCount() {
        return counter;
    }

    public static void main(String[] args) throws InterruptedException {
        NonAtomicCounter nac = new NonAtomicCounter();

        Runnable task = () -> {
            for (int i = 0; i < 1000; i++) {
                nac.increment();
            }
        };

        // 10 threads incrementing
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(task);
            threads[i].start();
        }

        for (Thread t : threads) t.join();

        System.out.println("Final count (expected 10000): " + nac.getCount());
    }
}
```

- Even though we expect `10000`, the result may be **less** (e.g., `9324`).
- This is because `counter++` is **not atomic** (it’s effectively `read → increment → write`).

We can use synchronized to fix this issue 
```java
public synchronized void increment() {
    counter++;
}
```
- This fixes the race condition because only one thread can enter increment() at a time.
- But it introduces a lock , which leads to slower performance under high concurrency compared to AtomicInteger.

###  AtomicInteger Counter (Thread-Safe)
###  Solution
```java
import java.util.concurrent.atomic.AtomicInteger;

public class AtomicCounter {
    private final AtomicInteger counter = new AtomicInteger(0);

    public void increment() {
        counter.incrementAndGet(); // Atomic!
    }

    public int getCount() {
        return counter.get();
    }

    public static void main(String[] args) throws InterruptedException {
        AtomicCounter ac = new AtomicCounter();

        Runnable task = () -> {
            for (int i = 0; i < 1000; i++) {
                ac.increment();
            }
        };

        // 10 threads incrementing
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(task);
            threads[i].start();
        }

        for (Thread t : threads) t.join();

        System.out.println("Final count (expected 10000): " + ac.getCount());
    }
}
```

- `incrementAndGet()` is **atomic** → no race conditions.
- Multiple threads can safely update the counter without locks.
- Used in **metrics, rate-limiting, ID generation, retries, queues**.

- The output will **always be 10000**, because `incrementAndGet()` ensures atomicity using **CAS (Compare-And-Swap)** under the hood.


| Counter Type    | Thread-Safe? | Output Consistency     |
| --------------- | ------------ | ---------------------- |
| `int` / `long`  | ❌ No         | Wrong (race condition) 🏃🏃🏃|
| `synchronized increment` | ✅ Yes        | Slower under high load ⏳|
| `AtomicInteger` | ✅ Yes        | Correct, consistent and fast  🚀 |




This tiny example is why **Atomic classes** are widely used in **counters, statistics, ID generators, retries, concurrency utilities (like `ConcurrentHashMap`)**.


---

## Working of AtomicInteger

###  CAS  Compare And Swap 

`AtomicInteger` uses **CPU instructions (CAS)** provided by `Unsafe` in the JDK.
CAS works like this:

```text
CAS(memory_location, expected_value, new_value)
```

* Check if the value at `memory_location` equals `expected_value`.
* If **yes**, set it to `new_value` (success).
* If **no**, do nothing (fail) — retry needed.

 This ensures **atomicity without locking**.
- Thread reads the current value (say oldValue) into its local variable / CPU register.
- This "expectedValue" is what the thread thinks is in memory.
- This happens before calling CAS.
- CAS instruction at the CPU level checks:
- Does the value at the memory address (the actual heap memory, e.g., inside AtomicInteger.value) equal expectedValue?
- If yes → replace it with newValue (atomic update).
- If no → do nothing, CAS returns false.
- If CAS fails, the thread usually retries (loops until success).

###  Example with `AtomicInteger`

```java
import java.util.concurrent.atomic.AtomicInteger;

public class AtomicCASDemo {
    private static AtomicInteger counter = new AtomicInteger(0);

    public static void main(String[] args) {
        Runnable task = () -> {
            for (int i = 0; i < 5; i++) {
                int oldValue, newValue;
                do {
                    oldValue = counter.get();       // read
                    newValue = oldValue + 1;        // calculate
                } while (!counter.compareAndSet(oldValue, newValue)); // CAS
                System.out.println(Thread.currentThread().getName() + " -> " + newValue);
            }
        };

        new Thread(task, "T1").start();
        new Thread(task, "T2").start();
    }
}
```

---

###   Sample Execution

```
T1 -> 1
T2 -> 2
T1 -> 3
T2 -> 4
T1 -> 5
T2 -> 6
...
```

Here `compareAndSet()` retries until it succeeds, so increments are **safe without locks**.



###  How `incrementAndGet()` is Implemented Internally

If you peek into OpenJDK’s `AtomicInteger.java`:

```java
public final int incrementAndGet() {
    return unsafe.getAndAddInt(this, valueOffset, 1) + 1;
}
```

Where `unsafe.getAndAddInt()` → is implemented with **CAS loop** at the JVM/CPU level.



###   Why is CAS better than Locks?

* **Non-blocking** → no thread suspension, better performance under high contention.
* **Scales well** for multicore CPUs.
* Avoids **deadlocks** (locks may cause them).
* `AtomicInteger` achieves atomicity using **CAS (Compare-And-Swap)**.
* CAS ensures that increments happen safely **without synchronization blocks**.

 

