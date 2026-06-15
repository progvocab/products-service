

*non-synchronized* method does **not** acquire or associate with a monitor lock when it runs.

Only **synchronized** methods (or synchronized blocks) use an object’s **monitor** (lock) mechanism.


### What is a Monitor?

A **monitor** is a synchronization construct used by the JVM to enforce *mutual exclusion* and *coordination* between threads.

Every Java object has:

* a **monitor** (used by `synchronized`)
* and a **Mark Word** in its object header (which can store lock state, identity hash, age, etc.)

 

### What Happens in a Synchronized vs Non-Synchronized Method

### Non-synchronized method

```java
class Example {
    void print() { // Not synchronized
        System.out.println("Non-synchronized");
    }
}
```

* Any thread can call `print()` at any time.
* The JVM **does not** attempt to acquire or release the object’s monitor.
* The **Mark Word** of the object stays in the **unlocked** state (or possibly biased state if enabled).
* There’s **no blocking**, **no ownership**, and **no waiting queue**.

 

### Synchronized instance method

```java
class Example {
    synchronized void print() {
        System.out.println("Synchronized method");
    }
}
```

* The JVM implicitly executes something like:

```java
monitorenter(this);
try {
    // method body
} finally {
    monitorexit(this);
}
```

* The executing thread must **acquire the object’s monitor** before entering.
* Other threads attempting to call any other synchronized method on the same object will block until the monitor is released.
* The **Mark Word** of the object is updated to indicate the lock state.



### Mark Word States (Object Header View)

The **Mark Word** in a 64-bit JVM typically looks like this (simplified):

| Lock State           | Mark Word bits (simplified) | Description |                                      |                                                         |
| -------------------- | --------------------------- | ----------- | ------------------------------------ | ------------------------------------------------------- |
| **Unlocked**         | `[hashcode                  | age         | 01]`                                 | Default state, no synchronization                       |
| **Biased Lock**      | `[thread id                 | epoch       | 01]`                                 | Optimized for one thread repeatedly locking same object |
| **Lightweight Lock** | `[ptr to lock record        | 00]`        | When thread contends for the monitor |                                                         |
| **Heavyweight Lock** | `[ptr to monitor            | 10]`        | When contention escalates            |                                                         |
| **Marked for GC**    | `[forwarding ptr]`          | During GC   |                                      |                                                         |

➡️ **Non-synchronized methods** leave the mark word in *Unlocked* or *Biased* state.
➡️ **Synchronized methods** cause the mark word to temporarily change to *Lightweight* or *Heavyweight*.

 

| Aspect                   | Non-synchronized method             | Synchronized method                        |
| ------------------------ | ----------------------------------- | ------------------------------------------ |
| Acquires monitor?        | ❌ No                                | ✅ Yes                                      |
| Affects Mark Word?       | ❌ No                                | ✅ Yes (lock bits change)                   |
| Can block other threads? | ❌ No                                | ✅ Yes                                      |
| Allows wait/notify?      | ❌ No (IllegalMonitorStateException) | ✅ Yes (only when owning monitor)           |
| Typical use case         | Non-critical sections               | Critical sections needing mutual exclusion |

 

You **cannot call `wait()`, `notify()`, or `notifyAll()`** inside a non-synchronized method on the same object —
because those methods require ownership of the monitor.

Otherwise, you’ll get:

```
java.lang.IllegalMonitorStateException
```



### Java Locks

* Every Java object has a **mark word** in its header.
* The **lock state of the object** is encoded in the mark word (2 bits typically).
* JVM supports **three main lock states**:

  1. **No lock (unlocked)**
  2. **Lightweight lock** (thin lock)
  3. **Heavyweight lock** (fat lock / monitor)
  4. **Biased lock** (optional optimization)



### Lightweight Lock (Thin Lock)

* Used when **no contention exists** (only one thread trying to lock the object).
* Implemented via **CAS (Compare-And-Swap) on the mark word**.
* **Fast** because it avoids kernel-level operations.
* No OS mutex involved, purely in **user-space**.

**How it works:**

1. Thread tries to acquire the lock.
2. JVM stores **thread ID + pointer to stack frame** in mark word.
3. If successful, thread enters the synchronized block.
4. Unlock = restore mark word to previous state.

**Advantages:**

* Very fast for uncontended locks.
* Low memory overhead.

**Disadvantages:**

* If **another thread attempts to acquire the lock simultaneously**, it must **inflate the lock** into a heavyweight lock.



### Heavyweight Lock (Fat Lock / Monitor)

* Used when **multiple threads contend** for the same lock.
* Implemented using **OS-level mutex / monitor**.
* More expensive because it may involve **context switches** and **kernel calls**.
* The mark word is updated to point to a **monitor structure**.

**How it works:**

1. When contention is detected, JVM creates a **monitor object**.
2. Threads trying to acquire the lock block (park) until lock is free.
3. Monitor keeps track of waiting threads.

**Advantages:**

* Handles **high contention** reliably.

**Disadvantages:**

* Slower than lightweight lock due to kernel involvement.
* Memory overhead for monitor object.



### JVM Lock Transition

Java uses a **lock escalation strategy**:

```
Biased Lock (optional) --> Lightweight Lock --> Heavyweight Lock
```

* **Biased lock** → avoids CAS if the same thread repeatedly acquires lock.
* **Lightweight lock** → fast path for uncontended locks.
* **Heavyweight lock** → fallback for contention.



### Mark Word Representation

| Lock Type   | Mark Word (64-bit)                                                  |
| ----------- | ------------------------------------------------------------------- |
| Unlocked    | Default mark word, lock bits = 01                                   |
| Lightweight | Mark word stores thread ID + pointer to stack frame, lock bits = 00 |
| Heavyweight | Mark word points to **monitor object**, lock bits = 10              |
| Biased      | Thread ID encoded in mark word, lock bits = 01 (biased bit set)     |



| Feature         | Lightweight Lock                        | Heavyweight Lock                     |
| --------------- | --------------------------------------- | ------------------------------------ |
| Contention      | Low / none                              | High / multiple threads              |
| Implementation  | CAS on mark word (user-space)           | Monitor / OS mutex (kernel)          |
| Speed           | Very fast                               | Slower (blocking + context switches) |
| Memory overhead | Low                                     | Higher (monitor object)              |
| Usage           | Optimized path for single-threaded lock | Escalation for contention            |

 

  

* Lightweight locks = **fast, CAS-based, user-space**.
* Heavyweight locks = **kernel-based, slow, used only under contention**.
* HotSpot tries to **use the lightest lock possible** to improve performance.

 
