A **race condition** occurs when **two or more threads access shared data simultaneously**, and **the program’s output depends on the timing of their execution**.

Because thread scheduling is unpredictable, one thread might **read or modify** a shared variable **before another finishes updating it**, leading to **inconsistent or incorrect results**.

✅ **Example:**

```java
counter++;  // Not atomic — read, increment, write
```

If two threads run this line at the same time, both may read the same old value of `counter`, causing a lost update.

**Solution:** Use synchronization mechanisms like `synchronized`, `Lock`, or atomic classes (`AtomicInteger`, etc.) to make access thread-safe.
