
**Thread contention** happens when multiple threads try to access the **same shared resource** (like a variable, lock, or I/O) at the same time.
Only one thread can hold the lock, so others must **wait or block**, causing **performance slowdown** and **context switching overhead**.

### Best Practices 
Minimize shared state and synchronization.
Use **concurrent utilities** (`ConcurrentHashMap`, `Atomic*` classes), **lock-free algorithms**, or **immutable objects**.
Prefer **thread-local storage**, **parallel streams**, or **CompletableFuture** to reduce direct locking and contention.
