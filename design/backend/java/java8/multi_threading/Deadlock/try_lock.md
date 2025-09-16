Let’s go step by step into **`tryLock(timeout)` in Java**.

---

# 🔹 What is `tryLock(timeout)`?

* Defined in **`java.util.concurrent.locks.Lock`** (commonly used with `ReentrantLock`).
* Unlike `lock()`, which **blocks indefinitely** until the lock is available,
  `tryLock(timeout, unit)` **waits for a limited time**.

👉 If the lock is acquired within the given time → returns `true`.
👉 If the lock is not acquired before timeout → returns `false`.

This helps **prevent deadlocks**, since a thread won’t wait forever.

---

# 🔹 Syntax

```java
boolean tryLock(long timeout, TimeUnit unit) throws InterruptedException
```

* `timeout`: how long to wait.
* `unit`: `TimeUnit.SECONDS`, `TimeUnit.MILLISECONDS`, etc.
* Returns:

  * `true` → Lock acquired.
  * `false` → Could not acquire within timeout.

---

# 🔹 Example 1: Simple Usage

```java
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class TryLockExample {
    private final ReentrantLock lock = new ReentrantLock();

    public void doWork() {
        try {
            if (lock.tryLock(2, TimeUnit.SECONDS)) {  // wait up to 2 seconds
                try {
                    System.out.println(Thread.currentThread().getName() + " acquired lock");
                    Thread.sleep(3000); // simulate work
                } finally {
                    lock.unlock();
                    System.out.println(Thread.currentThread().getName() + " released lock");
                }
            } else {
                System.out.println(Thread.currentThread().getName() + " could not acquire lock");
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        TryLockExample example = new TryLockExample();

        Runnable task = example::doWork;

        new Thread(task, "Thread-1").start();
        new Thread(task, "Thread-2").start();
    }
}
```

### 📝 Output (sample):

```
Thread-1 acquired lock
Thread-2 could not acquire lock
Thread-1 released lock
```

* Thread-1 holds the lock for 3s.
* Thread-2 tries for 2s → fails → avoids deadlock.

---

# 🔹 Example 2: Preventing Deadlock

Suppose two threads need two locks:

```java
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class DeadlockPrevention {
    private static final ReentrantLock lock1 = new ReentrantLock();
    private static final ReentrantLock lock2 = new ReentrantLock();

    public static void main(String[] args) {
        Runnable task1 = () -> {
            try {
                if (lock1.tryLock(1, TimeUnit.SECONDS)) {
                    try {
                        Thread.sleep(50); // simulate work
                        if (lock2.tryLock(1, TimeUnit.SECONDS)) {
                            try {
                                System.out.println("Task1 acquired both locks");
                            } finally {
                                lock2.unlock();
                            }
                        } else {
                            System.out.println("Task1 could not acquire lock2, avoiding deadlock");
                        }
                    } finally {
                        lock1.unlock();
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        };

        Runnable task2 = () -> {
            try {
                if (lock2.tryLock(1, TimeUnit.SECONDS)) {
                    try {
                        Thread.sleep(50);
                        if (lock1.tryLock(1, TimeUnit.SECONDS)) {
                            try {
                                System.out.println("Task2 acquired both locks");
                            } finally {
                                lock1.unlock();
                            }
                        } else {
                            System.out.println("Task2 could not acquire lock1, avoiding deadlock");
                        }
                    } finally {
                        lock2.unlock();
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        };

        new Thread(task1).start();
        new Thread(task2).start();
    }
}
```

### 📝 Possible Output:

```
Task1 acquired both locks
Task2 could not acquire lock1, avoiding deadlock
```

✅ This avoids deadlock because if one thread fails to acquire the second lock within timeout, it releases the first lock.

---

# 🔹 Key Points

* **`lock()`** → waits forever → risk of deadlock.
* **`tryLock()` without timeout** → immediately returns if lock unavailable.
* **`tryLock(timeout, unit)`** → waits up to given time → safer for concurrent systems.

---

👉 Do you want me to also explain **how `tryLock()` differs from using `synchronized` blocks** (since synchronized doesn’t have timeout behavior)?

Perfect 👍 Let’s carefully compare **`tryLock()` vs `synchronized`** in Java — since both are ways to protect shared resources, but they behave differently.

---

# 🔹 1. **Acquisition Behavior**

| Feature          | `synchronized`                                          | `tryLock()` (ReentrantLock)                                               |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| Lock acquisition | Thread **blocks indefinitely** until lock is available. | `tryLock()` returns **immediately** (`true` if acquired, `false` if not). |
| With timeout     | ❌ Not supported. Always waits.                          | ✅ `tryLock(timeout, unit)` waits for limited time.                        |
| Fairness         | No fairness control (depends on JVM/OS scheduler).      | Can create **fair lock** with `new ReentrantLock(true)`.                  |

---

# 🔹 2. **Interruptibility**

* `synchronized`:

  * A thread waiting for a synchronized block **cannot be interrupted**.
  * It must wait until lock is released.

* `ReentrantLock.tryLock(timeout)`:

  * Supports **interruptible waits**.
  * You can break out of waiting using `thread.interrupt()`.

---

# 🔹 3. **Reentrancy**

* Both are **reentrant**:

  * A thread holding the lock can reacquire it without blocking itself.

---

# 🔹 4. **Unlocking**

* `synchronized`: Lock is **automatically released** when the synchronized block/method exits.
* `ReentrantLock`: You must call `unlock()` explicitly in a `finally` block. Forgetting unlock → deadlock risk.

---

# 🔹 5. **Code Examples**

### Using `synchronized`

```java
public class SyncExample {
    private final Object lock = new Object();

    public void doWork() {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " acquired lock");
            try { Thread.sleep(2000); } catch (InterruptedException e) {}
        } // Lock automatically released here
    }
}
```

👉 If another thread enters `doWork()`, it **waits forever** until lock is released.

---

### Using `tryLock(timeout)`

```java
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class TryLockExample {
    private final ReentrantLock lock = new ReentrantLock();

    public void doWork() {
        try {
            if (lock.tryLock(1, TimeUnit.SECONDS)) {
                try {
                    System.out.println(Thread.currentThread().getName() + " acquired lock");
                    Thread.sleep(2000);
                } finally {
                    lock.unlock();
                }
            } else {
                System.out.println(Thread.currentThread().getName() + " could not acquire lock");
            }
        } catch (InterruptedException e) {
            System.out.println("Thread interrupted while waiting for lock");
        }
    }
}
```

👉 Here a thread will **only wait 1s**. If lock not acquired → it moves on (avoiding deadlock).

---

# 🔹 6. **When to Use What**

* ✅ Use **`synchronized`** if:

  * Simple locking is enough.
  * No need for timeout / interruptibility.
  * You want automatic lock release (less error-prone).

* ✅ Use **`ReentrantLock.tryLock(timeout)`** if:

  * You want to **avoid deadlocks** (by backing off after timeout).
  * You need **fairness policies**.
  * You want to be able to **interrupt threads** waiting for locks.

---

✅ **Summary**:

* `synchronized` → simple, safe, but inflexible.
* `tryLock(timeout)` → more control, avoids deadlock, but requires manual unlock.

---

👉 Would you like me to also show you a **real-world example** (like a bank account transfer) where `synchronized` can cause **deadlock**, but `tryLock(timeout)` solves it?


Awesome 👍 Let’s walk through a **real-world example: Bank Account Transfer** — where `synchronized` can cause **deadlock**, and how `tryLock(timeout)` solves it.

---

# 🔹 Problem: Transfer Between Two Accounts

Imagine two threads:

* **Thread 1**: Transfer money from Account A → B
* **Thread 2**: Transfer money from Account B → A

If both try at the same time:

* Thread 1 locks Account A, waits for B
* Thread 2 locks Account B, waits for A
  👉 **Deadlock**.

---

# 🔹 Version 1: Using `synchronized` (Deadlock Possible)

```java
public class Account {
    private int balance;
    public Account(int balance) { this.balance = balance; }

    public void deposit(int amount) { balance += amount; }
    public void withdraw(int amount) { balance -= amount; }
    public int getBalance() { return balance; }
}

public class Bank {
    public void transfer(Account from, Account to, int amount) {
        synchronized (from) {
            synchronized (to) {
                from.withdraw(amount);
                to.deposit(amount);
                System.out.println(Thread.currentThread().getName() + " transferred " + amount);
            }
        }
    }

    public static void main(String[] args) {
        Bank bank = new Bank();
        Account a = new Account(1000);
        Account b = new Account(1000);

        // Thread 1: A -> B
        new Thread(() -> bank.transfer(a, b, 100), "Thread-1").start();

        // Thread 2: B -> A
        new Thread(() -> bank.transfer(b, a, 200), "Thread-2").start();
    }
}
```

⚠️ **Risk**:
If Thread-1 holds `a` and Thread-2 holds `b`, both wait forever → **deadlock**.

---

# 🔹 Version 2: Using `tryLock(timeout)` (Deadlock Avoidance)

```java
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class Account {
    private int balance;
    private final ReentrantLock lock = new ReentrantLock();

    public Account(int balance) { this.balance = balance; }

    public void deposit(int amount) { balance += amount; }
    public void withdraw(int amount) { balance -= amount; }
    public int getBalance() { return balance; }

    public ReentrantLock getLock() { return lock; }
}

public class Bank {
    public boolean transfer(Account from, Account to, int amount) {
        try {
            if (from.getLock().tryLock(1, TimeUnit.SECONDS)) { // lock from
                try {
                    if (to.getLock().tryLock(1, TimeUnit.SECONDS)) { // lock to
                        try {
                            from.withdraw(amount);
                            to.deposit(amount);
                            System.out.println(Thread.currentThread().getName() + 
                                               " transferred " + amount);
                            return true;
                        } finally {
                            to.getLock().unlock();
                        }
                    } else {
                        System.out.println(Thread.currentThread().getName() + 
                                           " could not acquire lock on target account");
                    }
                } finally {
                    from.getLock().unlock();
                }
            } else {
                System.out.println(Thread.currentThread().getName() + 
                                   " could not acquire lock on source account");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return false;
    }

    public static void main(String[] args) {
        Bank bank = new Bank();
        Account a = new Account(1000);
        Account b = new Account(1000);

        // Thread 1: A -> B
        new Thread(() -> bank.transfer(a, b, 100), "Thread-1").start();

        // Thread 2: B -> A
        new Thread(() -> bank.transfer(b, a, 200), "Thread-2").start();
    }
}
```

---

# 🔹 How This Prevents Deadlock

* If Thread-1 can’t acquire both locks within 1 second → it backs off and releases its lock.
* Thread-2 can then proceed.
* At worst, one transfer **fails temporarily**, but the system doesn’t freeze.

---

✅ **Summary**:

* With `synchronized`, deadlock is possible if threads acquire locks in opposite order.
* With `tryLock(timeout)`, threads don’t wait forever → they retry or fail gracefully.

---

👉 Do you want me to also add a **retry mechanism** (so if a transfer fails due to lock timeout, it retries automatically) to make it more robust?

