### `await()`

* Causes the current thread to **release the lock and wait** on the associated `Condition`.
* The thread remains blocked until another thread calls `signal()` or `signalAll()`.
* Before returning from `await()`, the thread **re-acquires the lock**.

### `signal()`

* Wakes up **one waiting thread** from the `Condition`'s wait queue.
* The awakened thread does not run immediately; it must first re-acquire the lock.
* Similar to `notify()` in the traditional monitor model.

### `signalAll()`

* Wakes up **all waiting threads** on the `Condition`.
* Each awakened thread competes to re-acquire the lock.
* Similar to `notifyAll()` in the traditional monitor model.

---

## Demonstration Program

```java
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class ConditionDemo {

    private static final ReentrantLock lock = new ReentrantLock();
    private static final Condition condition = lock.newCondition();

    static class Worker extends Thread {

        public Worker(String name) {
            super(name);
        }

        @Override
        public void run() {
            lock.lock();
            try {
                System.out.println(getName() + " waiting...");
                condition.await();

                System.out.println(getName() + " resumed");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                lock.unlock();
            }
        }
    }

    public static void main(String[] args) throws Exception {

        Worker t1 = new Worker("T1");
        Worker t2 = new Worker("T2");
        Worker t3 = new Worker("T3");

        t1.start();
        t2.start();
        t3.start();

        Thread.sleep(2000);

        lock.lock();
        try {
            System.out.println("\nCalling signal()");
            condition.signal();      // wakes one thread
        } finally {
            lock.unlock();
        }

        Thread.sleep(2000);

        lock.lock();
        try {
            System.out.println("\nCalling signalAll()");
            condition.signalAll();   // wakes remaining threads
        } finally {
            lock.unlock();
        }
    }
}
```

### Possible Output

```text
T1 waiting...
T2 waiting...
T3 waiting...

Calling signal()
T1 resumed

Calling signalAll()
T2 resumed
T3 resumed
```

### Interview One-Liner

| Method        | Effect                   |
| ------------- | ------------------------ |
| `await()`     | Release lock and wait    |
| `signal()`    | Wake one waiting thread  |
| `signalAll()` | Wake all waiting threads |

**Important:** `await()`, `signal()`, and `signalAll()` must be called while holding the associated lock, otherwise an `IllegalMonitorStateException` is thrown.
