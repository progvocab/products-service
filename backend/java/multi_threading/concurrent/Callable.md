Great question 👍 Callable and Runnable look similar at first glance, but they solve different problems in **concurrent programming**.

---

## 🔹 Runnable

* Introduced in **Java 1.0**.
* Represents a **task that can be executed** by a thread.
* Does **not return a result** and cannot throw checked exceptions.

### Example:

```java
Runnable task = () -> {
    System.out.println("Running a task");
};
new Thread(task).start();
```

* `run()` returns **void**.
* Used when you just need to **execute code in parallel**.

---

## 🔹 Callable

* Introduced in **Java 5** (as part of `java.util.concurrent`).
* Represents a **task that returns a result** and may throw a checked exception.
* Designed to work with **ExecutorService** and **Future**.

### Example:

```java
import java.util.concurrent.*;

Callable<Integer> task = () -> {
    System.out.println("Computing...");
    return 42; // returns a value
};

ExecutorService executor = Executors.newSingleThreadExecutor();
Future<Integer> future = executor.submit(task);

System.out.println("Result = " + future.get()); // waits for result
executor.shutdown();
```

* `call()` returns a **value (generic type T)**.
* Can throw **checked exceptions**.

---

## 🔹 Key Differences

| Feature                       | Runnable                              | Callable                   |
| ----------------------------- | ------------------------------------- | -------------------------- |
| **Method**                    | `public void run()`                   | `public T call()`          |
| **Returns a value**           | ❌ No                                  | ✅ Yes (generic type `T`)   |
| **Throws checked exceptions** | ❌ No                                  | ✅ Yes                      |
| **Introduced in**             | Java 1.0                              | Java 5                     |
| **Usage**                     | `Thread`, `ExecutorService.execute()` | `ExecutorService.submit()` |

---

## 🔑 Summary

* Use **Runnable** → when you just want to execute code in another thread, no result needed.
* Use **Callable** → when you want the task to produce a result or throw exceptions.

---

👉 Do you want me to also show you how **Callable + Future** works with **parallel execution** (like submitting multiple tasks and collecting results)?
