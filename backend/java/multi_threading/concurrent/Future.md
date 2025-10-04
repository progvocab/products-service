

The **most important use case of `Future` in Java** is:

## 🔹 **Managing the result of an asynchronous computation**

When you submit a **`Callable`** task to an `ExecutorService`, instead of blocking the main thread, it immediately returns a `Future<T>` object.

* You can continue doing other work.
* Later, you can use the `Future` to **get the result**, **check completion**, or **cancel the task**.

---

## 🔹 Example – Fetching Data in Parallel

Imagine you want to **fetch employee details from a database** and **salary details from an external API** in parallel, then combine them.

```java
import java.util.concurrent.*;

public class FutureExample {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);

        Callable<String> employeeTask = () -> {
            Thread.sleep(2000); // simulate DB call
            return "Employee: Alice";
        };

        Callable<String> salaryTask = () -> {
            Thread.sleep(3000); // simulate API call
            return "Salary: $5000";
        };

        Future<String> employeeFuture = executor.submit(employeeTask);
        Future<String> salaryFuture = executor.submit(salaryTask);

        // Do some other work here while tasks run...
        System.out.println("Fetching data in parallel...");

        // Now block only when needed
        String employee = employeeFuture.get(); // waits if not done
        String salary = salaryFuture.get();

        System.out.println(employee + ", " + salary);

        executor.shutdown();
    }
}
```

### Output:

```
Fetching data in parallel...
Employee: Alice, Salary: $5000
```

---

## 🔹 Why this is the **key use case**

1. **Parallelism** – run multiple independent tasks at once.
2. **Non-blocking start** – you don’t block the main thread when tasks are submitted.
3. **Result management** – you can:

   * `get()` → retrieve result (blocking wait).
   * `isDone()` → check if completed.
   * `cancel()` → cancel if not started.

---

✅ **In short**:
The main use case of `Future` is to **submit long-running background tasks and retrieve their result later, without blocking immediately**.

---

👉 Would you like me to also show you how this becomes more powerful with **`CompletableFuture`** (introduced in Java 8), which allows chaining and non-blocking callbacks?
