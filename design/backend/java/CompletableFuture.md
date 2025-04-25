## **Complete Guide to CompletableFuture in Java**
`CompletableFuture` is a powerful feature introduced in **Java 8** to handle **asynchronous programming** more effectively. It extends `Future` and provides a non-blocking API for handling async tasks.

---

## **1. What is CompletableFuture?**
- `CompletableFuture` allows running tasks asynchronously.
- It provides **functional-style chaining** of tasks.
- Unlike `Future`, it allows **combining multiple async operations** and **handling errors**.
- Supports **callbacks** for when the task completes.

---

## **2. Creating a CompletableFuture**

### **2.1. Basic Asynchronous Execution**
`CompletableFuture.runAsync()` runs a task asynchronously but does **not return a result**.

```java
import java.util.concurrent.CompletableFuture;

public class CFExample {
    public static void main(String[] args) {
        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            System.out.println("Running in: " + Thread.currentThread().getName());
        });

        future.join(); // Waits for the completion of the task
    }
}
```
**Output:**
```
Running in: ForkJoinPool.commonPool-worker-1
```
---

### **2.2. Asynchronous Execution with a Result**
`CompletableFuture.supplyAsync()` runs a task asynchronously and **returns a result**.

```java
import java.util.concurrent.CompletableFuture;

public class CFExample {
    public static void main(String[] args) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            return "Hello from " + Thread.currentThread().getName();
        });

        System.out.println(future.join()); // Waits and retrieves the result
    }
}
```
**Output:**
```
Hello from ForkJoinPool.commonPool-worker-1
```
---

## **3. Chaining CompletableFutures**
### **3.1. Using `thenApply()` for Transformation**
- `thenApply()` transforms the result **(sync, blocking)**.

```java
import java.util.concurrent.CompletableFuture;

public class CFExample {
    public static void main(String[] args) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Hello")
                .thenApply(result -> result + " World")
                .thenApply(String::toUpperCase);

        System.out.println(future.join()); // Output: HELLO WORLD
    }
}
```

---

### **3.2. Using `thenAccept()` for Side Effects**
- `thenAccept()` consumes the result **(no return value)**.

```java
CompletableFuture.supplyAsync(() -> "Data Processed")
        .thenAccept(System.out::println);
```

---

### **3.3. Using `thenRun()` When No Result Needed**
- `thenRun()` runs a task **after previous completes, but doesn’t use the result**.

```java
CompletableFuture.supplyAsync(() -> "Task Completed")
        .thenRun(() -> System.out.println("Done!"));
```
---

### **3.4. Using `thenCompose()` for Dependent Futures**
- `thenCompose()` **flattens** nested `CompletableFuture` (sequential execution).

```java
import java.util.concurrent.CompletableFuture;

public class CFExample {
    public static void main(String[] args) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Hello")
                .thenCompose(result -> CompletableFuture.supplyAsync(() -> result + " World"));

        System.out.println(future.join()); // Output: Hello World
    }
}
```

---

### **3.5. Using `thenCombine()` for Parallel Tasks**
- `thenCombine()` combines results from two independent tasks.

```java
import java.util.concurrent.CompletableFuture;

public class CFExample {
    public static void main(String[] args) {
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> " World");

        CompletableFuture<String> result = future1.thenCombine(future2, (r1, r2) -> r1 + r2);
        System.out.println(result.join()); // Output: Hello World
    }
}
```

---

## **4. Handling Errors in CompletableFuture**
### **4.1. Using `exceptionally()` to Handle Exceptions**
- Catches exceptions and provides a fallback value.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (true) throw new RuntimeException("Error occurred!");
    return 42;
}).exceptionally(ex -> {
    System.out.println("Caught exception: " + ex.getMessage());
    return 0;
});

System.out.println(future.join()); // Output: Caught exception: Error occurred! 0
```

---

### **4.2. Using `handle()` for Exception Handling & Recovery**
- Unlike `exceptionally()`, `handle()` runs **even when there's no exception**.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (true) throw new RuntimeException("Error occurred!");
    return 42;
}).handle((result, ex) -> {
    if (ex != null) {
        System.out.println("Handled exception: " + ex.getMessage());
        return 0;
    }
    return result;
});

System.out.println(future.join()); // Output: Handled exception: Error occurred! 0
```

---

## **5. Running Multiple CompletableFutures**
### **5.1. Running Multiple Tasks in Parallel with `allOf()`**
- Runs multiple `CompletableFutures` in parallel and waits for all to complete.

```java
CompletableFuture<Void> future1 = CompletableFuture.runAsync(() -> System.out.println("Task 1"));
CompletableFuture<Void> future2 = CompletableFuture.runAsync(() -> System.out.println("Task 2"));

CompletableFuture<Void> allTasks = CompletableFuture.allOf(future1, future2);
allTasks.join(); // Waits for all tasks to complete
```

---

### **5.2. Running Multiple Tasks & Getting the First One to Complete with `anyOf()`**
- Returns the result of the first completed task.

```java
CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Task 1");
CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "Task 2");

CompletableFuture<Object> firstCompleted = CompletableFuture.anyOf(future1, future2);
System.out.println(firstCompleted.join()); // Output: Task 1 OR Task 2
```

---

## **6. Custom Thread Pools in CompletableFuture**
- By default, `CompletableFuture` uses `ForkJoinPool.commonPool()`.
- We can specify a custom `ExecutorService`:

```java
import java.util.concurrent.*;

public class CustomThreadPoolExample {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(2);

        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            System.out.println("Task executed in: " + Thread.currentThread().getName());
        }, executor);

        future.join();
        executor.shutdown();
    }
}
```
**Output:**
```
Task executed in: pool-1-thread-1
```

---

## **7. Summary of CompletableFuture Methods**

| Method | Description |
|--------|-------------|
| `supplyAsync()` | Runs a task asynchronously and returns a result. |
| `runAsync()` | Runs a task asynchronously **without a result**. |
| `thenApply()` | Transforms the result. |
| `thenAccept()` | Consumes the result (no return value). |
| `thenRun()` | Runs after completion (no result used). |
| `thenCompose()` | Flattens nested `CompletableFuture` (sequential). |
| `thenCombine()` | Combines results of two `CompletableFuture`. |
| `exceptionally()` | Handles exceptions and provides a fallback value. |
| `handle()` | Handles both success and failure cases. |
| `allOf()` | Waits for all `CompletableFutures` to complete. |
| `anyOf()` | Returns the first completed `CompletableFuture`. |

---

## **Conclusion**
- `CompletableFuture` **simplifies async programming** in Java.
- It supports **chaining, parallel execution, error handling**, and **custom thread pools**.
- `CompletableFuture` is used in **Web Services, Database Calls, File Processing**, and more.

Would you like me to provide more advanced examples or debugging techniques?
