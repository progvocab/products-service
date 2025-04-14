# **Understanding `CompletionStage` in Java (`CompletableFuture`)**  

## **What is `CompletionStage`?**
- `CompletionStage<T>` is an interface that **represents a step in an asynchronous computation**.  
- **Does not block** the calling thread.  
- Used **internally** by `CompletableFuture<T>`.  
- Supports **chaining, combining, and error handling** for async tasks.  

## **Key Differences:**

| Feature | `Future` | `CompletableFuture` (`CompletionStage`) |
|---------|----------|------------------|
| **Blocking** | `get()` blocks the thread | Supports **non-blocking async chaining** |
| **Chaining** | Not supported | Supports **thenApply, thenCompose, thenCombine** |
| **Error Handling** | No built-in support | Provides **exceptionally() and handle()** |

---

# **1. Creating a CompletionStage (`CompletableFuture`)**
### **Basic Example:**
```java
import java.util.concurrent.*;

public class CompletionStageExample {
    public static void main(String[] args) {
        CompletionStage<String> stage = CompletableFuture.supplyAsync(() -> "Hello, CompletionStage!");

        stage.thenAccept(System.out::println); // Prints asynchronously
    }
}
```
**Output:**  
```
Hello, CompletionStage!
```
- `supplyAsync()` runs in the background.  
- `thenAccept()` consumes the result **without returning anything**.  

---

# **2. Transforming Data (`thenApply()`)**
- `thenApply()` **transforms the result** **(sync, blocking)**.

### **Example:**
```java
CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Java")
        .thenApply(data -> data + " 17")
        .thenApply(String::toUpperCase);

System.out.println(future.join()); // Output: JAVA 17
```
- Chained transformations occur **sequentially**.  

---

# **3. Chaining Async Tasks (`thenCompose()`)**
- `thenCompose()` **flattens nested `CompletionStage`**.  
- Useful for **dependent async calls**.

### **Example:**
```java
CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "User123")
        .thenCompose(userId -> getUserDetails(userId));

System.out.println(future.join());

static CompletableFuture<String> getUserDetails(String userId) {
    return CompletableFuture.supplyAsync(() -> "User Details for " + userId);
}
```
**Output:**  
```
User Details for User123
```
- **First Future**: Fetches `userId`.  
- **Second Future**: Fetches details **using the first result**.  

---

# **4. Combining Multiple Futures (`thenCombine()`)**
- `thenCombine()` **combines results from two independent async tasks**.

### **Example:**
```java
CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> " World");

CompletableFuture<String> combined = future1.thenCombine(future2, (a, b) -> a + b);

System.out.println(combined.join()); // Output: Hello World
```
- Both tasks **run in parallel**, and the results are merged.  

---

# **5. Running Independent Async Tasks (`allOf() & anyOf()`)**
### **5.1. `allOf()`: Wait for All Tasks**
```java
CompletableFuture<Void> allTasks = CompletableFuture.allOf(
        CompletableFuture.runAsync(() -> System.out.println("Task 1")),
        CompletableFuture.runAsync(() -> System.out.println("Task 2"))
);
allTasks.join(); // Waits for all tasks to finish
```
**Output:**  
```
Task 1
Task 2
```
---

### **5.2. `anyOf()`: Get First Completed Task**
```java
CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Result 1");
CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "Result 2");

CompletableFuture<Object> firstCompleted = CompletableFuture.anyOf(future1, future2);
System.out.println(firstCompleted.join()); // Output: Result 1 OR Result 2
```
- **Returns the first completed result** (non-deterministic).  

---

# **6. Handling Errors (`exceptionally()` & `handle()`)**
### **6.1. `exceptionally()`: Provide Fallback on Error**
```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (true) throw new RuntimeException("Something went wrong!");
    return 10;
}).exceptionally(ex -> {
    System.out.println("Caught Exception: " + ex.getMessage());
    return 0; // Fallback value
});

System.out.println(future.join()); // Output: Caught Exception: Something went wrong! 0
```

---

### **6.2. `handle()`: Process Success & Failure**
```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (true) throw new RuntimeException("Failure!");
    return 10;
}).handle((result, ex) -> {
    if (ex != null) {
        System.out.println("Handled Exception: " + ex.getMessage());
        return -1; // Default value
    }
    return result;
});

System.out.println(future.join()); // Output: Handled Exception: Failure! -1
```
- **Unlike `exceptionally()`**, `handle()` runs even on success.  

---

# **7. Using Custom Thread Pool**
- By default, `CompletableFuture` uses **ForkJoinPool**.  
- We can specify **custom thread pools**.

```java
ExecutorService executor = Executors.newFixedThreadPool(2);

CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
    System.out.println("Task running on: " + Thread.currentThread().getName());
}, executor);

future.join();
executor.shutdown();
```
**Output:**  
```
Task running on: pool-1-thread-1
```
- **Prevents overloading ForkJoinPool** in high-load apps.  

---

# **Summary of CompletionStage Methods**

| Method | Description |
|--------|-------------|
| `supplyAsync()` | Runs an async task and returns a result. |
| `runAsync()` | Runs an async task **without returning a result**. |
| `thenApply()` | Transforms the result **(sync)**. |
| `thenAccept()` | Consumes the result **without returning anything**. |
| `thenRun()` | Runs a task after completion **without using the result**. |
| `thenCompose()` | Chains async calls **(flattens nested futures)**. |
| `thenCombine()` | Combines results of two independent futures. |
| `allOf()` | Waits for **all** tasks to complete. |
| `anyOf()` | Returns the **first** completed task’s result. |
| `exceptionally()` | Handles errors and provides fallback values. |
| `handle()` | Handles **both success & failure cases**. |

---

# **Conclusion**
- `CompletionStage` provides **powerful async programming** tools.  
- Supports **chaining, combining, and error handling**.  
- **Non-blocking execution** improves **performance** in concurrent applications.  
- Ideal for **REST APIs, Database Queries, Microservices, and Parallel Computing**.

Would you like **real-world use cases** or help with **debugging `CompletableFuture`?**
