 **Fork/Join** works under the hood when you use a **parallel stream** in Java:





```java
import java.util.List;
import java.util.stream.IntStream;

public class ParallelStreamExample {
    public static void main(String[] args) {
        List<Integer> numbers = IntStream.rangeClosed(1, 10).boxed().toList();

        int sum = numbers.parallelStream()      // uses ForkJoinPool.commonPool()
                         .mapToInt(n -> {
                             System.out.println(Thread.currentThread().getName() + " processing " + n);
                             return n;
                         })
                         .sum();

        System.out.println("Sum = " + sum);
    }
}
```



1. `parallelStream()` divides the data source (here, 1–10) into **subtasks**.
2. These subtasks are submitted to the **ForkJoinPool.commonPool()**, which uses multiple worker threads (usually equal to CPU cores).
> worker thread count is configured by -Djava.util.concurrent.ForkJoinPool.common.parallelism=8

by default the value is `Runtime.availableProcessors() - 1`

one less than the number of CPU cores
3. Each worker thread **processes part of the stream** concurrently.
4. The **Fork/Join Framework** automatically **splits (fork)** the work and **combines (join)** results efficiently.
5. Finally, `sum()` (a terminal operation) **aggregates** the partial results into a single output.



> `parallelStream()` is a high-level abstraction built on top of the **Fork/Join Framework** that automatically handles task splitting, scheduling, and joining behind the scenes.



##  ForkJoinTask 
rewrite the code using ForkJoinTask or RecursiveTask API manually



- the **manual Fork/Join version** 





```java
import java.util.concurrent.*;

class SumTask extends RecursiveTask<Integer> {
    int[] arr; int start, end;
    SumTask(int[] arr, int start, int end) { this.arr = arr; this.start = start; this.end = end; }

    @Override
    protected Integer compute() {
        if (end - start <= 2)  // small task
            return IntStream.range(start, end).map(i -> arr[i]).sum();

        int mid = (start + end) / 2;
        SumTask left = new SumTask(arr, start, mid);
        SumTask right = new SumTask(arr, mid, end);
        left.fork();                       // run left in parallel
        return right.compute() + left.join(); // combine results
    }
}

public class ForkJoinExample {
    public static void main(String[] args) {
        int[] arr = {1,2,3,4,5,6,7,8,9,10};
        ForkJoinPool pool = new ForkJoinPool();
        int sum = pool.invoke(new SumTask(arr, 0, arr.length));
        System.out.println("Sum = " + sum);
    }
}
```



* **Fork:** splits task into smaller parts (`left`, `right`).
* **Join:** waits for subtask results and combines them.
* **ForkJoinPool:** runs tasks on multiple threads efficiently.

> Essentially, this manual version is what `parallelStream()` does internally — split work, execute in parallel, and merge results.




### **Fork/Join Pool in Java**

* Introduced in **Java 7** (`java.util.concurrent` package).
* It’s a **special thread pool** designed to efficiently execute tasks that can be **split (forked)** into smaller subtasks and then **combined (joined)** to get the final result.
* Implements the **Fork/Join framework**, which is ideal for **parallelism** (e.g., divide-and-conquer algorithms).



### **Core Idea**

1. A **big task** is recursively split into **smaller subtasks** (using `fork()`).
2. Subtasks run in parallel on different worker threads.
3. Results of subtasks are **combined (joined)** to produce the final output.

This follows the **Divide and Conquer** strategy.



### **Key Classes**

1. **`ForkJoinPool`**

   * A specialized `ExecutorService` that manages worker threads.
   * Uses a **work-stealing algorithm** → idle threads “steal” tasks from busy threads to maximize CPU usage.

2. **`RecursiveTask<V>`**

   * Used when the task **returns a value**.

3. **`RecursiveAction`**

   * Used when the task **doesn’t return a value**.

---

## 🔹 **Simple Exam

### Sum of an array using Fork/Join

```java
import java.util.concurrent.*;

class SumTask extends RecursiveTask<Long> {
    private final int[] arr;
    private final int start, end;
    private static final int THRESHOLD = 5;

    public SumTask(int[] arr, int start, int end) {
        this.arr = arr;
        this.start = start;
        this.end = end;
    }

    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // Base case: compute directly
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += arr[i];
            }
            return sum;
        } else {
            // Split into subtasks
            int mid = (start + end) / 2;
            SumTask left = new SumTask(arr, start, mid);
            SumTask right = new SumTask(arr, mid, end);

            left.fork(); // asynchronously execute left
            long rightResult = right.compute(); // compute right directly
            long leftResult = left.join(); // wait for left to finish

            return leftResult + rightResult;
        }
    }
}

public class ForkJoinExample {
    public static void main(String[] args) {
        int[] arr = {1,2,3,4,5,6,7,8,9,10};

        ForkJoinPool pool = new ForkJoinPool();
        SumTask task = new SumTask(arr, 0, arr.length);

        long result = pool.invoke(task);

        System.out.println("Sum = " + result);
    }
}
```



### **Why ForkJoinPool is Efficient**

* **Work-stealing algorithm:**
  Each worker thread has its own deque (double-ended queue).

  * It pushes/pops its own tasks from one end.
  * If idle, it steals tasks from the **opposite end** of another thread’s deque.
* This reduces contention and maximizes CPU usage.



## 🔹 **Comparison with ThreadPoolExecutor**

| Feature             | ForkJoinPool                            | ThreadPoolExecutor                        |
| ------------------- | --------------------------------------- | ----------------------------------------- |
| **Best for**        | Divide & Conquer, recursive parallelism | Independent tasks (web requests, jobs)    |
| **Splitting tasks** | Yes (`fork()` / `join()`)               | No                                        |
| **Work-stealing**   | Yes                                     | No                                        |
| **Performance**     | Better for parallel computation         | Better for general-purpose task execution |

 
`ForkJoinPool` = A high-performance thread pool specialized for **parallel divide-and-conquer tasks** with a **work-stealing scheduler**.



More :  Do you want me to also explain how **`parallelStream()` in Java 8** internally uses a **ForkJoinPool**?
