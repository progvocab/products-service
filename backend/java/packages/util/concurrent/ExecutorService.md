 **sequence diagram** 

* Submit two `Callable` tasks to an `ExecutorService`,
* Then call `shutdown()`, and
* Later call `future1.get()` and `future2.get()`.

---

```mermaid
sequenceDiagram
    participant MainThread
    participant ExecutorService
    participant WorkerThread1
    participant WorkerThread2
    participant Task1
    participant Task2
    participant Future1
    participant Future2

    MainThread->>ExecutorService: submit(Task1)
    ExecutorService->>WorkerThread1: assign(Task1)
    WorkerThread1->>Task1: call()
    Task1-->>Future1: result / exception stored

    MainThread->>ExecutorService: submit(Task2)
    ExecutorService->>Queue: enqueue(Task2)
    
    MainThread->>ExecutorService: shutdown()
    Note right of ExecutorService: stops accepting new tasks<br/>but runs queued ones

    WorkerThread1-->>ExecutorService: task1 complete
    ExecutorService->>WorkerThread2: assign(Task2)
    WorkerThread2->>Task2: call()
    Task2-->>Future2: result / exception stored

    MainThread->>Future1: get()
    Future1-->>MainThread: returns Task1 result

    MainThread->>Future2: get()
    Future2-->>MainThread: returns Task2 result
```

---

### 🧩 Explanation:

* `submit()` returns a `Future` immediately.
* The executor assigns tasks to available worker threads.
* `shutdown()` only prevents **new** submissions; existing and queued tasks still execute.
* `Future.get()` blocks until the corresponding task is done.

Would you like me to add what happens if you call `shutdownNow()` instead (with interrupts shown in the same diagram)?
