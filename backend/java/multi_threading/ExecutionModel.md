### Execution models in Java threads

These models refer to how threads are used, not just how they are created.


---

1. Thread-per-Request (Blocking Model)

Each task / request gets its own dedicated thread.

Common in traditional servlet containers (Tomcat blocking I/O).

Threads perform blocking I/O, sleep, wait, lock, etc.

Simple but limited by thread count → poor scalability with many connections.



---

2. Thread Pool / Worker Pool Model

A fixed or dynamic pool of worker threads (ExecutorService, ForkJoinPool).

Tasks are queued and executed by available workers.

Supports blocking operations but reduces overhead of thread creation.

Widely used in enterprise apps, JDBC calls, synchronous APIs.



---

3. Event-Loop (Non-Blocking) Model

A small number of event-loop threads handle many connections using async I/O.

Threads must never block.

Used in: Netty, Java NIO Selector, Reactor (reactor-http-nio), Vert.x.

Ideal for high-concurrency, low-latency apps.



---

4. Reactive / Non-Blocking Scheduler Model (Project Reactor, RxJava)

Built on event loops but tasks are executed by specialized schedulers:

parallel-*

boundedElastic-*

single-*

immediate


parallel-* uses non-blocking threads

boundedElastic-* provides blocking-friendly elastic worker threads

Execution is orchestrated via async pipelines, not direct thread usage.



---

5. Virtual Threads (Project Loom) — Thread-per-Task (Non-blocking-internally)

Lightweight threads managed by JVM, not OS.

Allow millions of threads.

Blocking operations (I/O, sleep, locks) are virtual-blocking, not OS-blocking.

Simplifies concurrency by combining “thread-per-task” model with non-blocking scalability.



---

6. Fork/Join Work-Stealing Model

Used mainly for parallel computation (Java Streams parallel, ForkJoinPool).

Tasks split into subtasks; worker threads steal work from each other.

Good for CPU-bound work, not I/O.



---

Summary Table

Execution Model	Threads Block?	When Used	Example Frameworks

Thread-per-request	Yes	Simple, blocking I/O apps	Classic server apps
Thread pool	Yes	Most enterprise workloads	ExecutorService
Event-loop	No	High concurrency, async I/O	Netty, NIO, Vert.x
Reactive scheduler	Mixed	Reactive pipelines	Reactor, RxJava
Virtual threads	Blocking (virtually)	Modern scalable apps	Loom (Java 21+)
Work-stealing (ForkJoin)	No (for compute)	CPU-bound parallel tasks	Streams parallel


Here is a concise, engineering-focused guide on choosing the right execution model for real systems:


Choose the execution model based on whether your workload is CPU-bound, I/O-bound, or concurrency-bound. Use the thread-per-request or thread-pool model when your code does blocking operations like JDBC calls, file I/O, or legacy APIs. For high-concurrency systems with many simultaneous connections and async I/O, use an event-loop or reactive execution model because non-blocking threads scale far better under heavy load. For CPU-intensive tasks, prefer the ForkJoin / work-stealing model to maximize parallelism. Finally, if you need simple synchronous code that scales like async, use virtual threads (Loom), which allow millions of concurrent tasks without the complexity of reactive programming. In practice, the correct model aligns with the nature of the workload and the cost of blocking at scale.

```mermaid 
flowchart TD

    A[Execution Models in Java] --> B[Thread-per-Request<br>(Blocking)]
    A --> C[Thread Pool / Worker Pool<br>(Blocking)]
    A --> D[Event-Loop Model<br>(Non-Blocking)]
    A --> E[Reactive Scheduler Model<br>(Non-Blocking / Hybrid)]
    A --> F[Virtual Threads (Loom)<br>(Lightweight Blocking)]
    A --> G[Fork/Join Work-Stealing<br>(CPU-Bound)]

    B --> B1[Each request gets its own thread]
    B --> B2[Good for simple blocking I/O]
    B --> B3[Low scalability]

    C --> C1[Executors, ThreadPoolExecutor]
    C --> C2[Handles blocking tasks efficiently]

    D --> D1[Netty, NIO Selector]
    D --> D2[Threads never block]
    D --> D3[High concurrency, low latency]

    E --> E1[Reactor, RxJava]
    E --> E2[Parallel Scheduler / Elastic Scheduler]
    E --> E3[Event-loop foundation + async pipelines]

    F --> F1[Millions of threads]
    F --> F2[Simple synchronous code that scales]
    F --> F3[JVM manages scheduling]

    G --> G1[ForkJoinPool]
    G --> G2[Work-stealing for parallel compute]
    G --> G3[Best for CPU-heavy tasks]

```


A diagram showing the models

How to choose the right execution model for real systems