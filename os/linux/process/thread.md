Your answer captures only a very small part and needs important corrections.
Here is the correct, concise, accurate explanation:


---

✅ Correct Answer

Process

A process is a heavyweight execution unit.

It has its own memory space (virtual address space), file descriptors, resources, and execution context.

Created using fork() (or exec()) and is fully isolated from other processes.


Thread

A thread is a lightweight execution unit inside a process.

Threads share the same memory, file handles, heap, and global variables of the parent process.

Only the stack, registers, and thread-local storage are unique per thread.

Created using pthread_create() or Java thread APIs etc.



---

✅ How Linux Treats Them

Linux uses the same underlying structure (task_struct) for both threads and processes.

But:

Processes have separate memory maps.

Threads share the same memory map, so context switching is faster.


The scheduler schedules threads, not processes—the OS treats each thread as an independent schedulable unit.



---

⚠️ Corrections to your answer

❌ “Process is lightweight thread” → Incorrect
✔️ A thread is lightweight; a process is heavier.

❌ “Thread copies same data of process” → Incorrect
✔️ A thread does NOT copy process memory; it shares it.

✔️ “Thread is child of process” → Correct.



---

If you want the next DevOps/Linux question, say “Next question.”