`pidstat` is a Linux command (from the **sysstat** package) used to monitor **per-process** resource usage — CPU, memory, I/O, threads, context switches, etc. It is extremely useful when you want to understand **which process** is causing load on your system.

---

# ✅ **What `pidstat` Does**

It reports **statistics for every process** or for a specific PID at a given interval. Unlike `top`, which is interactive, `pidstat` gives **time-based reports** that you can log or analyze.

---

# 📌 **Basic Syntax**

```
pidstat [options] [interval] [count]
```

* **interval** → seconds between samples
* **count** → how many samples to take

---

# ✨ Common Options & Meaning

Below is the most useful set of options explained clearly:

### 🟦 **CPU Usage**

```
pidstat -u 1
```

Shows CPU usage of all processes every 1 second.

Columns:

| Column  | Meaning                 |
| ------- | ----------------------- |
| UID     | User ID                 |
| PID     | Process ID              |
| %usr    | CPU time in user mode   |
| %system | CPU time in kernel mode |
| %guest  | Guest time (VMs)        |
| %CPU    | Total CPU used          |

---

### 🟥 **Memory Usage**

```
pidstat -r 1
```

Shows per-process memory stats.

| Column   | Meaning                    |
| -------- | -------------------------- |
| minflt/s | Minor page faults/sec      |
| majflt/s | Major page faults/sec      |
| VSZ      | Virtual memory size        |
| RSS      | Resident memory (RAM used) |

---

### 🟩 **I/O Usage**

```
pidstat -d 1
```

| Column    | Meaning               |
| --------- | --------------------- |
| kB_rd/s   | KB read per sec       |
| kB_wr/s   | KB written per sec    |
| kB_ccwr/s | Write-back from cache |

Helps detect high disk I/O processes.

---

### 🟨 **Per-Thread Stats**

```
pidstat -t 1
```

Shows all threads for each PID.

---

### 🟧 **Context Switches**

```
pidstat -w 1
```

| Column    | Meaning                      |
| --------- | ---------------------------- |
| cswch/s   | Voluntary context switches   |
| nvcswch/s | Involuntary context switches |

Useful for diagnosing thread contention or lock issues.

---

### 🟪 **Specific Process**

```
pidstat -p <pid> 1
```

Example:

```
pidstat -p 1234 -u -r -d 1
```

Shows CPU + memory + I/O for that PID every second.

---

### 🔵 **Track Processes That Start or Stop**

```
pidstat -H 1
```

Shows historical process activity including exited processes.

---

# 🚀 **Useful Real-World Examples**

### 1️⃣ Find which process is causing high CPU

```
pidstat -u 1
```

### 2️⃣ Monitor a Java application’s GC or thread behavior

```
pidstat -p <java_pid> -t 1
```

### 3️⃣ Identify processes doing high I/O

```
pidstat -d 1
```

### 4️⃣ Detect thread lock issues (high involuntary context switches)

```
pidstat -w 1
```

---

# 🧠 **Why pidstat Is Better Than top**

* Shows **historical sampling**, not just real-time
* Can monitor **one PID**
* Can show **threads**, not just processes
* Helps identify **I/O bottlenecks**
* Easy to log to a file:

  ```
  pidstat -u 1 > cpu.log
  ```

---

If you want, I can also explain:

✅ how to interpret each column in detail
✅ how to use `pidstat` for performance debugging (CPU, memory leaks, I/O)
✅ comparison with sar, top, ps, vmstat

Just tell me!

