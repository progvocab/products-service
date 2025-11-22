Linux exposes **many OS counters** (metrics) that help you understand CPU, memory, I/O, network, processes, and system health. These counters come from `/proc`, system calls, kernel subsystems, and tools like `vmstat`, `mpstat`, `iostat`, `pidstat`, `netstat`, `sar`, and `top`.

Below is a structured list of **all major OS counters in Linux**, grouped by subsystem.

---

# 🧠 **1. CPU Counters**

These come from `/proc/stat` and are used by `top`, `mpstat`, `sar` etc.

### **CPU time breakdown**

* **user** – time spent running user-space processes
* **nice** – time spent on low-priority user processes
* **system** – kernel time
* **idle** – idle CPU time
* **iowait** – waiting for disk I/O
* **irq** – servicing hardware interrupts
* **softirq** – servicing software interrupts
* **steal** – vCPU stolen by hypervisor
* **guest / guest_nice** – time spent running virtual CPUs

### **Load average**

(from `/proc/loadavg`)

* **1 min load**
* **5 min load**
* **15 min load**
* **Running/total processes count**
* **Last PID created**

### **Context switches & process creation**

(from `/proc/stat`)

* **ctxt** – total context switches
* **processes** – number of processes created
* **procs_running**
* **procs_blocked**

---

# 🧠 **2. Memory Counters**

From `/proc/meminfo`, `vmstat`, `free`.

### **Physical Memory**

* **MemTotal**
* **MemFree**
* **Buffers**
* **Cached**
* **SwapCached**
* **Active**
* **Inactive**
* **Slab** (kernel structures)
* **PageTables**
* **CommitLimit / Committed_AS**

### **Swap**

* **SwapTotal**
* **SwapFree**
* **SwapUsed**

### **Paging**

* **pgpgin / pgpgout** – pages paged in/out
* **pgfault** – minor page faults
* **pgmajfault** – major page faults

---

# 🧠 **3. Disk I/O Counters**

From `/proc/diskstats`, `iostat`, `sar -d`.

### Per-disk counters

* **Reads completed**

* **Reads merged**

* **Read sectors**

* **Read time (ms)**

* **Writes completed**

* **Writes merged**

* **Write sectors**

* **Write time (ms)**

* **I/O in progress**

* **Time spent doing I/Os**

* **Weighted time doing I/Os**

### High-level I/O metrics

* **r/s** – reads per sec
* **w/s** – writes per sec
* **rKB/s & wKB/s**
* **avgqu-sz** – queue size
* **await** – avg I/O latency
* **svctm** – service time

---

# 🧠 **4. Filesystem Counters**

From `df`, `du`, `/proc/mounts`, `statfs`.

* **Filesystem capacity** (used/free)
* **Inode usage**
* **Read/write error counters**
* **Filesystem type (ext4/xfs/nfs)**
* **Mount options**
* **Free blocks**
* **Block size**

---

# 🧠 **5. Network Counters**

From `/proc/net/dev`, `/proc/net/snmp`, `ss`, `netstat`, `sar -n`.

### Per-interface counters

* **rx_packets / tx_packets**
* **rx_bytes / tx_bytes**
* **rx_errors / tx_errors**
* **rx_dropped / tx_dropped**
* **fifo errors**
* **collisions**
* **carrier errors**

### TCP counters

(from `/proc/net/snmp`)

* **ActiveOpens / PassiveOpens**
* **AttemptFails**
* **EstabResets**
* **InSegs / OutSegs**
* **RetransSegs**
* **InErrs**

### UDP counters

* **Datagrams In/Out**
* **Datagrams Dropped**
* **InErrors**

### Socket states

(from `ss -s`)

* **ESTABLISHED**
* **TIME_WAIT**
* **CLOSE_WAIT**, etc.

---

# 🧠 **6. Process Counters**

From `/proc/<pid>/`, `ps`, `pidstat`.

### CPU-related

* **utime / stime**
* **%CPU**
* **Threads count**
* **Context switches (voluntary / involuntary)**

### Memory-related

* **RSS** – resident memory
* **VMS** – virtual memory
* **shared memory**
* **swap usage**
* **page faults**

### I/O-related

* **read bytes / write bytes**
* **syscr / syscw** (read/write syscalls)
* **iowait time**
* **IO throttling time**

### File descriptors

* **open fds count**
* **limits (ulimit)**

---

# 🧠 **7. Kernel Counters**

From `/proc`, sysctl.

* **Interrupts** (`/proc/interrupts`)
* **Softirqs** (`/proc/softirqs`)
* **Slabinfo** (kernel caches)
* **NUMA memory stats**
* **Hugepages statistics**
* **Scheduler statistics**

---

# 🧠 **8. Systemwide Performance Counters**

Tools like `vmstat`, `sar`, `dstat`.

### vmstat counters

* **r** – run queue
* **b** – blocked
* **swpd** – swap used
* **free / buff / cache**
* **si / so** – swap in/out
* **bi / bo** – I/O blocks
* **in** – interrupts
* **cs** – context switches
* **us / sy / id / wa / st** – CPU %

---

# 🧠 **9. Hardware Counters (Perf Events)**

From `perf stat`:

* CPU cycles
* Instructions
* Cache references
* Cache misses
* Branches
* Branch mispredictions
* Task-clock
* Context-switches
* Page-faults

---

# ✔️ Summary Table

Below table groups **major Linux OS counters**:

| Category      | Key Counters                                                         |
| ------------- | -------------------------------------------------------------------- |
| CPU           | user, system, idle, iowait, irq, softirq, load avg, context switches |
| Memory        | used/free, cached, swap, page faults, slab, active/inactive          |
| Disk I/O      | reads/writes, await, IOPS, throughput, queue size                    |
| Network       | rx/tx bytes, errors, dropped, TCP stats, UDP stats                   |
| Processes     | CPU %, RSS, VMS, faults, I/O, threads                                |
| Kernel        | interrupts, softirqs, slab, hugepages                                |
| Filesystem    | disk usage, inode usage, fs type, free blocks                        |
| Perf counters | cycles, instructions, cache misses, branches                         |

---

If you want, I can also:
✅ explain each `/proc` file in detail
✅ provide diagrams
✅ give commands to extract ALL counters
✅ explain which counters matter for performance tuning (CPU, DB, JVM, network, etc.)
