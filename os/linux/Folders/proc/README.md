The **/proc** folder in Linux is a *virtual filesystem* that exposes **live system information** directly from the kernel.
It is not stored on disk — it’s dynamically generated on demand.

In simple words:

> **/proc is the window into the running Linux kernel.**
> It shows real-time info about CPU, memory, processes, hardware, network, and kernel settings.

---

# 🗂️ **What is /proc?**

* A **pseudo-filesystem** mounted at `/proc`
* Provided by the Linux kernel
* Contains files that look like text files but actually represent **kernel data structures**
* Zero disk usage (stored in memory)
* Updated instantly as the system runs
* Used by tools like `top`, `ps`, `free`, `iostat`, `vmstat`, `uptime`

---

# 📌 **Main Categories Inside /proc**

There are **two major types of entries**:

1. **System-wide information** (e.g., `/proc/meminfo`, `/proc/cpuinfo`)
2. **Per-process information** (directories named by PID: `/proc/1234/`)

Let's go through both.

---

# 🧠 **1. System-Wide Files in /proc (Most Important Ones)**

Here are the key files and what they represent:

### 📄 **/proc/cpuinfo**

Information about each CPU core:

* model
* MHz
* cache size
* flags (virtualization, SSE, AVX)

### 📄 **/proc/meminfo**

Complete memory statistics:

* MemTotal
* MemAvailable
* Buffers
* Cached
* SwapTotal
* SwapFree
* HugePages info

Used by: `free`, `vmstat`

---

### 📄 **/proc/loadavg**

Load average + running processes:

* 1/5/15 min load
* number of running processes
* last PID created

Used by: `uptime`, `top`

---

### 📄 **/proc/uptime**

* system uptime
* amount of time CPU spent idle

---

### 📄 **/proc/stat**

One of the most important files: **CPU and kernel statistics**

Contains:

* CPU usage (user/system/idle/iowait/irq/softirq/steal)
* context switches (ctxt)
* number of processes created
* running/blocked processes

Used by: `top`, `mpstat`

---

### 📄 **/proc/diskstats**

Per-disk counters:

* reads/writes count
* read/write bytes
* read/write time
* queue size

Used by: `iostat`

---

### 📄 **/proc/net/**

Network statistics:

* `/proc/net/dev` – per-interface counters
* `/proc/net/tcp` – active TCP connections
* `/proc/net/snmp` – TCP/UDP counters

Used by: `netstat`, `ss`, `ip`

---

### 📄 **/proc/interrupts**

Interrupts per CPU:

* device IRQ counts
* CPU distribution

Used for detecting hardware/NUMA imbalances.

---

### 📄 **/proc/swaps**

Active swap devices.

---

### 📄 **/proc/version**

Kernel version + GCC version used to compile.

---

### 📄 **/proc/cmdline**

Kernel boot parameters passed by GRUB.

---

### 📄 **/proc/modules**

Loaded kernel modules (`lsmod` reads this).

---

### 📄 **/proc/filesystems**

List of supported file system types.

---

### 📄 **/proc/sys/**

Kernel tunables exposed via sysctl:

* `kernel/`
* `vm/`
* `net/`
* `fs/`

Example:

`/proc/sys/vm/swappiness` → memory swapping aggressiveness
You can modify values at runtime.

---

# 🧠 **2. Per-Process Directories in /proc**

Every process gets its own folder:

```
/proc/<pid>/
```

Example:

```
/proc/1234/
```

This directory contains the live info of process **1234**.

### Important files inside a process directory:

| File        | Meaning                              |
| ----------- | ------------------------------------ |
| **cmdline** | Full command used to start process   |
| **cwd**     | Symlink to current working directory |
| **environ** | Environment variables                |
| **exe**     | Symlink to executable file           |
| **fd/**     | Open file descriptors                |
| **maps**    | Memory mappings                      |
| **mem**     | Process memory                       |
| **stat**    | CPU, memory, state of process        |
| **status**  | Human-readable process status        |
| **tasks/**  | Threads in this process              |
| **io**      | Read/write I/O bytes                 |

### Example

`/proc/1234/stat` contains:

* user CPU time
* kernel CPU time
* RSS (resident memory)
* minor & major page faults
* process state
* priority
* nice value

`ps` and `pidstat` read this file.

---

# 🧠 **3. Why /proc is Important?**

| Tool      | Reads                                    |
| --------- | ---------------------------------------- |
| `top`     | /proc/stat, /proc/meminfo, /proc/loadavg |
| `ps`      | /proc/*/stat, /proc/*/status             |
| `iostat`  | /proc/diskstats                          |
| `vmstat`  | /proc/meminfo, /proc/stat                |
| `free`    | /proc/meminfo                            |
| `sar`     | many /proc files                         |
| `netstat` | /proc/net/*                              |

---

# 🧩 **4. Why is /proc called a virtual filesystem?**

* It is not stored on storage disks
* No physical space used
* Kernel generates content on access
* Always up-to-date
* Exists only when Linux is running

If you reboot, everything under `/proc` disappears because it’s not persistent.

---

# ✔️ TL;DR (Simple Explanation)

**/proc is a virtual folder that shows live data about Linux kernel, hardware, system resources, and every running process.**

If you want, I can also provide:
✔ Diagram of /proc hierarchy
✔ Real output examples and explanations
✔ How tools like `top`, `free`, `iostat` use /proc internally
✔ Most important /proc files for troubleshooting performance
