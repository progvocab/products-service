Here is a **clear, concise, technically accurate** explanation of the **`mpstat` command in Linux**, focusing on CPU-level internals, per-core monitoring, interrupts, scheduling, and real performance interpretation.

---

# 🐧 **`mpstat` Command in Linux — Complete Explanation**

`mpstat` (Multiprocessor Statistics) reports **per-CPU and global CPU usage**, showing how each core is being utilized.
It is part of the **sysstat** toolset.

---

# ### **Why `mpstat` Is Important**

Most tools (top, vmstat) show **aggregate CPU usage**, which can hide real problems.

`mpstat` tells you:

* Is a single CPU core overloaded?
* Is CPU time spent in user/kernel?
* Is the scheduler overloaded?
* Are there interrupt storms?
* Are processes blocked on I/O?

---

# ### **How to Run**

```bash
mpstat 1
```

Shows CPU stats every second.

To show **per-core** stats:

```bash
mpstat -P ALL 1
```

---

# ### **Sample Output**

```
Linux 5.10.0

11:30:02 AM  CPU   %usr  %sys  %nice %idle %iowait %irq %soft %steal %guest %gnice
11:30:03 AM  all     10     5      0    80       2    0     1     0       0      0
11:30:03 AM    0     20    10      0    60      10    0     0     0       0      0
11:30:03 AM    1      5     2      0    92       1    0     0     0       0      0
```

---

# ### **Column-by-Column Explanation**

| Column      | Meaning                                      | Kernel Component                         |
| ----------- | -------------------------------------------- | ---------------------------------------- |
| **%usr**    | Time in user mode                            | Applications                             |
| **%sys**    | Time in kernel mode                          | Linux kernel (syscalls, I/O, networking) |
| **%nice**   | Time spent on niced (low-priority) processes | Scheduler                                |
| **%idle**   | CPU idle time                                | CPU idle thread                          |
| **%iowait** | CPU idle because waiting for I/O             | Block I/O scheduler                      |
| **%irq**    | Time spent handling hardware interrupts      | Interrupt handler                        |
| **%soft**   | Time on software interrupts                  | softirq (network, timer)                 |
| **%steal**  | CPU stolen by hypervisor                     | KVM/VMWare/Xen                           |
| **%guest**  | Time running VM guest                        | Virtualization                           |
| **%gnice**  | Time running niced guest tasks               | Virtualization                           |

---

# ### **What Each Field Indicates (Real Performance Insights)**

## **1. %usr high**

Your application code is CPU-bound.
Example: sorting, encryption, compression.

---

## **2. %sys high**

Kernel is busy.
Causes:

* heavy networking
* many syscalls
* filesystem metadata operations
* context switching overhead

---

## **3. %iowait high**

CPU is idle because disk/network is slow.
Common in:

* databases
* Kafka
* large file reads/writes
* NFS/EFS latency

---

## **4. %irq or %soft high**

Interrupt storm.

Typical reasons:

* NIC handling heavy packets
* high-frequency timers
* storage interrupts

---

## **5. %steal high**

You are on cloud VM (AWS EC2, GCP, Azure) and the hypervisor is stealing CPU.

Meaning:
**Your VM is starved → upgrade instance type.**

---

# ### **Why `mpstat -P ALL` Matters**

`top` might say:

```
CPU idle = 90%
```

But `mpstat -P ALL` might show:

```
CPU0 = 10% idle  
CPU1 = 99% idle  
CPU2 = 99% idle  
CPU3 = 99% idle
```

This means the load is stuck on **one core**, causing:

* thread bottleneck
* lock contention
* poor performance
* latency spikes

This is extremely common in:

* Java applications
* Node.js apps
* Python GIL
* Databases
* NGINX/Apache

---

# ### **Use Cases of mpstat**

## **1. Debug CPU bottlenecks**

See if 1 core is overloaded while others are idle.

---

## **2. Detect interrupt storms**

High irq/soft means NIC or disk interrupts are spiking.

---

## **3. Detect I/O bottlenecks**

High iowait → your disk / EBS / NFS is slow.

---

## **4. Detect virtualization issues**

High steal → noisy neighbor problem on cloud.

---

## **5. Validate thread distribution**

Check if Java, Kafka, DB threads are evenly spread across CPUs.

---

# ### **Interpretation Example**

Output:

```
CPU0: 90% usr  
CPU1: 5% usr  
CPU2: 4% usr  
CPU3: 2% usr  
```

Meaning:

* one thread is doing all work
* fix: use multi-threading or tune thread pool

---

# 🧩 Summary Table

| Key Issue        | Indicator     | Meaning                  |
| ---------------- | ------------- | ------------------------ |
| CPU bound        | high usr      | app too busy             |
| Kernel bound     | high sys      | syscalls/network heavy   |
| I/O bound        | high iowait   | disk/network slow        |
| Interrupt storm  | high irq/soft | NIC/disk interrupts      |
| VM contention    | high steal    | CPU stolen by hypervisor |
| Thread imbalance | one CPU hot   | poor parallelism         |

---

If you want, I can also explain:

✔ `mpstat` vs `vmstat` vs `iostat`
✔ How Java/Kafka workloads appear in mpstat
✔ How Linux CPU scheduling works internally
✔ How interrupts and softirqs work

Just ask!
