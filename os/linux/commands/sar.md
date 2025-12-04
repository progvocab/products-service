Below is a **clear, concise, technically accurate** explanation of the **`sar` command in Linux**, focusing on what it measures, how it works internally, and how to interpret results.

---

# ### **What is `sar`?**

`sar` (**System Activity Reporter**) is part of **sysstat** and is the **most complete performance monitoring tool in Linux**.
It collects and reports **historical** and **real-time** system metrics, including:

* CPU
* Memory
* I/O
* Network
* Load
* Process creation
* Context switches
* Disk utilization
* Page faults
* Interrupts
* Per-device network stats

It is the only built-in utility that gives **historical performance data**, not just current snapshots.

---

# ### **Where `sar` Gets Its Data**

`sar` uses:

* `/proc` filesystem (kernel metrics)
* sysstat background collector: **`/usr/lib/sysstat/sadc`**
* data stored in `/var/log/sa/saXX` (daily files)

---

# ### **How to Enable Data Collection**

```bash
sudo systemctl enable sysstat
sudo systemctl start sysstat
```

---

# ### **Basic Usage**

## **Show CPU usage**

```bash
sar -u 1 5
```

Every 1 second, 5 samples.

## **Show memory usage**

```bash
sar -r 1 5
```

## **Show load average**

```bash
sar -q
```

## **Show I/O usage**

```bash
sar -b
```

## **Show network stats**

```bash
sar -n DEV 1
```

## **Show disk usage (per block device)**

```bash
sar -d 1
```

---

# ### **Popular `sar` Metrics and Their Meaning**

## ### **1. CPU (`sar -u`)**

| Field      | Meaning                          |
| ---------- | -------------------------------- |
| %usr       | User-mode CPU time               |
| %sys       | Kernel-mode CPU time             |
| %iowait    | CPU waiting for disk I/O         |
| %idle      | Idle CPU time                    |
| %irq/%soft | Hardware/software interrupt time |

---

## ### **2. Memory (`sar -r`)**

| Field             | Meaning                       |
| ----------------- | ----------------------------- |
| kbmemfree         | Free RAM                      |
| kbmemused         | Used RAM                      |
| %commit           | Memory requested vs available |
| kbactive/inactive | Page cache workings           |

---

## ### **3. Paging (`sar -B`)**

| Field            | Meaning                           |
| ---------------- | --------------------------------- |
| pgpgin / pgpgout | Pages in/out from disk            |
| pgfault          | Minor page faults                 |
| pgmajfault       | Major faults → disk read required |

High **pgmajfault** = memory pressure.

---

## ### **4. I/O (`sar -b`)**

| Field   | Meaning                     |
| ------- | --------------------------- |
| tps     | I/O transactions per second |
| rtps    | Read ops/sec                |
| wtps    | Write ops/sec               |
| bread/s | Bytes read per second       |
| bwrtn/s | Bytes written per second    |

---

## ### **5. Disk Devices (`sar -d`)**

| Field    | Meaning                 |
| -------- | ----------------------- |
| tps      | I/O ops/sec             |
| rd_sec/s | Sectors read/sec        |
| wr_sec/s | Sectors written/sec     |
| avgrq-sz | Avg request size        |
| avgqu-sz | Queue length            |
| await    | Total time per I/O (ms) |

**High `await` = slow disk or overloaded EBS/NFS.**

---

## ### **6. Network (`sar -n DEV`)**

| Field   | Meaning              |
| ------- | -------------------- |
| rxpck/s | Packets received/sec |
| txpck/s | Packets sent/sec     |
| rxkB/s  | Incoming bandwidth   |
| txkB/s  | Outgoing bandwidth   |

---

## ### **7. Process Scheduler (`sar -w`, `sar -q`)**

### `sar -w` — context switches

| Field     | Meaning                    |
| --------- | -------------------------- |
| cswch/s   | Voluntary context switches |
| nvcswch/s | Involuntary switches       |

### `sar -q` — run queue

| Field    | Meaning                  |
| -------- | ------------------------ |
| runq     | Number of runnable tasks |
| plist_sz | Processes in task list   |
| load     | Load average             |

---

# ### **Real Performance Use Cases**

### **1. Debug CPU bottlenecks**

```
sar -u
```

Shows if bottleneck is user mode, kernel, I/O wait, or interrupts.

---

### **2. Investigate memory pressure**

```
sar -r
sar -B
```

High major faults → RAM is insufficient.

---

### **3. Diagnose I/O bottlenecks**

```
sar -b
sar -d
```

High `await` or queue length means slow disk or overloaded NFS/EBS.

---

### **4. Debug network performance**

```
sar -n DEV
sar -n TCP
sar -n SOCK
```

Helps diagnose:

* packet drops
* TCP retransmissions
* socket exhaustion

---

### **5. Historical Troubleshooting**

If a server slowed down yesterday at 3 PM:

```bash
sar -u -f /var/log/sa/sa11
```

This gives **exact CPU usage at 3 PM yesterday**.

No other built-in tool does this.

---

# ### **Why `sar` Is More Powerful Than top/vmstat/iostat**

| Tool    | Real-time | Historical | Per-device/PID detail | Best For                    |
| ------- | --------- | ---------- | --------------------- | --------------------------- |
| top     | ✔         | ❌          | PID                   | single-process view         |
| vmstat  | ✔         | ❌          | ❌                     | CPU/mem snapshot            |
| iostat  | ✔         | ❌          | block devices         | I/O troubleshooting         |
| mpstat  | ✔         | ❌          | per-CPU               | multi-core analysis         |
| **sar** | ✔         | ✔          | CPU/mem/disk/net      | **complete system history** |

---

# ### **Simple Summary**

`sar` helps you answer:

* Was my CPU 100% at midnight?
* Was the disk overloaded 3 hours ago?
* Did memory spike last night?
* Was the network saturated?

**It is the single most powerful performance-analysis tool included in Linux.**

---

If you want, I can also generate:

✔ Mermaid diagrams of the sar architecture
✔ Cheat sheet of all sar commands
✔ Interpretation examples (what is high/low)
✔ How Java/Kafka/DB workloads appear in sar output

Just tell me!
