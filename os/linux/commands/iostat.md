Here is a clear explanation of **the `iostat` command**, what it shows, and how to read it—especially for debugging slow servers, disk issues, and high I/O wait.

---

# ✅ **What is `iostat`?**

`iostat` (Input/Output Statistics) is a Linux command used to monitor:

* **Disk I/O performance**
* **CPU I/O wait**
* **Throughput (MB/s)**
* **Device utilization (%)**
* **Latency**

It helps you find bottlenecks caused by slow disks, busy disks, AWS EBS performance issues, or filesystem overload.

`iostat` is part of the **sysstat** package.

---

# ✅ **Typical Usage**

```
iostat -xz 1
```

Explanation:

* `-x` → extended stats (most useful fields)
* `-z` → remove devices with no activity
* `1` → refresh every 1 second

---

# ✅ **Key Output Fields (Most Important)**

You will typically see output like:

```
Device            r/s    w/s   rkB/s  wkB/s  await  svctm  %util
nvme0n1           10     20    5120   2048    15     1      50
```

Let’s break down the essential fields.

---

# 🔹 **1. r/s & w/s — Read and Write Requests per Second**

* `r/s` → read operations per second
* `w/s` → write operations per second

High numbers mean the disk is busy.

---

# 🔹 **2. rkB/s & wkB/s — Read/Write Throughput (KB/s)**

* `rkB/s` = data read per second
* `wkB/s` = data written per second

Useful for checking if EBS is throttled.

---

# 🔹 **3. %util — Disk Utilization**

**Most important field.**

* Shows **how busy the disk is**.
* If **%util is near 100%**, your disk is the bottleneck.

**Interpretation:**

* `< 60%` → healthy
* `60%–80%` → moderate
* `> 90%` → disk saturated
* `100%` → fully busy, I/O waits will increase

This directly affects system slowdown and SSH lag.

---

# 🔹 **4. await — Average Wait Time (ms)**

This tells you how long each I/O request waits before being processed.

* **< 10 ms** → excellent
* **10–20 ms** → okay
* **20–50 ms** → slow
* **> 50 ms** → disk is overloaded
* **> 100 ms** → severe bottleneck

If your SSH is slow and you see **high await**, the disk is the reason.

---

# 🔹 **5. svctm — Service Time (ms)**

How long the device *takes to complete* each request.

Modern kernels often calculate this differently, but still useful.

* High `svctm` = slow disk
* High `await` but **low `svctm`** = too many requests waiting in queue (I/O queue congestion)

---

# 🔹 **6. avgqu-sz — Queue Size**

How many I/O operations are waiting.

* Should be **near 0**.
* A value > **1** means disk cannot keep up.
* A value > **10** = major disk bottleneck.

---

# 🔹 **7. %iowait (CPU Section)**

Before device stats, iostat shows:

```
%user %nice %system %iowait %steal %idle
```

`%iowait` = CPU waiting for disk.

Interpretation:

* `< 5%` → normal
* `5–10%` → disk slow
* `> 20%` → major disk issue
* `> 40%` → system almost frozen due to disk

High `iowait` is the **#1 cause of slow SSH** on cloud instances.

---

# 📌 Example: Diagnosing a Slow Server Using iostat

If you run:

```
iostat -xz 1
```

And see something like:

```
Device:    %util  await  avgqu-sz
nvme0n1    99.8   150    25
```

This means:

* **Disk is 100% busy**
* I/O is waiting **150ms** → very slow
* Queue backlog of **25** → system overloaded
* CPU `%iowait` will be high
* SSH, commands, applications will lag

**Fix:** Choose bigger EBS volume/IOPS, reduce I/O, or optimize application.

---

# 📌 Example: Healthy Output

```
Device:    %util  await  avgqu-sz
nvme0n1     2.0   0.5    0.01
```

This means:

* Disk almost idle
* Very low latency
* No queue buildup
* Server is healthy

---

# 🧠 Why `iostat` Matters

`iostat` helps diagnose issues like:

* Slow SSH login
* Slow database queries
* High disk latency
* EBS throttling
* File system load
* Swap thrashing
* Application slowdowns

It shows you **what the disk is doing right now** and whether it's overloaded.

---

If you want, I can explain:

* How to use `iostat` + `pidstat` to find the exact process causing I/O
* How to check if AWS EBS throttling is happening
* How to use `iostat` to diagnose swap issues

Just tell me!

