Here are **medium-level Linux questions** for a DevOps engineer — clear, focused, and 3–4 lines each:

---

### **1️⃣ How does Linux handle process isolation, and what is the relationship between namespaces and cgroups in container runtimes like Docker? Explain with real use cases.**

---

### **2️⃣ A service keeps failing randomly. How would you combine `journalctl`, `systemctl`, and `/var/log` tools to trace the root cause? Explain your approach.**

---

### **3️⃣ What is the difference between a hard link and a symbolic link? In which situations would each fail or become invalid?**

---

### **4️⃣ Explain how `strace` works internally and how you would use it to debug a hanging process in production without restarting it.**

---

### **5️⃣ How does Linux determine which DNS server to use when resolving a hostname? Explain the flow involving `/etc/resolv.conf`, `nsswitch.conf`, and glibc.**

---

### **6️⃣ A disk is showing 100% usage but you cannot find large files. What’s happening, and how would you troubleshoot using `lsof`, `inode` checks, and deleted file handles?**

---

### **7️⃣ What is the difference between `nice`, `renice`, and `ionice`, and how do they influence CPU and I/O scheduling for long-running background jobs?**

---

### **8️⃣ Explain the life cycle of a packet in Linux networking when a request enters an interface. Include ARP, routing table lookup, and iptables involvement.**

---

### **9️⃣ How does Linux handle file system caching? Explain page cache, buffer cache, and why `free -h` often shows low “available” memory.**

---

### **🔟 What is the difference between a process and a thread in Linux from a kernel perspective? How does the scheduler treat them differently?**

---

If you want, I can also generate:

✅ **Hard DevOps Linux questions**
✅ **Command-based scenario questions (e.g., CPU spike, disk full, network down)**
✅ **Kubernetes + Linux combined questions**
Here are **very short, concise paragraph answers** for all 10 Linux DevOps questions:

---

### **1️⃣ Namespaces vs cgroups**

Linux **namespaces** isolate resources (PIDs, network, mounts, users), while **cgroups** limit and track resource usage (CPU, memory, I/O). Containers like Docker use namespaces to isolate each container and cgroups to enforce limits, giving lightweight process-level virtualization.

---

### **2️⃣ Debugging with journalctl + systemctl**

`systemctl status` shows the service state and recent failures, `journalctl -u service` gives full historical logs, and `/var/log` provides application-level logs. Combining them reveals whether the failure is due to misconfiguration, dependency issues, crashes, or environment errors.

---

### **3️⃣ Hard link vs symlink**

A **hard link** points directly to the inode, so it survives file moves but cannot span filesystems. A **symlink** is a shortcut path, works across filesystems, but breaks if the target is deleted or moved. Hard links fail only if the inode itself is removed.

---

### **4️⃣ How strace helps debugging**

`strace` intercepts system calls of a running process, letting you see what the kernel is doing. For a hanging process, using `strace -p <pid>` immediately reveals if it's stuck on I/O, waiting for a lock, or failing syscalls—without restarting the service.

---

### **5️⃣ DNS resolution flow**

glibc follows rules in `/etc/nsswitch.conf` to pick resolvers, then reads `/etc/resolv.conf` for DNS servers. The resolver library performs lookups in the specified order (files → DNS → others), meaning DNS behavior depends on both nsswitch and resolver config.

---

### **6️⃣ 100% disk but no big files visible**

Often caused by a **deleted file still held open by a running process**, consuming space invisibly. Using `lsof | grep deleted` shows which process retains the file. Another cause is **inode exhaustion**, detectable via `df -i`, even when space appears free.

---

### **7️⃣ nice, renice, ionice**

`nice` sets initial CPU priority, `renice` changes it for running processes, and `ionice` sets I/O scheduling priority. Together they let DevOps reduce interference from background tasks by lowering CPU or disk access priority without stopping work.

---

### **8️⃣ Packet life cycle**

When a packet arrives, Linux checks ARP for MAC resolution, then routes it using the routing table. Netfilter hooks (iptables) inspect or modify it, and finally the kernel hands it to the socket buffer of the target process. Outbound packets follow the reverse stages.

---

### **9️⃣ Linux memory caching**

Linux uses **page cache** for file reads and **buffer cache** for block device operations, aggressively caching to improve performance. This makes free memory look low, but it’s intentional—cached memory is instantly reclaimable, shown under “available” in `free`.

---

### **🔟 Process vs thread**

A process has its own virtual memory and resources, while threads share the same address space within a process. The Linux scheduler treats threads as separate schedulable tasks (lightweight processes), so each thread competes independently for CPU time.

---

If you want, I can generate **more medium/hard Linux questions**, or produce **scenario-based DevOps problems (real-world style)**.
Here are **medium-to-advanced Linux administrator questions**, each 3–4 lines, suitable for interviews:

---

### **1️⃣ Explain how Linux boots from BIOS/UEFI to the login prompt. Which components (GRUB, initramfs, systemd) participate, and what does each one do?**

---

### **2️⃣ A server is slow and shows high load average, but CPU usage is low. What conditions cause this, and how would you diagnose using `vmstat`, `iostat`, and run queue analysis?**

---

### **3️⃣ How does Linux handle file permissions internally? Explain the role of inode metadata, umask, SUID/SGID/sticky bits, and ACLs.**

---

### **4️⃣ Describe how Linux handles memory pressure. What are page cache, swap, page faults, OOM killer, and how does the kernel decide which processes to kill?**

---

### **5️⃣ How does Linux networking determine packet routing? Explain the routing table, ARP, NAT, and how `ip route` and `ip neigh` fit together.**

---

### **6️⃣ What is the difference between EXT4, XFS, and Btrfs? Explain their metadata handling, journaling, snapshot support, and typical use cases.**

---

### **7️⃣ Why would a process show “D” (uninterruptible sleep) state? What usually causes it, and how do you investigate it using `ps`, `strace`, and kernel logs?**

---

### **8️⃣ How does Linux schedule CPU time among processes? Explain CFS (Completely Fair Scheduler), time slices, priority classes, and how `nice` affects scheduling.**

---

### **9️⃣ Explain how RAID works in Linux using `mdadm`. What are the differences between RAID 0, 1, 5, 6, and 10, and when would you choose each?**

---

### **🔟 Describe how SELinux enforces security. Explain contexts, policies, modes, labeling, and how AVC denials are logged and resolved.**

---

### **1️⃣1️⃣ A file system reports 100% usage even after deleting files. What causes this, and how do you fix it without rebooting?**

---

### **1️⃣2️⃣ How does systemd manage services? Explain units, dependencies, targets, and how systemd’s parallel startup differs from SysV.**

---

### **1️⃣3️⃣ How does Linux assign IP addresses and manage DHCP? Explain NetworkManager vs `netplan`, `/etc/network/interfaces`, and DHCP lease handling.**

---

### **1️⃣4️⃣ What happens internally when you run `kill -9` on a process? Why can't a SIGKILL signal be trapped?**

---

### **1️⃣5️⃣ How do Linux file systems recover from crashes? Explain journaling modes, fsck phases, and what happens when metadata corruption is detected.**

---

If you want, I can also provide:

✅ concise answers
✅ scenario-based Linux admin problems
✅ command-based troubleshooting questions
✅ shell scripting questions

Just tell me!
Here are **very short, concise paragraph answers** for all Linux Administrator questions:

---

### **1️⃣ Linux boot process**

Linux boots via BIOS/UEFI → GRUB loads the kernel → kernel loads `initramfs` (drivers, root FS setup) → kernel mounts root filesystem → `systemd` starts, launching services and bringing the system to the login target.

---

### **2️⃣ High load, low CPU**

High load with low CPU usually indicates tasks stuck in disk I/O, waiting for locks, or running in uninterruptible sleep (D-state). Tools like `vmstat`, `iostat`, and run queue stats show if the bottleneck is I/O, swapping, or blocked processes.

---

### **3️⃣ File permissions**

Linux permissions come from inode metadata, using user/group/others with rwx bits. Special bits include SUID, SGID, and sticky bit. ACLs extend permissions. `umask` subtracts permissions when new files are created.

---

### **4️⃣ Memory pressure**

Under pressure, Linux evicts file cache pages, then swaps anonymous memory. Page faults increase, and if memory is still low, the **OOM Killer** terminates the most memory-heavy process to protect system stability.

---

### **5️⃣ Routing logic**

Linux uses the routing table to choose the next hop, ARP resolves IP → MAC, and NAT rewrites IP headers if needed. `ip route` shows path decisions, while `ip neigh` shows ARP cache entries.

---

### **6️⃣ EXT4 vs XFS vs Btrfs**

EXT4 is stable with journaling, XFS handles large files and parallel I/O well, and Btrfs supports snapshots, compression, and CoW. Btrfs is flexible, XFS is high-performance, EXT4 is general-purpose.

---

### **7️⃣ D-state process**

A “D” state means a process is stuck in kernel mode waiting for I/O that cannot be interrupted—usually due to bad hardware, NFS hang, or disk issues. `strace` and kernel logs show where it is blocked.

---

### **8️⃣ CPU scheduling**

Linux uses CFS, which gives each process fair CPU time based on virtual runtime. `nice` changes priority by adjusting weight, so low-priority tasks get smaller slices and wait longer.

---

### **9️⃣ RAID levels**

RAID0 stripes for speed, RAID1 mirrors for redundancy, RAID5/6 use parity for balance between space and protection, RAID10 combines striping + mirroring for speed and safety. Managed using `mdadm`.

---

### **🔟 SELinux**

SELinux enforces mandatory access control using labels (user:role:type:level). Policy rules control which domains can access which objects. Violations create AVC denials logged in `/var/log/audit`. Modes include enforcing, permissive, and disabled.

---

### **1️⃣1️⃣ Disk full after delete**

A deleted file may still be held open by a running process, so space isn’t freed. `lsof | grep deleted` shows culprits. Restarting or forcing the process to close the file releases space.

---

### **1️⃣2️⃣ systemd**

Systemd uses unit files with dependencies and targets (runlevels). It starts services in parallel, unlike SysV’s sequential scripts. `systemctl` controls starting, stopping, enabling, and viewing logs.

---

### **1️⃣3️⃣ IP assignment**

Linux uses NetworkManager, netplan, or legacy `/etc/network/interfaces` to configure networking. DHCP clients obtain an IP, store leases, and renew them. Network stack updates routing and DNS accordingly.

---

### **1️⃣4️⃣ kill -9**

`kill -9` sends SIGKILL, which the kernel handles directly, terminating the process immediately. It cannot be trapped or ignored because it bypasses user-level signal handlers.

---

### **1️⃣5️⃣ File system recovery**

Journaling records metadata updates before writing them to disk. After a crash, fsck replays the journal, checks inodes, blocks, and directory structure, and repairs corruption if detected.

---

If you want, I can generate **hard-level Linux admin questions**, **scenario troubleshooting questions**, or **shell scripting questions**.
