### Predecessors of Unix

**Multics** was the major predecessor of Unix. It introduced key ideas such as hierarchical file systems, dynamic linking, multi-user time-sharing, and security rings. Multics’ complexity inspired the creation of a simpler, cleaner OS that became Unix.

### Unix and Its Major Successors

**BSD (Berkeley Software Distribution)** added TCP/IP networking stack, sockets API, virtual memory, job control, and the fast filesystem (FFS).
**Linux** adopted POSIX behavior, modular kernel architecture, dynamically loadable modules, ext2/ext3/ext4 filesystems, improved networking, and wide hardware support.

### New Features Introduced by BSD

* Integrated TCP/IP stack used everywhere today
* Sockets API enabling modern network programming
* Fast File System improving disk access performance

### New Features Introduced by Linux

* Open-source monolithic modular kernel
* Robust driver ecosystem and portability
* Advanced filesystems like ext4, XFS, Btrfs
* Process namespaces and cgroups enabling containers

### Successors of BSD

**FreeBSD**, **OpenBSD**, and **NetBSD** evolved from BSD.

* FreeBSD added advanced networking, jails for lightweight isolation, and ZFS support.
* OpenBSD focused on security with code auditing, PF firewall, and secure defaults.
* NetBSD specialized in portability across many hardware platforms.

### Example of Sockets API Introduced by BSD

```
int sock = socket(AF_INET, SOCK_STREAM, 0);
connect(sock, (struct sockaddr*)&addr, sizeof(addr));
send(sock, "hello", 5, 0);
```

This pseudo code represents a basic TCP client, used in real services such as web servers connecting to upstream APIs.

### Example of Linux Namespaces for Container Isolation

```
unshare(CLONE_NEWNET | CLONE_NEWPID);
execvp("/bin/bash", args);
```

This creates a shell inside isolated PID and network namespaces, forming the basis of container engines like Docker or Kubernetes runtimes.
