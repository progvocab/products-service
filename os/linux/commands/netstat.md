`netstat` is a **network statistics** command in Linux used to view **network connections, routing tables, interface stats, listening ports, and protocol details**.

It is one of the most important commands for diagnosing:

* High network usage
* Port not listening
* Socket leaks
* TCP/UDP issues
* Connection states (TIME_WAIT, ESTABLISHED)

Although `ss` is the modern alternative, **netstat is still widely used**.

---

# 🔍 **What netstat Does**

`netstat` can show:

### ✅ Active network connections

### ✅ Ports your system is listening on

### ✅ Network interface statistics

### ✅ Routing table

### ✅ Multicast group info

### ✅ Protocol statistics (TCP/UDP)

---

# 🧠 **Basic Syntax**

```bash
netstat [options]
```

---

# 🧩 **Most Useful netstat Options**

## 1️⃣ **Show all listening ports**

```bash
netstat -tuln
```

Breakdown:

* **-t** → TCP
* **-u** → UDP
* **-l** → listening ports
* **-n** → show numeric (no DNS lookup)

**Output example**

| Proto | Local Address | Foreign Address | State  |
| ----- | ------------- | --------------- | ------ |
| tcp   | 0.0.0.0:22    | 0.0.0.0:*       | LISTEN |

---

## 2️⃣ **Show all connections including ESTABLISHED**

```bash
netstat -tunap
```

Options:

* **-a** → all connections (listening + active)
* **-p** → show process ID and program name

Example:

```
tcp  0  0 10.0.0.1:8080 10.0.0.5:50010 ESTABLISHED 1234/java
```

---

## 3️⃣ **Show routing table**

Same as `route -n`:

```bash
netstat -rn
```

Output:

| Destination | Gateway | Genmask | Iface |
| ----------- | ------- | ------- | ----- |

---

## 4️⃣ **Show per-interface statistics**

```bash
netstat -i
```

Shows:

* MTU
* RX/TX packets
* Errors
* Drops

Equivalent to:

```bash
ip -s link
```

---

## 5️⃣ **Show only TCP connections**

```bash
netstat -atn
```

---

## 6️⃣ **Show only UDP connections**

```bash
netstat -aun
```

---

## 7️⃣ **Show kernel protocol statistics (TCP/UDP counters)**

```bash
netstat -s
```

Output includes:

* TCP retransmissions
* connection resets
* listen overflows
* UDP errors
* ICMP echo counts

Example:

```
Tcp:
    135 active connections openings
    22 passive connection openings
    18 failed connection attempts
    10 connection resets received
    5 segments retransmitted
```

---

## 8️⃣ **Find which process is using a port**

```bash
netstat -tunlp | grep 8080
```

Example output:

```
tcp   0   0 0.0.0.0:8080   0.0.0.0:*   LISTEN   1234/java
```

---

# 📦 **Interpreting TCP States with netstat**

| State           | Meaning                                       |
| --------------- | --------------------------------------------- |
| **LISTEN**      | Waiting for incoming connections              |
| **ESTABLISHED** | Active connection                             |
| **TIME_WAIT**   | Connection closed, waiting for kernel timeout |
| **CLOSE_WAIT**  | Remote closed, local not closed               |
| **SYN_SENT**    | Client sent SYN                               |
| **SYN_RECV**    | Server received SYN                           |
| **FIN_WAIT1/2** | Closing handshake                             |
| **LAST_ACK**    | Waiting for remote ACK                        |
| **CLOSED**      | No connection                                 |

---

# 🧠 Why netstat is still used?

Even though **ss** is faster and newer, netstat remains useful because:

* It's installed on older servers
* Outputs are familiar for troubleshooting
* Many scripts rely on it
* Easier for beginners

---

# ✔️ Summary

| Task                    | netstat Command  |              |
| ----------------------- | ---------------- | ------------ |
| Show listening ports    | `netstat -tuln`  |              |
| Show active connections | `netstat -tunap` |              |
| Find process on a port  | `netstat -tunlp  | grep <port>` |
| Show interface stats    | `netstat -i`     |              |
| Show routing table      | `netstat -rn`    |              |
| Show protocol stats     | `netstat -s`     |              |

---

If you want, I can also explain:
✔ output line-by-line
✔ difference between `ss` and `netstat`
✔ how to detect network issues, socket leaks, TIME_WAIT storms
✔ commands to debug slow connections with examples
