TCP (Transmission Control Protocol) ensures **reliable, ordered, and congestion-aware** delivery of data across unreliable IP networks using a combination of **algorithms**, **design mechanisms**, and **control protocols**.
Let’s break this down by the main aspects: reliability, flow control, and congestion control.

---

## 🧩 1. **Reliable Delivery**

TCP guarantees that data sent by one end arrives correctly and in order at the other end.

### 🔹 Key Algorithms and Mechanisms:

| Mechanism                        | Description                                                                    | Algorithm / Design                    |
| -------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------- |
| **Sequence Numbers**             | Each byte is numbered; receiver uses this to reassemble data in order.         | Part of TCP header.                   |
| **Acknowledgments (ACKs)**       | Receiver sends ACKs for successfully received segments.                        | Cumulative ACKs (RFC 793).            |
| **Retransmission Timeout (RTO)** | If sender doesn’t receive ACK within timeout, retransmit.                      | **RTO = SRTT + 4×RTTVar** (RFC 6298). |
| **Fast Retransmit**              | Detects loss using duplicate ACKs instead of waiting for timeout.              | Triggered after 3 duplicate ACKs.     |
| **Checksum**                     | Detects bit errors in segment header and data.                                 | 16-bit 1’s complement sum.            |
| **Out-of-Order Buffering**       | Receiver buffers segments that arrive out of order until missing data arrives. | Implemented in TCP reassembly queue.  |

🧠 **Design Principle:** End-to-end acknowledgment and retransmission provide reliability even over unreliable IP.

---

## ⚖️ 2. **Flow Control**

Flow control prevents the sender from overwhelming the receiver’s buffer.

### 🔹 Mechanism: **Sliding Window Protocol**

| Component                             | Description                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| **Receiver Advertised Window (rwnd)** | Receiver tells sender how much data it can accept (in bytes).                 |
| **Sender Window**                     | Limits unacknowledged data to the min of `rwnd` and congestion window (cwnd). |
| **Dynamic Adjustment**                | As the receiver processes data, it sends updated `rwnd` in ACKs.              |

**Algorithm:**

* Implemented using a **credit-based** system (receiver advertises remaining buffer space).
* Sender maintains a **send window = min(rwnd, cwnd)**.

🧠 **Design Principle:** Keep sender and receiver in sync, ensuring the receiver never runs out of buffer space.

---

## 🌐 3. **Congestion Control**

Congestion control prevents the network itself from being overloaded.

### 🔹 Core Algorithms (RFC 5681 and later):

| Algorithm                           | Purpose                                      | Behavior                                                               |
| ----------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------- |
| **Slow Start**                      | Probe available bandwidth gradually.         | Start with cwnd = 1 MSS; double cwnd every RTT until loss or ssthresh. |
| **Congestion Avoidance (AIMD)**     | Control growth after threshold reached.      | cwnd += 1/cwnd per ACK (linear growth).                                |
| **Fast Retransmit / Fast Recovery** | Detect and recover from packet loss quickly. | Retransmit after 3 duplicate ACKs; halve cwnd; skip slow start.        |
| **Tahoe / Reno / NewReno / Cubic**  | Variants optimizing cwnd adjustment.         | Cubic is default in Linux.                                             |

### ⚙️ Modern Enhancements:

* **TCP Cubic (Linux default):** cwnd growth follows a cubic function → faster recovery and better high-speed utilization.
* **BBR (Bottleneck Bandwidth and RTT):** Uses bandwidth and RTT models instead of packet loss as congestion signals.

🧠 **Design Principle:** Treat the network as a shared resource; probe for bandwidth while backing off during congestion.

---

## 🧱 4. **Putting It All Together**

```
Application
   ↓
TCP Layer
   ├── Sequence Numbers + ACKs  → Reliable delivery
   ├── Sliding Window           → Flow control
   ├── Congestion Window (cwnd) → Congestion control
   ↓
IP Layer
   ↓
Network
```

---

## 🧮 Summary Table

| Category           | Mechanism            | Algorithm / Concept  | Key RFC / Idea |
| ------------------ | -------------------- | -------------------- | -------------- |
| Reliability        | ACK + Retransmission | RTO, Fast Retransmit | RFC 6298       |
| Reliability        | Ordering             | Sequence Numbers     | RFC 793        |
| Reliability        | Error Detection      | Checksum             | RFC 793        |
| Flow Control       | Receiver Window      | Sliding Window       | RFC 793        |
| Congestion Control | Bandwidth Probing    | Slow Start, AIMD     | RFC 5681       |
| Congestion Control | Fast Recovery        | Reno, NewReno, Cubic | RFC 6582, 8312 |
| Congestion Control | Model-based          | BBR                  | Google (2016)  |

---

Would you like me to include a **diagram showing how TCP manages its send and receive windows** (with cwnd, rwnd, ACKs, and retransmissions)?
Excellent question — let’s go **under the hood** and see **where exactly TCP’s reliability, flow control, and congestion control mechanisms are implemented in Linux.**

---

# 🧩 Where TCP Algorithms Are Implemented (in Linux)

The TCP stack in Linux is implemented in the **kernel networking subsystem**, primarily under:

```
/net/ipv4/tcp_*
```

Let’s break it down by functionality 👇

---

## ⚙️ 1. **Core TCP Logic**

| Functionality                                            | Source File              | Description                                                                                                                                          |
| -------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core TCP operations (connect, send, recv, state machine) | `/net/ipv4/tcp.c`        | Implements `tcp_sendmsg()`, `tcp_recvmsg()`, connection handling, and TCP state transitions (SYN, ACK, FIN, etc).                                    |
| Socket operations interface                              | `/net/ipv4/af_inet.c`    | Defines TCP as an internet domain socket type (`SOCK_STREAM`) and links user-space syscalls (like `send()`, `recv()`, `connect()`) to TCP functions. |
| TCP state transitions                                    | `/net/ipv4/tcp_input.c`  | Handles incoming segments, ACKs, retransmissions, and duplicate detection.                                                                           |
| Outgoing segment handling                                | `/net/ipv4/tcp_output.c` | Builds TCP headers, manages send queue, fragmentation, retransmission.                                                                               |

---

## 🧱 2. **Reliability Mechanisms**

| Mechanism                  | Implementation Location                     | Key Functions / Structures                                               |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| Sequence numbers, ACKs     | `/net/ipv4/tcp_input.c`                     | `tcp_ack()`, `tcp_clean_rtx_queue()`                                     |
| Retransmission timer (RTO) | `/net/ipv4/tcp_timer.c`                     | `tcp_retransmit_timer()`, `tcp_retransmit_skb()`                         |
| Fast retransmit            | `/net/ipv4/tcp_input.c`                     | `tcp_fastretrans_alert()`                                                |
| Checksum                   | `/net/ipv4/tcp_output.c`, `/lib/checksum.c` | `tcp_v4_send_check()` (software) or offloaded to NIC (checksum offload). |

---

## ⚖️ 3. **Flow Control**

| Mechanism                  | Implementation                                    | Description                                               |
| -------------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| Receiver window (rwnd)     | `tcp_select_window()` in `/net/ipv4/tcp_output.c` | Advertises available buffer space to sender.              |
| Sliding window tracking    | `struct tcp_sock` (in `/include/net/tcp.h`)       | Maintains `snd_una`, `snd_nxt`, `rcv_nxt`, and `rcv_wnd`. |
| Send queue / receive queue | `sk_buff` structures                              | Buffers data packets before/after transmission.           |

> 🔍 **Key data structure:**
> `struct tcp_sock` — stores per-connection state like sequence numbers, cwnd, RTT estimates, timers, etc.

---

## 🌐 4. **Congestion Control**

| Algorithm                      | File                    | Key Functions                                                                           |
| ------------------------------ | ----------------------- | --------------------------------------------------------------------------------------- |
| Generic congestion control API | `/net/ipv4/tcp_cong.c`  | Provides hooks to register congestion control algorithms (`struct tcp_congestion_ops`). |
| Reno (classic)                 | `/net/ipv4/tcp_cong.c`  | Built-in default fallback.                                                              |
| Cubic (default on Linux)       | `/net/ipv4/tcp_cubic.c` | Implements cubic cwnd growth and fast convergence.                                      |
| BBR (model-based)              | `/net/ipv4/tcp_bbr.c`   | Estimates bottleneck bandwidth and RTT (Google’s algorithm).                            |

> You can check which congestion algorithms are available on your system:
>
> ```bash
> sysctl net.ipv4.tcp_available_congestion_control
> ```
>
> Change it with:
>
> ```bash
> sysctl -w net.ipv4.tcp_congestion_control=cubic
> ```

---

## 🧮 5. **Timers and Retransmission**

| Component            | File                    | Function                 |
| -------------------- | ----------------------- | ------------------------ |
| Retransmission timer | `/net/ipv4/tcp_timer.c` | `tcp_retransmit_timer()` |
| Delayed ACK timer    | `/net/ipv4/tcp_timer.c` | `tcp_delack_timer()`     |
| Keepalive timer      | `/net/ipv4/tcp_timer.c` | `tcp_keepalive_timer()`  |

Timers are managed using **kernel’s softirq (software interrupt)** system for efficient async processing.

---

## 🧠 6. **Where Flow Meets OS Kernel**

The TCP stack interacts with the OS kernel and hardware through:

| Layer         | Functionality                           | Code                    |
| ------------- | --------------------------------------- | ----------------------- |
| Socket Layer  | Converts syscalls → protocol operations | `/net/socket.c`         |
| Network Layer | Encapsulates TCP inside IP packets      | `/net/ipv4/ip_output.c` |
| Device Layer  | Passes packets to/from NIC              | `/net/core/dev.c`       |
| NIC Driver    | DMA + Interrupts                        | `/drivers/net/*`        |

---

## 📜 7. **User-space Interface**

From user space (like a C program), when you call:

```c
send(sock, buf, len, 0);
```

The kernel call stack roughly looks like:

```
sys_sendto()
 └── sock_sendmsg()
      └── inet_sendmsg()
           └── tcp_sendmsg()
                └── tcp_write_xmit()
                     └── ip_queue_xmit()
                          └── dev_queue_xmit()
                               └── NIC Driver
```

Each layer applies its own logic:

* TCP handles retransmission, ACKs, congestion control.
* IP handles routing and fragmentation.
* NIC driver handles transmission to the physical network.

---

## 📘 Summary Diagram

```
User Space
└── send(), recv()
     ↓
Kernel Space (Networking Stack)
 ├── tcp.c (Core TCP logic)
 ├── tcp_input.c (ACKs, reassembly)
 ├── tcp_output.c (Send, retransmit)
 ├── tcp_timer.c (Timers)
 ├── tcp_cong.c / tcp_cubic.c / tcp_bbr.c (Congestion Control)
     ↓
 └── ip_output.c → dev_queue_xmit() → NIC driver
         ↓
Hardware / Network
```

---

Would you like me to show a **call flow diagram** (with function names) for a single TCP packet transmission — from `send()` in user-space → down to NIC driver?
