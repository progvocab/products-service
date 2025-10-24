Layer 4 of the **OSI model** is the **Transport Layer** — it’s responsible for **end-to-end communication**, **reliability**, **flow control**, and **segmentation/reassembly** of data between applications running on different hosts.

Here are the **main protocols** that operate at **Layer 4 (Transport Layer):**

---

### 🧩 **1. Transmission Control Protocol (TCP)**

* **Type:** Connection-oriented
* **Reliability:** Reliable (guarantees delivery, order, and integrity)
* **Use cases:** Web (HTTP/HTTPS), Email (SMTP, IMAP), File transfer (FTP)
* **Features:**

  * 3-way handshake for connection establishment
  * Sequencing and acknowledgment
  * Flow control (windowing)
  * Error detection and retransmission
  * Congestion control

---

### ⚡ **2. User Datagram Protocol (UDP)**

* **Type:** Connectionless
* **Reliability:** Unreliable (no guarantee of delivery or order)
* **Use cases:** Real-time apps (VoIP, video streaming, DNS, DHCP, gaming)
* **Features:**

  * Low latency
  * No handshake or retransmission
  * Simple header, less overhead

---

### 🔐 **3. Datagram Congestion Control Protocol (DCCP)**

* **Type:** Connection-oriented but unreliable
* **Reliability:** Provides congestion control but no guaranteed delivery
* **Use cases:** Streaming media, telephony (where delay matters more than loss)
* **Features:**

  * Negotiates congestion control mechanisms
  * Designed for applications that prefer timely delivery over reliability

---

### 🔁 **4. Stream Control Transmission Protocol (SCTP)**

* **Type:** Connection-oriented, message-based
* **Reliability:** Reliable and ordered (but supports multiple streams)
* **Use cases:** Telecommunication signaling (SS7 over IP), WebRTC, some IoT
* **Features:**

  * Multi-streaming (parallel logical channels in one connection)
  * Multi-homing (can use multiple network paths)
  * Reliable message delivery

---

### 🌐 **5. RSVP (Resource Reservation Protocol)**

* **Type:** Connection-oriented (for QoS)
* **Reliability:** Works with IP to reserve resources, not for data delivery
* **Use cases:** Multimedia streaming, VoIP with QoS
* **Features:**

  * Reserves bandwidth across a network path
  * Works alongside IP to manage Quality of Service

---

### 🧠 **6. QUIC (Quick UDP Internet Connections)** *(Modern / Layer 4–5 hybrid)*

* **Type:** Connection-oriented (built over UDP)
* **Reliability:** Reliable, multiplexed streams (like TCP)
* **Use cases:** HTTP/3, modern web apps
* **Features:**

  * Combines TCP-like reliability with UDP’s low latency
  * Built-in encryption (TLS 1.3)
  * Faster connection establishment (0-RTT)

---

### Summary Table

| Protocol | Connection Type           | Reliable   | Key Features                                   | Common Use Cases     |
| -------- | ------------------------- | ---------- | ---------------------------------------------- | -------------------- |
| **TCP**  | Connection-oriented       | ✅ Yes      | Sequencing, retransmission, congestion control | HTTP, FTP, SMTP      |
| **UDP**  | Connectionless            | ❌ No       | Fast, minimal overhead                         | DNS, VoIP, streaming |
| **DCCP** | Connection-oriented       | ⚠️ Partial | Congestion control without reliability         | Media streaming      |
| **SCTP** | Connection-oriented       | ✅ Yes      | Multi-streaming, multi-homing                  | Telecom signaling    |
| **RSVP** | Connection-oriented       | ❌ N/A      | Resource reservation (QoS)                     | VoIP, multimedia     |
| **QUIC** | Over UDP (pseudo-layer 4) | ✅ Yes      | TLS encryption, multiplexing                   | HTTP/3               |

---

Would you like me to show a **diagram of how Layer 4 fits with Layers 3 and 5**, showing how these protocols interact with IP and applications?
