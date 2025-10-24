Excellent question — understanding the difference between **Layer 3** and **Layer 4 load balancers** is fundamental to networking and system design.
Let’s break this down clearly 👇

---

## 🌐 OSI Layer Reference

| OSI Layer                     | Example Protocols | What it Handles                |
| ----------------------------- | ----------------- | ------------------------------ |
| **Layer 3 – Network Layer**   | IP, ICMP          | Routing packets (IP addresses) |
| **Layer 4 – Transport Layer** | TCP, UDP          | Managing connections & ports   |

---

## ⚙️ Layer 3 Load Balancer (Network Layer)

### 💡 What it Does

A **Layer 3 (L3) load balancer** routes traffic purely based on **IP addresses**.
It doesn’t look at TCP ports or payload data — it just decides **which server IP** should receive a packet.

### 🧠 How it Works

* Operates at **IP packet level**
* Uses **Network Address Translation (NAT)** or **IP routing**
* Often implemented as part of routers or firewalls
* Doesn’t maintain per-session state (stateless)

### ⚡ Characteristics

| Feature                | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| **Routing basis**      | Source/destination IP                                                  |
| **Protocol awareness** | None (blind to TCP/UDP)                                                |
| **Performance**        | Very fast — simple packet forwarding                                   |
| **Use case**           | Large-scale network routing, DDoS protection, global load distribution |

### 🔧 Examples

* ECMP (Equal Cost Multi-Path Routing) in routers
* Linux `ipvs` in L3 mode
* AWS **Gateway Load Balancer**

---

## ⚙️ Layer 4 Load Balancer (Transport Layer)

### 💡 What it Does

A **Layer 4 (L4) load balancer** operates on **TCP/UDP connections** — it can inspect **IP + port numbers** and manage **entire sessions**.

### 🧠 How it Works

* Makes decisions using **<src IP, src port, dst IP, dst port>**
* Can handle **TCP handshakes**
* May support **health checks** and **session persistence**
* Can perform **NAT**, **connection tracking**, and **port translation**

### ⚡ Characteristics

| Feature                | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| **Routing basis**      | IP + Port                                                   |
| **Protocol awareness** | Transport (TCP, UDP)                                        |
| **Performance**        | High — minimal overhead, but tracks sessions                |
| **Use case**           | Web traffic load balancing, game servers, streaming systems |

### 🔧 Examples

* AWS **Network Load Balancer (NLB)**
* **HAProxy**, **NGINX Stream module** (TCP mode)
* **Envoy Proxy** (L4 mode)

---

## 🧩 Quick Comparison

| Feature            | Layer 3 (Network)                 | Layer 4 (Transport)                  |
| ------------------ | --------------------------------- | ------------------------------------ |
| Operates On        | IP packets                        | TCP/UDP connections                  |
| Decides Using      | IP address                        | IP + Port                            |
| Protocol Awareness | None                              | TCP/UDP aware                        |
| Connection State   | Stateless                         | Stateful                             |
| Latency            | Lowest                            | Slightly higher                      |
| Load Decision      | Based on routing                  | Based on session/port                |
| Typical Use        | Routers, firewalls, edge gateways | Web servers, APIs, DB load balancing |
| Example            | AWS Gateway LB                    | AWS Network LB, HAProxy              |

---

## 🔭 In System Design Context

When designing scalable systems:

* **L3 LB** — used for routing traffic across **data centers or zones**
* **L4 LB** — used within a cluster to distribute **TCP connections** (e.g., NGINX, HAProxy before your app servers)

---

Would you like me to extend this with **Layer 7 load balancer** (HTTP-aware, e.g., NGINX, ALB) — so you can see how it fits into a **three-tier load balancing architecture (L3 → L4 → L7)** diagram?
