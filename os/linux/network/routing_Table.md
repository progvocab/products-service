 the **Linux routing table** is at the heart of how your system decides **which network interface (NIC)** and **next hop** to use when sending packets.

Let’s break it down clearly — from **concept → structure → commands → examples → packet flow**.

---

## 🧭 **1. What is the Linux Routing Table?**

The **routing table** is a set of **rules** that tell the Linux kernel **where to send packets** based on their **destination IP address**.

Every outgoing packet passes through the routing table.
It determines:

* Which **interface** (NIC) to use
* Which **gateway** (next hop) to send it to
* Whether the destination is **local**, **on the same network**, or **remote**

---

## ⚙️ **2. Routing Table Structure**

Each routing table entry (route) has key fields:

| Field                | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| **Destination**      | The destination network or host (e.g., `192.168.1.0/24`, `default`) |
| **Gateway**          | Next hop IP address (router) to forward packets to                  |
| **Genmask / Prefix** | Network mask defining destination range                             |
| **Iface**            | Network interface used to send packets                              |
| **Metric**           | Priority — lower metric = higher preference                         |
| **Flags**            | Special attributes (U=up, G=uses gateway, H=host route)             |

---

## 🧰 **3. Viewing the Routing Table**

### 🔹 Modern Command (recommended)

```bash
ip route show
```

Example output:

```
default via 192.168.1.1 dev eth0 proto dhcp metric 100
192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.10 metric 100
```

### 🔹 Legacy Command

```bash
route -n
```

Example:

```
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         192.168.1.1     0.0.0.0         UG    100    0        0 eth0
192.168.1.0     0.0.0.0         255.255.255.0   U     100    0        0 eth0
```

---

## 🧠 **4. Understanding the Entries**

### 1️⃣ **Default Route**

* Used when no other route matches.
* Tells kernel which gateway to use to reach the internet.

```
default via 192.168.1.1 dev eth0
```

→ Send unknown traffic to router at `192.168.1.1` via interface `eth0`.

### 2️⃣ **Local Network Route**

```
192.168.1.0/24 dev eth0 scope link src 192.168.1.10
```

→ For destinations in `192.168.1.0–192.168.1.255`, send directly via `eth0`.

### 3️⃣ **Host Route**

```
10.0.0.5 via 10.0.0.1 dev eth1
```

→ Only packets to host `10.0.0.5` go through gateway `10.0.0.1` on `eth1`.

---

## 🧮 **5. Routing Decision Process**

When the kernel sends a packet, it checks the **routing table in this order**:

1. **Exact host route** (`/32`)
2. **Network route** (e.g. `/24`)
3. **Default route** (`0.0.0.0/0`)
4. If nothing matches → packet dropped (`Destination unreachable`)

📘 Example:

```
ip route show
default via 192.168.1.1 dev eth0
10.0.0.0/8 dev eth1 proto kernel scope link src 10.0.0.2
```

* Packet to `10.2.3.4` → goes via `eth1` (matches 10.0.0.0/8)
* Packet to `8.8.8.8` → goes via `192.168.1.1` (default)

---

## 🧱 **6. Adding or Deleting Routes**

### ➕ Add a route

```bash
sudo ip route add 10.0.0.0/24 via 192.168.1.254 dev eth0
```

### ❌ Delete a route

```bash
sudo ip route del 10.0.0.0/24
```

### 🧩 Add a default route

```bash
sudo ip route add default via 192.168.1.1
```

---

## 🌐 **7. Example Full Routing Table**

| Destination      | Gateway         | Interface | Purpose                  |
| ---------------- | --------------- | --------- | ------------------------ |
| `default`        | `192.168.1.1`   | `eth0`    | Internet (default route) |
| `10.0.0.0/24`    | `0.0.0.0`       | `eth1`    | Local office LAN         |
| `172.16.0.0/16`  | `192.168.1.254` | `eth0`    | Route to VPN via gateway |
| `192.168.1.0/24` | `0.0.0.0`       | `eth0`    | Local subnet             |
| `127.0.0.0/8`    | `0.0.0.0`       | `lo`      | Loopback interface       |

---

## 📊 **8. Packet Flow Example**

When you do:

```bash
ping 8.8.8.8
```

1. Kernel checks routing table:

   * No specific `8.8.8.8` route
     → Use **default route** via `192.168.1.1` on `eth0`.

2. Sends ARP request to find MAC of gateway `192.168.1.1`.

3. NIC `eth0` sends packet to gateway → router → Internet → 8.8.8.8.

---

## 🧠 **9. Advanced Topics**

* **Multiple Routing Tables:** `ip rule show` → policy-based routing.
* **Metrics:** Lower = higher priority.
* **Scope:**

  * `link`: directly reachable
  * `host`: local machine
  * `global`: other networks
* **Routing cache:** Kernel caches routes for efficiency.

---

## ✅ **Summary**

| Concept           | Description                                         |
| ----------------- | --------------------------------------------------- |
| **Routing Table** | List of rules telling kernel how to forward packets |
| **Default Route** | Used when no specific route matches                 |
| **Gateway**       | Next-hop router                                     |
| **Interface**     | NIC used for transmission                           |
| **Metric**        | Priority of route                                   |
| **Command**       | `ip route show` / `ip route add` / `ip route del`   |

---

Would you like me to draw a **Mermaid diagram** showing how Linux decides which interface or gateway to use when multiple routes match (host, network, default)? It makes the decision process visually clear.


In Linux, subnets are not defined in a single “subnet file.”
Instead, they are defined as part of the network interface IP configuration and stored inside the Linux kernel's routing table and interface configuration.

Here is exactly where they live:


---

✅ 1. Subnets are defined on network interfaces (NICs)

When you assign an IP to an interface, you also assign its subnet mask.

Example:

ip addr add 192.168.1.10/24 dev eth0

Here:

IP: 192.168.1.10

Subnet: /24 (255.255.255.0)

Interface: eth0


This defines the subnet for that interface.

You can view all subnets Linux knows using:

ip addr show

Linux does not keep a separate “subnet table.”
The interface configuration implicitly defines the subnets.


---

✅ 2. Subnets are also reflected in the Routing Table

Whenever you configure an IP/subnet on an interface, Linux automatically adds a route for that subnet.

Example:

ip route show

You will see:

192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.10

This line tells Linux:

Subnet 192.168.1.0/24 is reachable directly via eth0


So routing table = where Linux stores subnet routes.


---

✅ 3. Persistent definitions are stored in OS network configuration files

📌 On Ubuntu/Debian:

/etc/netplan/*.yaml

Example:

addresses:
  - 192.168.1.10/24
gateway4: 192.168.1.1

📌 On old Debian systems:

/etc/network/interfaces

📌 On RedHat/CentOS/Rocky:

/etc/sysconfig/network-scripts/ifcfg-eth0

Contains:

IPADDR=192.168.1.10
PREFIX=24

These files make the IP + subnet persistent across reboots.


---

🧠 Where exactly is the subnet stored?

There is no separate “subnet table.”

Subnets in Linux are stored in two places:

✔ Interface configuration

(ip addr → stores IP + prefix length)

✔ Kernel routing table

(ip route → stores local subnet routes)

This combination tells Linux:

Which subnets local interfaces belong to

Which subnets are directly reachable



---

📝 Simple Answer

Linux defines subnets as part of the IP configuration on each network interface, and these become entries in the kernel routing table. There is no separate subnet database.


---

If you want, I can also explain:

How Linux decides if a destination IP is in the same subnet

How ARP interacts with subnets

How AWS EC2 Linux handles subnet configuration


Just tell me!