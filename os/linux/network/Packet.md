# Network Packet

**packet life cycle** inside a Linux system  from user space → kernel → NIC → network → remote host → back:

```mermaid
sequenceDiagram
    participant App as Application (User Space)
    participant Socket as Socket API
    participant Kernel as Kernel (TCP/IP Stack)
    participant NIC as Network Interface Card
    participant Wire as Network (Switch/Router)
    participant Remote as Remote Host

    App->>Socket: Write data (send())
    Socket->>Kernel: Syscall triggers packet creation
    Kernel->>Kernel: TCP/IP processing\n(segment, headers, checksum)
    Kernel->>NIC: Pass packet to NIC driver (skb)
    NIC->>NIC: Packet queued & DMA transfer
    NIC->>Wire: Transmit frame (Ethernet)
    Wire->>Remote: Route packet to destination

    Remote->>Wire: Send ACK/response
    Wire->>NIC: Deliver incoming frame
    NIC->>Kernel: Interrupt → hand packet to stack
    Kernel->>Socket: Deliver payload to socket buffer
    Socket->>App: Application receives data (read())
```
**Network Interface Card (NIC)**
> A **Network Interface Card (NIC)** in Linux is the hardware (or virtual) component that connects a system to a network and handles the framing, sending, and receiving of packets at Layer 2. Each NIC is represented in Linux as a network device under `/sys/class/net` (e.g., `eth0`, `enp3s0`). The NIC offloads several low-level operations from the CPU, such as checksum calculation, segmentation (TSO/GSO), and packet filtering via hardware queues. The Linux kernel uses the **netdevice** subsystem and **drivers** to communicate with the NIC, managing RX/TX rings, interrupts (or NAPI polling), and buffer allocation through sk_buffs. The NIC advertises capabilities like speed, duplex, and offloads via `ethtool`. Overall, it serves as the bridge between the OS networking stack and the physical network medium, ensuring efficient and reliable packet transmission.

**Network Switch**

>A **network switch** is a Layer 2 device that connects multiple devices within the same local network and forwards Ethernet frames intelligently using MAC addresses. It maintains a **MAC address table** that maps each port to the devices connected behind it, allowing the switch to send frames only to the correct destination port instead of broadcasting to all ports. Switches operate using full-duplex links, support features like VLANs, Spanning Tree Protocol (STP), link aggregation, and QoS, and can be unmanaged or fully managed. By reducing collisions and isolating traffic per port, a switch provides fast, efficient, and secure communication within a LAN.

**Network Router**
>A **network router** is a Layer 3 device responsible for directing packets between different networks using IP addresses. Unlike switches, which forward frames within a single LAN, a router determines the best path for packets to reach destinations across multiple networks or the internet. It maintains a **routing table**, uses protocols like OSPF, BGP, RIP, or static routes, and performs tasks such as NAT, firewall filtering, DHCP, and packet forwarding. Routers break broadcast domains, support inter-VLAN routing, and ensure efficient, secure, policy-driven movement of traffic between internal networks and external networks.

Refer for Details  :
- [Layer 3 Routing Protocols](routing_protocols.md)
- [Layer 3 Protocols](layer_3_protocols.md)
- [Layer 4 Protocols](/design/system-design/load_balancer/layer4_protocols.md)
- [Layer 7 Protocols](/design/system-design/browser/protocols.md)

More :
- Packet life cycle in **reverse (incoming first)**
- Packet life cycle through **iptables/nftables hooks**
- Packet life cycle through **Linux routing and ARP**

 

## **Packet Flow Through Netfilter Hooks**

```mermaid
flowchart TD

subgraph INGRESS["Ingress Path (Incoming Packet)"]
    A[Packet Arrives on NIC] --> B[netdev: Receive]
    B --> C["PREROUTING (mangle, raw, nat)"]
    C --> D{Routing Decision}
    D -->|Destined to local machine| E["INPUT (mangle, filter)"]
    E --> F["Delivered to Socket / Application"]
    D -->|Forwarded packet| G["FORWARD (mangle, filter)"]
    G --> H[outdev Queue]
end

subgraph EGRESS["Egress Path (Outgoing Packet)"]
    I[Application Generates Packet] --> J["OUTPUT (raw, mangle, nat, filter)"]
    J --> K{Routing Decision}
    K --> L["POSTROUTING (mangle, nat)"]
    L --> M[netdev: Transmit]
    M --> N[Packet Leaves NIC]
end

H --> L
```


Incoming packets hit **PREROUTING** before routing; then based on routing they go to **INPUT** (local) or **FORWARD** (routed). Outgoing packets from applications go through **OUTPUT**, then **POSTROUTING**. NAT typically occurs in **PREROUTING** (DNAT) and **POSTROUTING** (SNAT). All packets finally pass through `netdev` before leaving/after arriving at the NIC.


More :  **only iptables**, **only nftables**, or **with connection tracking (conntrack)** .

 

##  Packet Life Cycle Through Linux Routing + ARP

```mermaid
flowchart TD

A["Application Generates Packet\n(Socket Write)"] --> B["Transport Layer\nTCP/UDP Header Added"]
B --> C["IP Layer\nSource+Destination IP Added"]
C --> D["Routing Lookup\n(FIB Table)"]
D -->|Destination is Local Subnet| E[Check ARP Cache]

D -->|Destination is Remote Network| F[Select Default Gateway]
F --> G["Check ARP Cache (Gateway IP)"]

E -->|MAC Found| H[Encapsulate Ethernet Frame]
G -->|MAC Found| H

E -->|MAC Not in Cache| I["Send ARP Request\n(Broadcast)"]
G -->|MAC Not in Cache| I

I --> J[Wait for ARP Reply]
J --> K[Update ARP Cache]
K --> H[Encapsulate Ethernet Frame]

H --> L[Send to NIC Driver]
L --> M[Packet Put on Wire]
```

**FIB (Forwarding Information Base)**

> A **FIB (Forwarding Information Base)** is the optimized, runtime version of the routing table used by routers to forward packets at high speed. While the routing table stores all learned routes, the FIB keeps only the best next-hop entries for each destination. It enables fast, hardware-accelerated packet forwarding by mapping destination IP prefixes directly to outgoing interfaces.

**ARP (Address Resolution Protocol)**

> The **ARP (Address Resolution Protocol)** maps an IPv4 address to its corresponding MAC address on a local network. When a device wants to send a packet to another host on the same LAN, it broadcasts an ARP request to learn the target’s MAC address. The resolved mappings are stored in an ARP cache to avoid repeated broadcasts and speed up communication.


When an application sends a packet, Linux performs a **routing lookup** to decide if the destination is on the same subnet or requires a gateway. Linux then checks the **ARP cache** for the corresponding MAC address. If missing, it sends an **ARP request**, waits for the reply, updates the ARP cache, wraps the IP packet inside an **Ethernet frame**, and finally sends it through the **NIC** onto the wire.

 More :**reverse path (incoming packets)**, **proxy ARP**, **gratuitous ARP**, or **ARP cache states (REACHABLE / STALE / DELAY / PROBE)**.
 
## **Packet Life Cycle Through Linux Routing + ARP**

including:
✔ **Incoming + outgoing packets**
✔ **ARP cache states (REACHABLE, STALE, DELAY, PROBE)**
✔ **Proxy ARP logic**
✔ **Gratuitous ARP handling**

```mermaid
flowchart TD

%% OUTGOING PACKET FLOW
A1[Application Sends Packet] --> A2["Transport Layer\nTCP/UDP Header"]
A2 --> A3[IP Layer Adds Header]
A3 --> A4["Routing Lookup\n(FIB)"]
A4 -->|Local Subnet| A5[Check ARP Cache]
A4 -->|Remote Network| A6["Use Default Gateway\nThen Check ARP Cache"]

%% ARP CACHE CHECK
A5 -->|"MAC Found (REACHABLE)"| A10[Encapsulate Ethernet Frame]
A6 -->|"MAC Found (REACHABLE)"| A10

A5 -->|MAC STALE| A7["Move to DELAY State\nSend Next Packet"]
A7 --> A8["If no traffic → PROBE\nSend unicast ARP Probe"]
A8 --> A9[Update to REACHABLE]
A9 --> A10

A5 -->|MAC Not Found| A11["Send ARP Request\n(Broadcast)"]
A6 -->|MAC Not Found| A11

A11 --> A12[Wait for ARP Reply]
A12 -->|Reply Received| A13["Update ARP Cache → REACHABLE"]
A13 --> A10
A12 -->|Timeout| A14["Retry \n→ Eventually Fail"]

A10 --> A15["NIC Driver Sends Frame\nOut to Wire"]

%% INCOMING PACKET FLOW
B1[Frame Received by NIC] --> B2[Ethernet Layer]
B2 -->|Destination MAC matches NIC| B3[Pass to IP Layer]
B2 -->|Not NIC but Proxy ARP enabled| B20["Proxy ARP: Respond With Own MAC"]
B3 --> B4[IP Routing Decision]

B4 -->|IP = Local Machine| B5[Send to Transport Layer]
B4 -->|IP = Forwarding Enabled| B6["Route (FIB) Lookup"]

B6 -->|Next Hop MAC in ARP Cache| B7[Forward Packet]
B6 -->|Next Hop MAC Missing| B8[Send ARP Request]
B8 --> B9[Cache MAC and Forward]

%% GRATUITOUS ARP
C1[Gratuitous ARP Received] --> C2["ARP Cache Updated\nor Conflict Detection"]

```

### **1. Outgoing Packet**

Linux builds a TCP/UDP segment → wraps it inside an IP packet → performs a routing lookup.
If the destination is:

* **Local subnet** → resolve its MAC via ARP
* **Remote network** → resolve gateway’s MAC via ARP

Linux checks the **ARP cache**. If the entry is:

* **REACHABLE** → use immediately
* **STALE** → move to DELAY → PROBE → REACHABLE
* **Missing** → send **ARP request** (broadcast), wait for reply

Once MAC is known, Linux builds an **Ethernet frame** and sends it out through the NIC.


### **2. Incoming Packet**

NIC receives an Ethernet frame and:

* If **MAC matches NIC**, deliver to IP layer.
* If MAC doesn’t match but **Proxy ARP** is enabled, Linux may respond with its own MAC.

If the packet is:

* **For this host** → pass to TCP/UDP
* **For another host & IP forwarding enabled** → use routing table → forward
  (Possibly triggering ARP requests for next-hop MAC)


### **3. Gratuitous ARP**

Used for:

* **Detecting IP conflicts**
* **Updating ARP caches of other hosts**
* **Failover / IP takeover** (keepalived, VRRP)

Linux updates its ARP table on receiving it.

More :
📌 **ARP cache lifecycle diagram** (REACHABLE → STALE → DELAY → PROBE → FAILED)
📌 **Diagram showing Kernel components (FIB, ARP table, Neighbour subsystem)**
📌 **Packet path through iptables → routing → ARP → NIC** (all together)

## **ARP / Neighbor Cache Lifecycle — Mermaid Diagram**
states:

* **INCOMPLETE**
* **REACHABLE**
* **STALE**
* **DELAY**
* **PROBE**
* **FAILED**

```mermaid
flowchart TD

Start["Packet Needs MAC\nLookup Neighbor Cache"] --> A1{Entry Exists?}

A1 -->|No Entry| INCOMPLETE
A1 -->|Yes| A2{State?}

%% NEW ENTRY
INCOMPLETE["INCOMPLETE\n(No MAC yet)\nSend ARP Request"] --> A3{ARP Reply?}
A3 -->|Yes| REACHABLE
A3 -->|"No (Timeout)"| FAILED

FAILED["FAILED\n(Resolution failed)"] -->|Retry after timeout| INCOMPLETE


%% EXISTING ENTRY
A2 -->|REACHABLE| REACHABLE
A2 -->|STALE| STALE
A2 -->|DELAY| DELAY
A2 -->|PROBE| PROBE

%% REACHABLE
REACHABLE["REACHABLE\n(MAC known & valid)"] -->|Reachable timer expires| STALE
REACHABLE -->|Used by traffic| REACHABLE

%% STALE
STALE["STALE\n(MAC old but usable)"] -->|Traffic arrives| DELAY
STALE -->|"If forwarded w/o traffic"| PROBE

%% DELAY
DELAY["DELAY\n(wait ~5 sec for traffic)"] -->|No confirmation| PROBE
DELAY -->|Traffic confirms| REACHABLE

%% PROBE
PROBE["PROBE\n(Send Unicast ARP Probe)"] --> A4{Reply?}
A4 -->|Yes| REACHABLE
A4 -->|No| FAILED

```
 
 
### **1. INCOMPLETE**

* ARP request has been sent.
* Still waiting for an ARP reply.

### **2. REACHABLE**

* MAC is valid.
* Linux uses a **reachability timer** (typically 30s).
* Any traffic refreshes state.

### **3. STALE**

* Timer expired; MAC may still work.
* Linux *will not probe yet*.
* First use will transition to **DELAY** or **PROBE**.

### **4. DELAY**

* Linux waits ~5 seconds to see if incoming traffic confirms reachability.
* If none arrives → move to **PROBE**.

### **5. PROBE**

* Linux sends **unicast ARP probes**.
* If reply received → REACHABLE.
* If not → FAILED.

### **6. FAILED**

* MAC resolution failed.
* Retries after exponential backoff.


More :
📌 **Full Packet Flow + Neighbor Table (ARP) + Routing + iptables hooks** in a single diagram
📌 **Linux Neighbor Subsystem architecture diagram**
📌 **ARP vs NDP lifecycle (IPv4 vs IPv6)**
 
