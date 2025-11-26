Sure! Here are the main components of a router, explained clearly and grouped logically.
These apply to physical routers, virtual routers, and even routers in cloud platforms (AWS, GCP, Azure).


---

🧩 Major Components of a Router

1️⃣ Routing Table

Stores routes: networks → next-hop / interface

Used to decide where to forward packets

Contains:

Destination network

Subnet mask

Next-hop IP

Outgoing interface

Metric / cost




---

2️⃣ Forwarding Plane (FIB – Forwarding Information Base)

Optimized, hardware-level version of the routing table

Very fast lookup for forwarding packets

Built from routing table


Think: Routing table = brain, FIB = muscle


---

3️⃣ Routing Protocols (Control Plane)

Handle learning and updating routes.

Examples:

OSPF (interior routing)

BGP (internet-scale routing)

RIP (simple legacy)

EIGRP (Cisco)


Functions:

Exchange routes with neighbors

Detect network failures

Calculate best path

Update routing table



---

4️⃣ ARP Table (Address Resolution Protocol Table)

Maps:

IP → MAC


Used when sending packets on a LAN.

Example ARP entry:

10.0.0.5 → aa:bb:cc:dd:ee:ff


---

5️⃣ NAT Table (if router does NAT)

Used in home routers, cloud gateways, etc.

Handles:

Source NAT (SNAT)

Destination NAT (DNAT)

Port Address Translation (PAT)

Connection tracking


Example:

192.168.1.10:54321 → 100.10.20.30:40000


---

6️⃣ Firewall Rules (ACLs – Access Control Lists)

Many routers include Layer 3/4 firewall capabilities.

ACLs define:

Allow / deny rules

Inbound/outbound filtering

Source/destination IP/port rules


Example ACL:

deny tcp any 192.168.1.10 eq 22
permit ip any any


---

7️⃣ Interfaces (Physical & Logical)

Physical:

Ethernet ports

Fiber ports (SFP)

Wireless radios (in home routers)


Logical:

VLAN interfaces

Loopback interfaces

Subinterfaces

Tunnels (GRE/IPsec)



---

8️⃣ DHCP Server / Relay

Many routers include:

DHCP server: Assigns IPs to clients

DHCP relay: Forwards DHCP requests to a central server



---

9️⃣ QoS (Quality of Service)

Controls:

Priority of traffic

Bandwidth shaping

Rate limiting


Used for VoIP, video, gaming, etc.


---

🔟 Management Plane

Handles router administration:

SSH / Telnet

Web UI

SNMP monitoring

Logging

Firmware updates



---

🧠 Putting it all together

A router has 3 logical planes:

1. Control Plane

Routing protocols

Routing table

ARP table


2. Forwarding Plane

FIB

NAT translation

ACL filtering

Packet forwarding


3. Management Plane

Config/UI/SSH/SNMP



---

📦 Complete List of Router Components (Quick Summary)

Component	Function

Routing Table	Best paths to networks
FIB	Fast hardware forwarding
ARP Table	IP → MAC resolution
NAT Table	Address translation
ACL / Firewall	Allow/deny rules
Interfaces	Physical & logical interfaces
Routing Protocols	Learn/advertise routes
DHCP	IP allocation service
QoS Engine	Priority & bandwidth control
Management Plane	Admin access & monitoring



---

If you want, I can also explain:

How packets flow through a router step by step

Difference between routing table vs FIB vs ARP vs NAT

Router vs switch vs layer-3 switch


Just tell me!