Here are the main components of a switch, similar to how we listed components for a router — but focused on Layer 2 (Data Link Layer) operations.


---

🧩 Major Components of a Network Switch

1️⃣ MAC Address Table (CAM Table)

The core component of a switch.

Stores: MAC → Port mappings

Used to decide which port to forward frames to

Learns entries dynamically from incoming traffic


Example CAM entry:

AA:BB:CC:DD:EE:FF → Port 3


---

2️⃣ Switching Fabric (Forwarding Plane)

The internal high-speed circuitry that:

Moves frames from one port to another

Decides forwarding, flooding, or filtering

Operates at wire speed


This is the switch's "muscle".


---

3️⃣ Spanning Tree Protocol (STP / RSTP / MSTP)

Prevents Layer 2 loops.

Functions:

Detects loops

Blocks redundant links

Maintains one active path

Rapid failover when a link goes down


Variants:

STP (original)

RSTP (faster)

MSTP (multiple spanning trees)



---

4️⃣ VLAN Table / VLAN Database

Used by Layer 2 switches that support VLANs.

Stores:

VLAN IDs

Ports assigned to each VLAN

VLAN membership (tagged/untagged)


Example:

VLAN 10 → Ports 1,2,3 (untagged)
VLAN 20 → Port 4 (tagged)


---

5️⃣ ARP Table (Layer 3 switches only)

If the switch is a Layer 3 switch, it has:

ARP table

Routing table

FIB
Just like a router.


But a pure Layer 2 switch does not use ARP.


---

6️⃣ ACLs (Access Control Lists)

Many managed switches support filtering rules:

Allow/deny MAC addresses

Allow/deny IP based (L3 switches)

Port security


Used for:

Security

Limiting traffic

Preventing unauthorized devices



---

7️⃣ Port Security

Controls access at the MAC level.

Supports:

Sticky MAC (remember MACs)

Maximum MACs per port

Violation actions (shutdown, restrict)


Example:

Allow max 1 MAC per port


---

8️⃣ QoS Engine

Quality of service features:

Priority queues

Bandwidth limiting

Traffic shaping

DSCP or CoS prioritization


Used for VoIP, video, and critical apps.


---

9️⃣ Management Plane

All management tools for configuring the switch:

Web UI

CLI (SSH/Telnet)

SNMP

Logging

Firmware updates

Syslog



---

🔟 Multicast / Broadcast Handling

Switches handle:

Flooding unknown unicast

Forwarding broadcasts

IGMP Snooping for multicast efficiency



---

🧠 Putting it all together

A switch operates mostly at Layer 2, so its main job is:

Learn MACs

Build CAM table

Forward frames based on MAC


Main internal planes:

1. Control Plane

STP/MSTP

VLAN configuration

MAC learning

Port security


2. Forwarding Plane

CAM table lookup

Hardware-based forwarding

QoS

IGMP snooping


3. Management Plane

Admin configuration

SNMP

Logging



---

📦 Complete Summary: Components of a Switch

Component	Function

MAC Table (CAM)	Maps MAC → Port for frame forwarding
Switching Fabric	Hardware path for moving frames
VLAN Table	VLAN membership + segmentation
STP/RSTP/MSTP	Loop prevention
ARP (L3 switch)	IP → MAC mapping
Routing Table (L3 switch)	Inter-VLAN routing
ACLs	Filtering traffic
Port Security	MAC-level access control
QoS Engine	Packet prioritization
IGMP Snooping	Efficient multicast handling
Management Plane	UI, CLI, SNMP, config, logs



---

If you want, I can also explain:

Difference between hub vs switch vs router

Packet flow inside a switch

Difference between CAM table vs ARP table

Difference between Layer 2 switch vs Layer 3 switch


Just tell me!