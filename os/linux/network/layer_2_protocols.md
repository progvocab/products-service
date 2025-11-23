### All Layer 2 (Data Link Layer) Protocols

These protocols operate at **OSI Layer 2** and are processed by the **Linux kernel networking stack**, NIC drivers, or switching components (bridge, OVS).

### Ethernet & LAN Technologies

* **Ethernet (IEEE 802.3)**
* **MAC addressing**
* **VLAN – IEEE 802.1Q**
* **Q-in-Q – IEEE 802.1ad**
* **Provider Backbone Bridging – IEEE 802.1ah**

### Switching & Spanning Tree

* **STP (Spanning Tree Protocol – 802.1D)**
* **RSTP (Rapid Spanning Tree – 802.1w)**
* **MSTP (Multiple Spanning Tree – 802.1s)**
* **SPB (Shortest Path Bridging – 802.1aq)**

### Link Aggregation & Redundancy

* **LACP / 802.1AX (Link Aggregation)**
* **Bonding (Linux bonding driver)**
* **VRRP** (Often treated L3/L2 hybrid)

### Wireless L2 Protocols

* **Wi-Fi (IEEE 802.11 a/b/g/n/ac/ax)**
* **WPA/WPA2/WPA3 (Security at L2)**
* **EAPOL (Authentication frames)**

### Control Protocols (L2 Signaling)

* **ARP (Address Resolution Protocol)**
* **NDP (Neighbor Discovery for IPv6)**
* **LLDP (Link Layer Discovery Protocol)**
* **CDP (Cisco Discovery Protocol)**
* **L2CP (Link Layer Control Protocol)**
* **PAUSE Frames (802.3x Flow Control)**

### L2 Tunneling / Encapsulation

* **L2TP (Layer 2 Tunneling Protocol)**
* **PPPoE (Point-to-Point Protocol over Ethernet)**
* **VXLAN** (Encapsulates L2 but runs over L4)
* **GENEVE** (L2 over UDP)
* **NVGRE** (L2 over GRE)
* **MACsec (802.1AE)**

### Bridging & Virtualization Protocols

* **Linux Bridge** (Switching at L2 in kernel)
* **Open vSwitch (OVS)**
* **FDB (Forwarding Database)**
* **EVPN (MAC learning & control-plane for overlays)**

### WAN / Telco L2 Protocols

* **ATM (Asynchronous Transfer Mode)**
* **Frame Relay**
* **MPLS-TP (Transport Profile)**
* **HDLC**
* **PPP (Point-to-Point Protocol)**
* **Metro Ethernet E-LINE/E-LAN services**

### Mermaid Diagram (L2 Protocol Groups)

```mermaid
flowchart LR
    Ethernet[Ethernet/VLAN/Q-in-Q]
    STP[STP/RSTP/MSTP]
    Wifi[802.11/WPA]
    Control[ARP/NDP/LLDP]
    Tunnel[L2TP/PPPoE/VXLAN/GENEVE]
    Redundancy[LACP/Bonding/MACsec]
    Bridge[Linux Bridge/OVS]

    Ethernet --> Bridge
    STP --> Bridge
    Tunnel --> Bridge
    Control --> Ethernet
    Wifi --> Ethernet
```

If you want, I can also list **which Linux kernel modules implement each protocol** or explain **L2 forwarding path inside the kernel (bridge + FDB lookup)**.

