Below is a clear, concise explanation of the **ping command in Linux**, with **all headings as H3 only**, no emojis, and no extra line separators.

### What is the ping command

The **ping** command is a network diagnostic tool used to check whether a host (IP or domain) is reachable and how long packets take to travel from your machine to the destination and back.

### How ping works internally

1. It sends **ICMP Echo Request** packets to the target.
2. The target responds with **ICMP Echo Reply**.
3. Ping measures:

   * Packet loss
   * Round Trip Time (RTT)
   * Time-to-Live (TTL)

### Why ping is used

* To test network connectivity
* To diagnose packet loss
* To measure latency between two devices
* To verify DNS resolution

### Basic syntax

```
ping <hostname or IP>
```

### Common output fields

When you run:

```
ping google.com
```

You typically see:

* **64 bytes** → Size of response
* **icmp_seq=1** → Sequence number of packet
* **ttl=115** → Time-to-live left
* **time=21.5 ms** → Round-trip latency

Example:

```
64 bytes from 142.250.195.206: icmp_seq=1 ttl=115 time=21.5 ms
```

### Important options for ping

#### -c (count)

Sends a fixed number of packets.

```
ping -c 5 google.com
```

#### -i (interval)

Time between consecutive packets.

```
ping -i 2 google.com   # send every 2 seconds
```

#### -W (timeout for each reply)

Time to wait for a response.

```
ping -W 1 google.com   # 1 second
```

#### -s (packet size)

Set ICMP payload size.

```
ping -s 1024 google.com
```

#### -t (TTL value)

Set custom TTL.

```
ping -t 64 google.com
```

### Interpreting ping results

#### Successful pings

Shows replies and low latency.
Indicates the network path is working.

#### Request timed out

Indicates:

* Host unreachable
* Firewall blocking ICMP
* Network connectivity issue

#### High latency

Indicates congestion, routing delays, or slow network.

#### Packet loss

Even 1–2% loss can impact performance.
100% loss means no connection or ICMP blocked.

### Summary.statistics at end

After stopping ping with `Ctrl + C`, you get:

* **Packets transmitted**
* **Packets received**
* **Packet loss percentage**
* **Min/avg/max/stddev RTT**

Example:

```
5 packets transmitted, 5 received, 0% packet loss
rtt min/avg/max/mdev = 20.8/23.4/25.0/1.1 ms
```

If you want, I can also explain:

* Why ICMP may be blocked
* Difference between ping and traceroute
* How ping works at OSI layer level
