Here’s a **comprehensive list of the most important default port numbers** used by common internet and enterprise protocols — grouped by category for clarity.

---

## 🌐 **Web and Internet Protocols**

| Protocol       | Port                         | Transport | Description                           |
| -------------- | ---------------------------- | --------- | ------------------------------------- |
| HTTP           | **80**                       | TCP       | Standard web traffic                  |
| HTTPS          | **443**                      | TCP       | Secure web traffic (TLS/SSL)          |
| HTTP Alternate | **8080**, **8000**, **8888** | TCP       | Used for proxies, development servers |
| WebSocket      | **80 / 443**                 | TCP       | Same as HTTP(S), upgraded connection  |
| QUIC / HTTP/3  | **443**                      | UDP       | Modern transport for HTTP over QUIC   |

---

## 📧 **Email Protocols**

| Protocol                      | Port    | Transport | Description                                |
| ----------------------------- | ------- | --------- | ------------------------------------------ |
| SMTP (Send Mail)              | **25**  | TCP       | Mail transfer between servers              |
| SMTPS (Secure SMTP)           | **465** | TCP       | Encrypted SMTP (deprecated but still used) |
| Submission (Mail client send) | **587** | TCP       | Mail submission with authentication        |
| POP3 (Receive Mail)           | **110** | TCP       | Retrieve mail from server                  |
| POP3S (Secure POP3)           | **995** | TCP       | POP3 over SSL/TLS                          |
| IMAP (Receive Mail)           | **143** | TCP       | Mail retrieval with folders                |
| IMAPS (Secure IMAP)           | **993** | TCP       | IMAP over SSL/TLS                          |

---

## 🖧 **Remote Access & Management**

| Protocol  | Port          | Transport | Description                        |
| --------- | ------------- | --------- | ---------------------------------- |
| SSH       | **22**        | TCP       | Secure shell remote login          |
| Telnet    | **23**        | TCP       | Unencrypted remote login           |
| RDP       | **3389**      | TCP       | Remote Desktop Protocol (Windows)  |
| VNC       | **5900–5901** | TCP       | Remote desktop sharing             |
| SNMP      | **161**       | UDP       | Simple Network Management Protocol |
| SNMP Trap | **162**       | UDP       | Receives SNMP notifications        |
| LDAP      | **389**       | TCP/UDP   | Directory services                 |
| LDAPS     | **636**       | TCP       | Secure LDAP (SSL/TLS)              |

---

## 📁 **File Transfer**

| Protocol          | Port     | Transport | Description                    |
| ----------------- | -------- | --------- | ------------------------------ |
| FTP (Command)     | **21**   | TCP       | File Transfer Protocol control |
| FTP (Data)        | **20**   | TCP       | Data channel for FTP           |
| FTPS (Secure FTP) | **990**  | TCP       | FTP over SSL/TLS               |
| SFTP              | **22**   | TCP       | File transfer over SSH         |
| TFTP              | **69**   | UDP       | Trivial File Transfer Protocol |
| SMB / CIFS        | **445**  | TCP       | Windows file sharing           |
| NFS               | **2049** | TCP/UDP   | Network File System            |

---

## 🌍 **DNS and Networking**

| Protocol               | Port                           | Transport | Description                  |
| ---------------------- | ------------------------------ | --------- | ---------------------------- |
| DNS                    | **53**                         | TCP/UDP   | Domain Name System           |
| DHCP (Server → Client) | **67**                         | UDP       | IP address assignment        |
| DHCP (Client → Server) | **68**                         | UDP       | DHCP requests                |
| NTP                    | **123**                        | UDP       | Network Time Protocol        |
| ICMP                   | *No port (uses IP protocol 1)* | —         | Ping and network diagnostics |

---

## 💬 **Messaging and Real-Time Communication**

| Protocol      | Port            | Transport | Description                          |
| ------------- | --------------- | --------- | ------------------------------------ |
| IRC           | **6667**        | TCP       | Internet Relay Chat                  |
| XMPP / Jabber | **5222**        | TCP       | Instant messaging                    |
| SIP           | **5060**        | TCP/UDP   | VoIP signaling                       |
| SIPS          | **5061**        | TCP       | Secure SIP over TLS                  |
| RTP           | **16384–32767** | UDP       | Real-time media streams (VoIP/video) |

---

## 💾 **Databases**

| Protocol             | Port      | Transport | Description                    |
| -------------------- | --------- | --------- | ------------------------------ |
| MySQL                | **3306**  | TCP       | MySQL database server          |
| PostgreSQL           | **5432**  | TCP       | PostgreSQL database server     |
| Oracle DB            | **1521**  | TCP       | Oracle listener service        |
| Microsoft SQL Server | **1433**  | TCP       | SQL Server database            |
| MongoDB              | **27017** | TCP       | MongoDB                        |
| Redis                | **6379**  | TCP       | In-memory key-value store      |
| Cassandra            | **9042**  | TCP       | Cassandra CQL native transport |
| Elasticsearch        | **9200**  | TCP       | REST API endpoint              |
| Elasticsearch (Node) | **9300**  | TCP       | Cluster communication          |

---

## 🧩 **Application and Infrastructure**

| Protocol              | Port          | Transport | Description             |
| --------------------- | ------------- | --------- | ----------------------- |
| Docker Daemon         | **2375**      | TCP       | Docker API (insecure)   |
| Docker Daemon (TLS)   | **2376**      | TCP       | Docker API (secure)     |
| Kubernetes API Server | **6443**      | TCP       | K8s control plane API   |
| ETCD                  | **2379–2380** | TCP       | K8s cluster store       |
| Prometheus            | **9090**      | TCP       | Monitoring metrics      |
| Grafana               | **3000**      | TCP       | Visualization dashboard |
| Jenkins               | **8080**      | TCP       | CI/CD web interface     |

---

## 🔒 **Security / VPN**

| Protocol | Port              | Transport | Description              |
| -------- | ----------------- | --------- | ------------------------ |
| OpenVPN  | **1194**          | UDP       | VPN tunnel               |
| IPSec    | **500**, **4500** | UDP       | VPN key exchange         |
| L2TP     | **1701**          | UDP       | Layer 2 Tunneling        |
| PPTP     | **1723**          | TCP       | Point-to-Point Tunneling |
| Kerberos | **88**            | TCP/UDP   | Authentication protocol  |

---

## ⚙️ **IoT, Streaming, and Miscellaneous**

| Protocol | Port     | Transport | Description                          |
| -------- | -------- | --------- | ------------------------------------ |
| MQTT     | **1883** | TCP       | Lightweight IoT messaging            |
| MQTTS    | **8883** | TCP       | Secure MQTT                          |
| RTMP     | **1935** | TCP       | Real-Time Media Streaming            |
| CoAP     | **5683** | UDP       | IoT protocol for constrained devices |

---

### 🧠 Key Notes:

* **TCP vs UDP:**
  TCP is reliable and connection-oriented; UDP is faster but unreliable.
* **Ephemeral Ports:**
  Clients usually connect using **49152–65535** (dynamic) range.
* **Well-known Ports:**
  **0–1023** are assigned by IANA for specific protocols.
* **Registered Ports:**
  **1024–49151** for applications.
* **Dynamic Ports:**
  **49152–65535** for temporary client connections.

---

Would you like me to include a **Mermaid diagram** that shows how ports are used across **client–server communication** (e.g., browser connecting to web server, DNS lookup, etc.)?
