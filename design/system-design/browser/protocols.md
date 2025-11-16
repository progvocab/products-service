Layer 7 (Application Layer) protocols enable **end-user services**, **application messaging**, and **data exchange**. Here are the major ones grouped by category:

---

## ✅ **Common Layer 7 Protocols (Application Layer)**

### **Web & Browsing**

* **HTTP**
* **HTTPS**
* **HTTP/2**, **HTTP/3**
* **SPDY** (deprecated)

### **Email**

* **SMTP**
* **IMAP**
* **POP3**

### **File Transfer**

* **FTP**
* **SFTP**
* **FTPS**
* **TFTP**
* **NFS**
* **SMB/CIFS**

### **Remote Access**

* **SSH**
* **Telnet**
* **RDP** (Remote Desktop Protocol)
* **VNC**

### **Name Resolution**

* **DNS**
* **mDNS**

### **Messaging & Chat**

* **XMPP**
* **MQTT**
* **AMQP**
* **SIP** (for VoIP signaling)

### **Streaming / Real-Time**

* **RTSP**
* **RTP** (control plane handled at L7)

### **Directory & Authentication**

* **LDAP**
* **Kerberos**
* **OAuth / OpenID Connect**
* **SAML**

### **Network Management**

* **SNMP**
* **NETCONF**
* **RESTCONF**

### **Service Discovery**

* **DNS-SD**
* **SSDP** (UPnP)

### **IoT Protocols**

* **CoAP**
* **MQTT**
* **AMQP**

### **Database Protocols**

* **MySQL protocol**
* **PostgreSQL protocol**
* **MongoDB wire protocol**

---

 

Layer 7 protocols cover everything involving **application-level communication**, including **web**, **email**, **file transfer**, **auth**, **messaging**, and **database interactions**. They sit above the transport layer and define **how data is formatted, processed, and exchanged** for real-world applications.


Here are **3-line, crisp explanations** of the **top 10 Layer 7 protocols**:

---

### **1. HTTP**

HTTP is the foundational protocol for web communication, enabling clients to request resources from servers. It is stateless and text-based, making it simple and flexible. Most web APIs, browsers, and services depend on HTTP.

### **2. HTTPS**

HTTPS is HTTP secured with TLS/SSL encryption to protect data in transit. It ensures confidentiality, integrity, and authentication between client and server. Almost all modern web traffic uses HTTPS.

### **3. DNS**

DNS translates human-readable domain names into IP addresses. It is essential for routing traffic to the correct servers on the internet. DNS operates through hierarchical, distributed name servers.

### **4. SMTP**

SMTP is the standard protocol for sending emails between servers. It handles message transfer and routing but not retrieval. Most email delivery pipelines depend on SMTP for outgoing mail.

### **5. IMAP**

IMAP allows users to retrieve and manage emails from servers while keeping messages stored centrally. It supports folder management and multi-device sync. It is widely used in modern email clients.

### **6. FTP**

FTP transfers files between clients and servers using separate control and data channels. It is efficient but insecure without encryption (FTPS or SFTP preferred). Often used for backups, archives, and large file transfers.

### **7. SSH**

SSH provides secure remote login and command execution over an encrypted channel. It replaces insecure tools like Telnet and rsh. System administrators use SSH extensively for server management.

### **8. LDAP**

LDAP is a directory service protocol used for managing users, groups, and authentication data. Organizations rely on LDAP for centralized identity management. Systems like Active Directory use LDAP as a core access protocol.

### **9. SNMP**

SNMP monitors and manages network devices like routers, switches, and servers. It enables querying of metrics and pushing configuration changes. Network management tools rely heavily on SNMP polling and traps.

### **10. SIP**

SIP is the signaling protocol for initiating, modifying, and terminating VoIP calls. It handles call setup, ringing, registration, and presence. Most internet telephony systems (VoIP, video calls) use SIP as a control plane.

