### What is the nmap command

The **nmap** command (**Network Mapper**) is a powerful Linux tool used for **network discovery**, **port scanning**, **security auditing**, and **service detection**. It helps identify hosts, open ports, running services, operating systems, firewalls, and vulnerabilities.

### Why nmap is used

* To see what devices are active on a network
* To find open ports on a system
* To detect services running on each port
* To identify OS versions
* To check for firewall rules
* To perform security assessments
* To discover misconfigurations or unknown services

### Basic syntax

```
nmap <target>
```

Examples:

```
nmap 192.168.1.1
nmap google.com
```

### Key concepts in nmap

* **Host discovery** → Checks if host is online
* **Port scanning** → Discovers open/closed ports
* **Service detection** → Identifies services running on each port
* **OS detection** → Guesses OS based on TCP/IP stack behavior
* **Scripting engine (NSE)** → Allows vulnerability and security checks

### Common nmap scan types

#### Basic port scan

```
nmap 192.168.1.10
```

#### Scan a range of IPs

```
nmap 192.168.1.0/24
```

#### Scan specific ports

```
nmap -p 22,80,443 192.168.1.10
```

#### Scan all 65535 ports

```
nmap -p- 192.168.1.10
```

### Advanced and important options

#### Service version detection

Detects application version running on ports.

```
nmap -sV 192.168.1.10
```

#### OS detection

```
nmap -O 192.168.1.10
```

#### Aggressive scan (OS + version + script + traceroute)

```
nmap -A 192.168.1.10
```

Used for deep discovery.

#### Stealth (SYN) scan

Fast and common in security testing.

```
nmap -sS 192.168.1.10
```

#### UDP scan

```
nmap -sU 192.168.1.10
```

#### Scan without ping (for firewalled hosts)

```
nmap -Pn 192.168.1.10
```

### Nmap Scripting Engine (NSE)

Nmap includes a library of scripts to detect vulnerabilities, brute-force attacks, and gather detailed service info.

#### List scripts

```
ls /usr/share/nmap/scripts
```

#### Run all default scripts

```
nmap -sC 192.168.1.10
```

#### Run a specific script

```
nmap --script=http-title 192.168.1.10
```

#### Run vulnerability scripts

```
nmap --script vuln 192.168.1.10
```

### Example outputs and interpretation

#### Example: Simple scan

```
PORT     STATE SERVICE
22/tcp   open  ssh
80/tcp   open  http
443/tcp  open  https
```

Meaning:

* Target is online
* Ports 22, 80, 443 are open
* Services running on them are detected

#### Example: OS detection

```
OS details: Linux 3.x kernel
```

#### Example: Service version detection

```
80/tcp open  http  Apache 2.4.41
```

### Important Nmap scan modes

| Mode              | Command | What it does                        |
| ----------------- | ------- | ----------------------------------- |
| TCP Connect scan  | `-sT`   | Full connection (slow but accurate) |
| SYN scan          | `-sS`   | Stealth scan                        |
| UDP scan          | `-sU`   | Finds UDP ports                     |
| Aggressive        | `-A`    | OS + version + script + traceroute  |
| No ping           | `-Pn`   | Scan even when host drops ICMP      |
| Version detection | `-sV`   | Detect service versions             |

### When nmap is useful

* Testing firewall rules
* Penetration testing
* Network inventory
* Checking server exposure before deployment
* Identifying vulnerable services
* Network troubleshooting
* Checking open ports on cloud VMs

If you want, I can also explain:

* How nmap detects OS
* Difference between -sT vs -sS scanning
* How to write your own Nmap script (NSE)
