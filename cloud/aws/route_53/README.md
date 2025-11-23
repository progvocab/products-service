### Simple Routing

Sends traffic to **one single endpoint** (EC2, ALB, IP, etc.).
Used when you have **only one resource** serving the domain.

**Example**
`A → 3.110.22.5`
All clients always receive the same IP.

---

### Weighted Routing

Distributes traffic across multiple endpoints based on **weights**.
Used for **gradual deployments**, **blue-green**, or **A/B** testing.

**Example**

* Record 1: `A → 3.110.22.5 (weight 80)`
* Record 2: `A → 35.154.31.4 (weight 20)`
  80% traffic goes to server1, 20% to server2.

---

### Latency-Based Routing

Routes users to the region with the **lowest latency** based on AWS measurements.
Improves global performance.

**Example**
India user → ap-south-1
US user → us-east-1

---

### Geolocation Routing

Routes users based on **their physical location** (country, continent, or state).
Used for **regulatory** or **content localization**.

**Example**
USA users → `A → 54.23.x.x`
Europe users → `A → 18.211.x.x`

---

### Geoproximity Routing

Routes traffic based on a combination of **geographic location + bias (weight)**.
Requires **Route 53 Traffic Flow**.

**Example**
Increase bias for ap-south-1 by +20 → more traffic shifts to India region.

---

### Failover Routing

Supports **active-passive failover** using health checks.
If primary fails → traffic switches to secondary.

**Example**
Primary: ALB in us-east-1
Secondary: EC2 in us-west-2
Route 53 detects health failure → routes to backup.

---

### Multivalue Answer Routing

Returns **multiple healthy IP addresses** in response.
Traffic is distributed client-side (DNS resolver picks one).

**Example**
Returns 3 IPs:

* 3.110.22.5
* 3.110.22.6
* 3.110.22.7

If one IP is unhealthy, Route 53 removes it.

---

### IP-Based Routing

Routes DNS queries to specific **IP ranges / CIDR blocks** of clients.
Useful for **internal apps**, **enterprise networks**, or **custom routing needs**.

**Example**
Clients from `10.0.0.0/8` → Internal ALB
Public clients → Public ALB

