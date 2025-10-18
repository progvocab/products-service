Excellent — the **leaky bucket** is a core concept in **network rate limiting** and **traffic shaping** (also used in distributed systems and API gateways like NGINX, Kong, or Spring Cloud Gateway).

Let’s break it down clearly 👇

---

## 🧠 Concept

The **leaky bucket** is an **algorithm that controls the rate** at which data packets (or API requests) are processed.

It imagines a **bucket with a small hole at the bottom**:

* **Water poured into the bucket** = incoming requests or packets.
* **Water leaking out at a fixed rate** = processing or sending requests.
* **Overflow (when bucket is full)** = excess traffic is dropped or delayed.

---

## ⚙️ How It Works

```
           +--------------------+
           |                    |
Requests → |   Leaky Bucket     | →  Outflow (processed requests)
           | (Queue + Timer)    |
           |                    |
           +---------┬----------+
                     |
                Constant leak rate
```

### Steps:

1. Each incoming request adds a “drop” into the bucket.
2. The bucket leaks drops at a **constant rate**, meaning the system processes requests steadily.
3. If the bucket is full (capacity exceeded), new requests are **dropped** (or sometimes queued/delayed).

---

## 📘 Example

Suppose:

* Bucket capacity = 10 requests
* Leak rate = 1 request per second

| Time (s) | Requests In | Requests Out | Bucket Level   | Action       |
| -------- | ----------- | ------------ | -------------- | ------------ |
| 0        | 5           | 0            | 5              | OK           |
| 1        | 2           | 1            | 6              | OK           |
| 2        | 6           | 1            | 10             | OK           |
| 3        | 4           | 1            | 10+ (overflow) | Drop extra 3 |

So even if bursts arrive, output remains steady (1 request/sec).

---

## 🧩 Real-World Uses

| Context                 | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| **API Gateway / CDN**   | Rate limit users to prevent abuse          |
| **Network routers**     | Control packet flow and prevent congestion |
| **Distributed systems** | Smooth bursty workloads                    |
| **Microservices**       | Backpressure and fairness between clients  |

---

## 🔁 Related Algorithm: Token Bucket

The **token bucket** is similar but more flexible.

| Feature     | Leaky Bucket             | Token Bucket                            |
| ----------- | ------------------------ | --------------------------------------- |
| Output rate | Constant                 | Allows bursts                           |
| Buffering   | Excess requests dropped  | Requests can burst until tokens run out |
| Use case    | Traffic shaping (smooth) | Rate limiting with burst allowance      |

---

## 🧮 Pseudocode

```python
bucket_capacity = 10
leak_rate_per_sec = 1
water_level = 0

def request():
    global water_level
    if water_level < bucket_capacity:
        water_level += 1
        print("Request accepted")
    else:
        print("Request dropped")

def leak():
    global water_level
    if water_level > 0:
        water_level -= 1
```

You’d run `leak()` every second to simulate constant draining.

---

Would you like me to show **how to implement leaky bucket rate limiting** in a **Spring Boot Gateway filter or Redis-based distributed setup**? It’s a common interview + real-world question.
