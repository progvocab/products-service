### **Fail-Fast and Fail-Safe Design Patterns in Java**

These are **design behaviors** mostly related to **collections**, **concurrent programming**, and **system design principles**. They describe how a system or component reacts when a problem occurs.

---

### **1. Fail-Fast**

#### **Definition:**
A **fail-fast** system detects issues **immediately** when they occur and throws an exception rather than continuing with potentially corrupted data.

#### **Characteristics:**
- Detects errors **early**.
- Throws `ConcurrentModificationException` in collections.
- Typically used in **iterators** and **validators**.
- Helps in debugging and avoiding cascading failures.

#### **Examples in Java:**
```java
List<String> list = new ArrayList<>();
list.add("a");
Iterator<String> it = list.iterator();
list.add("b");   // Structural modification
it.next();       // Throws ConcurrentModificationException
```

#### **Usage Contexts:**
- Collection iterators (`ArrayList`, `HashMap`)
- Input validation systems
- Microservices checking preconditions

---

### **2. Fail-Safe**

#### **Definition:**
A **fail-safe** system **continues operation** even when errors occur. It avoids exceptions and provides a safe fallback behavior.

#### **Characteristics:**
- Tolerates failures gracefully.
- Usually works on a **copy** of data.
- Slower, but safer in concurrent environments.

#### **Examples in Java:**
```java
CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
list.add("a");
Iterator<String> it = list.iterator();
list.add("b");   // Allowed
it.next();       // No exception
```

#### **Usage Contexts:**
- Concurrent collections (`CopyOnWriteArrayList`, `ConcurrentHashMap`)
- High-availability systems
- Fault-tolerant APIs

---

### **Classification of Fail-Fast vs Fail-Safe**

| Criteria | Fail-Fast | Fail-Safe |
|---------|------------|-----------|
| **Detection Time** | Immediate | May delay or avoid detection |
| **Behavior on Error** | Throws exception | Continues silently |
| **Iterator Type** | Direct access to collection | Operates on a copy |
| **Concurrent Modification** | Not allowed | Allowed |
| **Performance** | Faster | Slightly slower |
| **Examples** | `ArrayList`, `HashMap` | `ConcurrentHashMap`, `CopyOnWriteArrayList` |

---

### **System Design Context**

- **Fail-Fast Systems**:
  - Used in **mission-critical systems** where bad data must be caught immediately.
  - Example: Transaction validators, financial applications.

- **Fail-Safe Systems**:
  - Used in **resilient systems** where **availability** is more important than correctness.
  - Example: Web services, distributed systems, circuit breakers.

---

Would you like a diagram or code examples for both patterns in a real-world context like an API validation or microservice system?