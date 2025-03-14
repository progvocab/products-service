### **🔹 Saga vs Two-Phase Commit (2PC) vs Compensating Transactions**  

These are **transaction management strategies** used in **distributed systems** to ensure **data consistency** across multiple services or databases.

---

## **🔹 1. Two-Phase Commit (2PC)**
📌 **What is 2PC?**  
- A **strongly consistent** (ACID-compliant) transaction protocol.  
- **Coordinator-based approach**: A **single coordinator** controls all participating services.  
- Used in **relational databases (RDBMS)** or **monolithic applications**.

### **✅ Steps in 2PC**
1️⃣ **Prepare Phase:**  
   - Coordinator asks all services, "Can you commit?"  
   - Services **lock resources** and respond **"Yes" or "No"**.  
2️⃣ **Commit/Rollback Phase:**  
   - If **all services** reply **"Yes"**, the transaction **commits**.  
   - If **any service** replies **"No"**, all services **rollback**.

### **🚨 Problems with 2PC**
❌ **Blocking Issue:** Locks resources, leading to deadlocks.  
❌ **Single Point of Failure:** Coordinator crash can leave transactions in an inconsistent state.  
❌ **Not Scalable:** Poor performance in distributed systems.  

### **🚀 Best Use Cases**
✔️ **Banking transactions (SQL databases)**  
✔️ **Distributed relational databases** (e.g., PostgreSQL, MySQL, Oracle)  

---

## **🔹 2. Saga Pattern**
📌 **What is a Saga?**  
- A **long-lived transaction** split into **multiple steps** across services.  
- Uses **event-driven choreography** or **orchestration** for coordination.  
- **Compensating transactions** are used to undo failed steps.

### **✅ Two Types of Sagas**
1️⃣ **Choreography-based Saga** (Event-Driven)  
   - Each service **reacts to events** without a central controller.  
   - Example: **Order Service → Payment Service → Inventory Service**  
   - If **Inventory fails**, **Payment Service compensates (refunds money)**.  

2️⃣ **Orchestration-based Saga** (Central Controller)  
   - A **Saga Coordinator** manages transactions.  
   - Sends **commands** to services and **handles failures**.  
   - Example: **AWS Step Functions, Camunda BPM**.  

### **🚀 Best Use Cases**
✔️ **Microservices architectures**  
✔️ **E-commerce order processing**  
✔️ **Travel booking systems**  

### **🚨 Challenges**
❌ **Eventual Consistency:** No immediate ACID guarantee.  
❌ **Complex Debugging:** Hard to trace failures.  

---

## **🔹 3. Compensating Transactions**
📌 **What is a Compensating Transaction?**  
- A **manual rollback mechanism** for distributed transactions.  
- Used when **a previous operation fails**, requiring a **reverse action**.  
- **Compensates for partially completed transactions**.  

### **✅ Example**
- **Hotel booking system**:  
  1. **Step 1**: User books a hotel.  
  2. **Step 2**: Flight booking fails.  
  3. **Step 3**: Compensating transaction cancels the hotel booking.  

### **🚀 Best Use Cases**
✔️ **Microservices where transactions cannot be locked**  
✔️ **Payment and refund systems**  
✔️ **Travel and airline reservations**  

### **🚨 Challenges**
❌ **Difficult to Implement**: Requires handling failure scenarios.  
❌ **May Not Be Feasible**: Some operations **cannot be undone** (e.g., sending an email).  

---

## **🔹 Key Differences**
| Feature            | Two-Phase Commit (2PC)  | Saga Pattern  | Compensating Transactions  |
|--------------------|------------------------|--------------|---------------------------|
| **Consistency**   | Strong (ACID)           | Eventual     | Eventual                  |
| **Scalability**   | Poor                    | High        | High                      |
| **Failure Handling** | Rollback (locks all) | Undo previous step | Manual rollback logic  |
| **Performance**   | Slow (locks resources)  | Fast        | Medium                    |
| **Use Case**      | Banking, SQL databases  | Microservices, E-commerce | Payment rollbacks, Booking systems |

---

## **🔹 Summary**
✅ **Use 2PC** when **strong consistency** is required (SQL transactions).  
✅ **Use Saga** when **scalability is important** (microservices).  
✅ **Use Compensating Transactions** when you **need manual rollback logic**.  

Would you like an **implementation guide for Saga in Spring Boot with Kafka?** 🚀

### **🔹 Code Examples & Real-Time Scenarios for 2PC, Saga, and Compensating Transactions**  

Let’s explore how **Two-Phase Commit (2PC)**, **Saga Pattern**, and **Compensating Transactions** work with **real-world scenarios and code examples**.

---

## **🔹 1. Two-Phase Commit (2PC) - Banking Transaction (SQL Databases)**
📌 **Scenario**: Transferring money between two bank accounts (**Account A → Account B**).  
📌 **Technology**: PostgreSQL with Java (Spring Boot + JDBC).  

### **✅ Code Example**
```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class TwoPhaseCommitExample {
    public static void main(String[] args) throws SQLException {
        Connection connA = DriverManager.getConnection("jdbc:postgresql://bank-db/accountA", "user", "pass");
        Connection connB = DriverManager.getConnection("jdbc:postgresql://bank-db/accountB", "user", "pass");

        try {
            connA.setAutoCommit(false);
            connB.setAutoCommit(false);

            // Step 1: Deduct amount from Account A
            connA.createStatement().executeUpdate("UPDATE accounts SET balance = balance - 100 WHERE id = 1");

            // Step 2: Add amount to Account B
            connB.createStatement().executeUpdate("UPDATE accounts SET balance = balance + 100 WHERE id = 2");

            // Step 3: Prepare Phase - Ensure both updates are valid
            connA.commit();
            connB.commit(); // Commit Phase

        } catch (Exception e) {
            connA.rollback();
            connB.rollback(); // Rollback if any step fails
            System.out.println("Transaction failed: " + e.getMessage());
        } finally {
            connA.close();
            connB.close();
        }
    }
}
```
### **🚨 Issues with 2PC**
❌ **Blocking issue** (locks both accounts).  
❌ **Single point of failure** (if DB crashes, transactions may hang).  
❌ **Not scalable** for distributed systems.  

---

## **🔹 2. Saga Pattern - E-Commerce Order Processing (Microservices)**
📌 **Scenario**: Customer places an **order**, payment is processed, and inventory is updated. If **inventory fails**, the payment is **refunded**.  
📌 **Technology**: Spring Boot + Kafka (Event-Based Saga).  

### **✅ Choreography-Based Saga (Event-Driven)**
#### **1️⃣ Order Service**
```java
@RestController
@RequestMapping("/orders")
public class OrderController {
    @Autowired private KafkaTemplate<String, String> kafkaTemplate;

    @PostMapping
    public String createOrder(@RequestBody Order order) {
        // Step 1: Save order
        orderRepository.save(order);

        // Step 2: Publish event for payment processing
        kafkaTemplate.send("order-topic", "Order Created: " + order.getId());

        return "Order Placed Successfully!";
    }
}
```

#### **2️⃣ Payment Service**
```java
@KafkaListener(topics = "order-topic", groupId = "payment-group")
public void processPayment(String message) {
    System.out.println("Processing Payment for: " + message);

    // Step 3: Process payment and publish inventory event
    kafkaTemplate.send("inventory-topic", "Payment Processed: " + message);
}
```

#### **3️⃣ Inventory Service**
```java
@KafkaListener(topics = "inventory-topic", groupId = "inventory-group")
public void updateInventory(String message) {
    System.out.println("Updating Inventory for: " + message);

    // Step 4: If inventory update fails, publish rollback event
    boolean inventorySuccess = false;  // Simulating failure
    if (!inventorySuccess) {
        kafkaTemplate.send("rollback-topic", "Inventory Failure: " + message);
    }
}
```

#### **4️⃣ Compensating Transaction - Refund Payment**
```java
@KafkaListener(topics = "rollback-topic", groupId = "payment-group")
public void refundPayment(String message) {
    System.out.println("Refunding Payment for: " + message);
}
```

### **🚀 Why is Saga Better?**
✅ **Asynchronous** (no blocking).  
✅ **Failure handling** via compensation (refund).  
✅ **Highly scalable** for microservices.  

---

## **🔹 3. Compensating Transactions - Travel Booking System**
📌 **Scenario**: Booking a flight + hotel. If **hotel booking fails**, the **flight is canceled** as a compensating transaction.  
📌 **Technology**: Spring Boot + REST API.  

### **✅ Booking Service**
```java
@RestController
@RequestMapping("/booking")
public class BookingController {
    @Autowired private FlightService flightService;
    @Autowired private HotelService hotelService;

    @PostMapping
    public String bookTrip(@RequestBody Trip trip) {
        boolean flightBooked = flightService.bookFlight(trip.getFlightId());

        if (!flightBooked) {
            return "Flight booking failed!";
        }

        boolean hotelBooked = hotelService.bookHotel(trip.getHotelId());

        if (!hotelBooked) {
            flightService.cancelFlight(trip.getFlightId()); // Compensating Transaction
            return "Hotel booking failed, rolling back flight!";
        }

        return "Trip booked successfully!";
    }
}
```

### **✅ Flight Service**
```java
public boolean bookFlight(String flightId) {
    System.out.println("Flight booked: " + flightId);
    return true; // Simulate success
}

public void cancelFlight(String flightId) {
    System.out.println("Flight cancelled: " + flightId); // Compensating Transaction
}
```

### **✅ Hotel Service**
```java
public boolean bookHotel(String hotelId) {
    System.out.println("Hotel booked: " + hotelId);
    return false; // Simulating failure
}
```

### **🚀 Why Use Compensating Transactions?**
✅ **Flexible rollback logic** for business-specific needs.  
✅ **Works well with APIs & external services** (e.g., third-party bookings).  
✅ **Better than 2PC** in distributed systems (no global locks).  

---

## **🔹 Key Differences**
| Feature            | Two-Phase Commit (2PC) | Saga Pattern (Event-Based) | Compensating Transactions |
|--------------------|------------------------|----------------------------|---------------------------|
| **Consistency**   | Strong (ACID)           | Eventual (BASE)            | Eventual (Manual Rollback) |
| **Scalability**   | Low                      | High                        | Medium                     |
| **Failure Handling** | Rollback entire transaction | Compensating transactions | Custom compensation logic |
| **Performance**   | Slow (locks resources)   | Fast (async events)        | Medium (depends on rollback logic) |
| **Use Case**      | Banking, SQL transactions | Microservices, E-commerce | Booking systems, Payments |

---

## **🔹 Summary**
✅ **Use 2PC** when you need **strict consistency** (SQL-based banking).  
✅ **Use Saga** when working with **microservices** (scalable & event-driven).  
✅ **Use Compensating Transactions** when rollback logic **varies per use case** (e.g., travel booking).  

Would you like a **step-by-step tutorial on implementing Saga with Spring Boot & Kafka?** 🚀