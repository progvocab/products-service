Here is a regenerated version of the **Saga Orchestrator microservice** using **Kafka-based choreography** to coordinate the following:

### ✅ Saga Flow with Kafka Topics

| Step | Action           | Service              | Kafka Topic              |
| ---- | ---------------- | -------------------- | ------------------------ |
| 1    | Start Order      | Orchestrator → Kafka | `order-created`          |
| 2    | Decrease Stock   | Product Service      | `stock-updated`          |
| 3    | Validate PIN     | Customer Service     | `pin-validated`          |
| 4    | Initiate Payment | Payment Service      | `payment-processed`      |
| 5    | Start Delivery   | Delivery Service     | `delivery-started`       |
| 6    | Finalize         | Orchestrator         | (aggregates final state) |

---

## 📦 Services Required

We'll generate only the **Saga Orchestrator**, but the downstream services listen to Kafka topics and emit their results to specific topics.

---

## ✅ Topics Overview

| Service          | Consumes from         | Produces to         |
| ---------------- | --------------------- | ------------------- |
| Orchestrator     | N/A (starts the flow) | `order-created`     |
| Product Service  | `order-created`       | `stock-updated`     |
| Customer Service | `stock-updated`       | `pin-validated`     |
| Payment Service  | `pin-validated`       | `payment-processed` |
| Delivery Service | `payment-processed`   | `delivery-started`  |
| Orchestrator     | `delivery-started`    | (final log only)    |

---

## ✅ Kafka Saga Orchestrator Microservice

### `pom.xml`

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.kafka</groupId>
        <artifactId>spring-kafka</artifactId>
    </dependency>
    <dependency>
        <groupId>com.fasterxml.jackson.core</groupId>
        <artifactId>jackson-databind</artifactId>
    </dependency>
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
</dependencies>
```

---

### `application.yml`

```yaml
server:
  port: 8085

spring:
  kafka:
    bootstrap-servers: localhost:9092
    consumer:
      group-id: saga-orchestrator-group
      auto-offset-reset: earliest
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.springframework.kafka.support.serializer.JsonSerializer
    consumer-properties:
      spring.json.trusted.packages: '*'
```

---

### `OrderEvent.java`

```java
package com.example.saga.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class OrderEvent {
    private String orderId;
    private String productId;
    private String customerId;
    private String deliveryPin;
    private double amount;
    private String idempotencyKey;
    private String status; // CREATED, STOCK_UPDATED, PIN_VALIDATED, PAID, DELIVERED
}
```

---

### `OrderController.java`

```java
package com.example.saga.controller;

import com.example.saga.dto.OrderEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/order")
@RequiredArgsConstructor
public class OrderController {

    private final KafkaTemplate<String, OrderEvent> kafkaTemplate;

    @PostMapping
    public String createOrder(@RequestBody OrderEvent request) {
        request.setOrderId(UUID.randomUUID().toString());
        request.setStatus("CREATED");

        kafkaTemplate.send("order-created", request);
        return "Order initiated with ID: " + request.getOrderId();
    }
}
```

---

### `DeliveryResultListener.java`

```java
package com.example.saga.listener;

import com.example.saga.dto.OrderEvent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class DeliveryResultListener {

    @KafkaListener(topics = "delivery-started", groupId = "saga-orchestrator-group")
    public void handleDelivery(OrderEvent event) {
        if ("DELIVERED".equals(event.getStatus())) {
            log.info("✅ Order {} fully completed. Final Saga Step.", event.getOrderId());
        } else {
            log.warn("⚠️ Delivery failed for Order {}", event.getOrderId());
        }
    }
}
```

---

### Kafka Flow (Downstream services should):

* **Product Service**:

  * Listen on `order-created`
  * Update stock
  * Publish to `stock-updated`

* **Customer Service**:

  * Listen on `stock-updated`
  * Validate delivery PIN
  * Publish to `pin-validated`

* **Payment Service**:

  * Listen on `pin-validated`
  * Process payment
  * Publish to `payment-processed`

* **Delivery Service**:

  * Listen on `payment-processed`
  * Start delivery
  * Publish to `delivery-started`

---

## ✅ Kafka Topics to Create

Use the CLI:

```bash
kafka-topics.sh --create --topic order-created --bootstrap-server localhost:9092
kafka-topics.sh --create --topic stock-updated --bootstrap-server localhost:9092
kafka-topics.sh --create --topic pin-validated --bootstrap-server localhost:9092
kafka-topics.sh --create --topic payment-processed --bootstrap-server localhost:9092
kafka-topics.sh --create --topic delivery-started --bootstrap-server localhost:9092
```

---

## ✅ Test it

```bash
curl -X POST http://localhost:8085/api/order \
  -H "Content-Type: application/json" \
  -d '{
    "productId": "P001",
    "customerId": "C001",
    "deliveryPin": "560001",
    "amount": 500.0,
    "idempotencyKey": "order-001"
}'
```

---

## ✅ Next Steps

Would you like me to:

* Generate Kafka listeners for **Product, Customer, Payment, and Delivery services**?
* Package this as a downloadable Spring Boot app?
* Add rollback support with compensating transactions?

Let me know how you want to expand this Saga Orchestrator.

