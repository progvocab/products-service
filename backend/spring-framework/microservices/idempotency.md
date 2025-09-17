Here is the **updated Payment Service** with:

* ✅ **Idempotency ID support** (to prevent duplicate transactions)
* ✅ **PostgreSQL** configuration
* ✅ **All required `import` statements**
* ✅ **Entity + Service + Controller + DTOs + Repository**
* ✅ **New DB column: `idempotency_key` with unique constraint**

---

### ✅ `PaymentTransaction.java` (Entity)

```java
package com.example.paymentservice.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Data
@NoArgsConstructor
@Table(name = "payment_transaction", uniqueConstraints = {
        @UniqueConstraint(columnNames = "idempotencyKey")
})
public class PaymentTransaction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String transactionId;

    private String fromVpa;
    private String toVpa;
    private double amount;
    private String status;
    private LocalDateTime createdAt;

    @Column(nullable = false, unique = true)
    private String idempotencyKey;
}
```

---

### ✅ `PaymentTransactionRepository.java`

```java
package com.example.paymentservice.repository;

import com.example.paymentservice.entity.PaymentTransaction;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface PaymentTransactionRepository extends JpaRepository<PaymentTransaction, Long> {
    Optional<PaymentTransaction> findByTransactionId(String transactionId);
    Optional<PaymentTransaction> findByIdempotencyKey(String idempotencyKey);
}
```

---

### ✅ `PaymentRequest.java`

```java
package com.example.paymentservice.dto;

import lombok.Data;

@Data
public class PaymentRequest {
    private String fromVpa;
    private String toVpa;
    private double amount;
    private String purpose;
    private String remarks;
    private String idempotencyKey;
}
```

---

### ✅ `PaymentResponse.java`

```java
package com.example.paymentservice.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class PaymentResponse {
    private String transactionId;
    private String status;
    private double amount;
}
```

---

### ✅ `PaymentStatusResponse.java`

```java
package com.example.paymentservice.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
public class PaymentStatusResponse {
    private String transactionId;
    private String status;
    private double amount;
    private LocalDateTime timestamp;
}
```

---

### ✅ `PaymentService.java`

```java
package com.example.paymentservice.service;

import com.example.paymentservice.dto.PaymentRequest;
import com.example.paymentservice.dto.PaymentResponse;
import com.example.paymentservice.dto.PaymentStatusResponse;
import com.example.paymentservice.entity.PaymentTransaction;
import com.example.paymentservice.repository.PaymentTransactionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class PaymentService {

    private final PaymentTransactionRepository repository;

    public PaymentResponse initiatePayment(PaymentRequest request) {
        return repository.findByIdempotencyKey(request.getIdempotencyKey())
                .map(existing -> new PaymentResponse(existing.getTransactionId(), existing.getStatus(), existing.getAmount()))
                .orElseGet(() -> {
                    PaymentTransaction txn = new PaymentTransaction();
                    txn.setTransactionId(UUID.randomUUID().toString());
                    txn.setFromVpa(request.getFromVpa());
                    txn.setToVpa(request.getToVpa());
                    txn.setAmount(request.getAmount());
                    txn.setStatus("PENDING");
                    txn.setCreatedAt(LocalDateTime.now());
                    txn.setIdempotencyKey(request.getIdempotencyKey());

                    repository.save(txn);
                    return new PaymentResponse(txn.getTransactionId(), txn.getStatus(), txn.getAmount());
                });
    }

    public PaymentStatusResponse getStatus(String transactionId) {
        PaymentTransaction txn = repository.findByTransactionId(transactionId)
                .orElseThrow(() -> new RuntimeException("Transaction not found"));

        return new PaymentStatusResponse(txn.getTransactionId(), txn.getStatus(), txn.getAmount(), txn.getCreatedAt());
    }
}
```

---

### ✅ `PaymentController.java`

```java
package com.example.paymentservice.controller;

import com.example.paymentservice.dto.PaymentRequest;
import com.example.paymentservice.dto.PaymentResponse;
import com.example.paymentservice.dto.PaymentStatusResponse;
import com.example.paymentservice.service.PaymentService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/payment")
@RequiredArgsConstructor
public class PaymentController {

    private final PaymentService paymentService;

    @PostMapping("/initiate")
    public ResponseEntity<PaymentResponse> initiatePayment(@RequestBody PaymentRequest request) {
        return ResponseEntity.ok(paymentService.initiatePayment(request));
    }

    @GetMapping("/status/{transactionId}")
    public ResponseEntity<PaymentStatusResponse> getStatus(@PathVariable String transactionId) {
        return ResponseEntity.ok(paymentService.getStatus(transactionId));
    }
}
```

---

### ✅ `application.yml` (PostgreSQL)

```yaml
server:
  port: 8082

spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/payment_db
    username: postgres
    password: password
    driver-class-name: org.postgresql.Driver

  jpa:
    hibernate:
      ddl-auto: update
    show-sql: true
    properties:
      hibernate:
        format_sql: true
```

---

### ✅ PostgreSQL DDL (for manual schema setup)

```sql
CREATE TABLE payment_transaction (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    from_vpa VARCHAR(100),
    to_vpa VARCHAR(100),
    amount NUMERIC(10, 2),
    status VARCHAR(20),
    created_at TIMESTAMP,
    idempotency_key VARCHAR(100) UNIQUE NOT NULL
);
```

---

Would you like:

* Docker + Docker Compose setup for PostgreSQL?
* Swagger/OpenAPI documentation?
* Integration with UPI or notification service?

Let me know!

