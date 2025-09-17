Complete architecture and code breakdown for implementing the **Saga Pattern using an Orchestrator** in Spring Boot:

---

## 🧱 Architecture Overview

```
Client
  ↓
Orchestrator Service (REST API + Saga Logic)
  ↓            ↓
Employee Service      Department Service
```

---

## 🎯 Use Case

> When a department is deleted, the orchestrator first deletes employees from `employee-service`, then deletes the department from `department-service`.
> If **employee deletion fails**, the orchestrator stops and can retry or log error.

---

## 📦 Project Structure

```
saga-orchestrator/
├── orchestrator-service/
├── department-service/
├── employee-service/
```

All services will communicate using **REST**, but the flow is managed centrally by the **orchestrator-service**.

---

## ✅ 1. `orchestrator-service`

### 🔧 `application.yml`

```yaml
spring:
  application.name: orchestrator-service
server:
  port: 8080
```

### 📄 `OrchestratorController.java`

```java
@RestController
@RequiredArgsConstructor
@RequestMapping("/saga")
public class OrchestratorController {

    private final RestTemplate restTemplate;

    @DeleteMapping("/department/{id}")
    public ResponseEntity<String> deleteDepartmentSaga(@PathVariable Long id) {
        try {
            // Step 1: Delete employees first
            ResponseEntity<Void> empRes = restTemplate.exchange(
                "http://localhost:8082/employees/department/" + id,
                HttpMethod.DELETE, null, Void.class
            );

            if (empRes.getStatusCode().is2xxSuccessful()) {
                // Step 2: Delete department
                ResponseEntity<Void> deptRes = restTemplate.exchange(
                    "http://localhost:8081/departments/" + id,
                    HttpMethod.DELETE, null, Void.class
                );

                if (deptRes.getStatusCode().is2xxSuccessful()) {
                    return ResponseEntity.ok("Saga complete: Department deleted");
                }
            }

            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Saga failed");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Saga exception: " + e.getMessage());
        }
    }
}
```

### ✅ Add Bean for `RestTemplate`

```java
@Configuration
public class AppConfig {
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

---

## ✅ 2. `department-service`

### 🔧 `application.yml`

```yaml
spring:
  application.name: department-service
server:
  port: 8081
```

### 📄 `DepartmentController.java`

```java
@RestController
@RequestMapping("/departments")
public class DepartmentController {

    @DeleteMapping("/{id}")
    public ResponseEntity<String> delete(@PathVariable Long id) {
        System.out.println("Department " + id + " deleted.");
        return ResponseEntity.ok("Department deleted");
    }
}
```

---

## ✅ 3. `employee-service`

### 🔧 `application.yml`

```yaml
spring:
  application.name: employee-service
server:
  port: 8082
```

### 📄 `EmployeeController.java`

```java
@RestController
@RequestMapping("/employees")
public class EmployeeController {

    @DeleteMapping("/department/{deptId}")
    public ResponseEntity<String> deleteEmployeesByDept(@PathVariable Long deptId) {
        System.out.println("Employees of department " + deptId + " deleted.");
        return ResponseEntity.ok("Employees deleted");
    }
}
```

---

## 🧪 How to Run

1. Start `employee-service` (port 8082)
2. Start `department-service` (port 8081)
3. Start `orchestrator-service` (port 8080)
4. Call:

```bash
curl -X DELETE http://localhost:8080/saga/department/10
```

✅ You’ll see logs:

```
Employees of department 10 deleted.
Department 10 deleted.
```

---

## 🧠 Summary

| Service                | Responsibility                                   |
| ---------------------- | ------------------------------------------------ |
| `orchestrator-service` | Orchestrates Saga steps                          |
| `employee-service`     | Deletes employees by department                  |
| `department-service`   | Deletes department if employee deletion succeeds |

---

convert the REST-based Saga Orchestrator to a **Kafka-based Orchestrator** using Spring Boot and Apache Kafka.

---

## 🧱 Architecture: Kafka-Based Saga with Orchestrator

```
Client
  ↓
Orchestrator Service (Kafka Producer/Consumer)
  ↓              ↓
Kafka Topics → [Saga Events] → Employee Service, Department Service
```

---

## 🔄 Event Flow for Deleting a Department

1. Orchestrator sends event: `saga.delete.department.start`
2. Employee service receives it and deletes employees.
3. On success → sends `saga.delete.employees.success`
4. Orchestrator receives success → sends `saga.delete.department.final`
5. Department service deletes the department.

---

## 🧰 Stack

* Spring Boot (3.x)
* Spring Kafka
* Apache Kafka (Docker)
* Services:

  * `orchestrator-service`
  * `employee-service`
  * `department-service`

---

## 🧪 Kafka Topics

| Topic                           | Publisher            | Listener             |
| ------------------------------- | -------------------- | -------------------- |
| `saga.delete.department.start`  | orchestrator-service | employee-service     |
| `saga.delete.employees.success` | employee-service     | orchestrator-service |
| `saga.delete.department.final`  | orchestrator-service | department-service   |

---

## ✅ 1. Orchestrator Service

### `application.yml`

```yaml
spring:
  kafka:
    bootstrap-servers: localhost:9092
```

### `SagaOrchestrator.java`

```java
@Component
@RequiredArgsConstructor
public class SagaOrchestrator {

    private final KafkaTemplate<String, String> kafkaTemplate;

    public void startDeleteSaga(Long departmentId) {
        kafkaTemplate.send("saga.delete.department.start", departmentId.toString());
        System.out.println("Saga started for department: " + departmentId);
    }

    @KafkaListener(topics = "saga.delete.employees.success")
    public void onEmployeeDeleteSuccess(String deptId) {
        System.out.println("Employee deletion confirmed. Sending final department delete for " + deptId);
        kafkaTemplate.send("saga.delete.department.final", deptId);
    }
}
```

### REST Trigger

```java
@RestController
@RequiredArgsConstructor
public class OrchestratorController {
    private final SagaOrchestrator orchestrator;

    @DeleteMapping("/saga/delete/department/{id}")
    public ResponseEntity<String> delete(@PathVariable Long id) {
        orchestrator.startDeleteSaga(id);
        return ResponseEntity.ok("Saga initiated for department: " + id);
    }
}
```

---

## ✅ 2. Employee Service

### `application.yml`

```yaml
spring:
  kafka:
    bootstrap-servers: localhost:9092
```

### `EmployeeSagaListener.java`

```java
@Component
@RequiredArgsConstructor
public class EmployeeSagaListener {

    private final KafkaTemplate<String, String> kafkaTemplate;

    @KafkaListener(topics = "saga.delete.department.start")
    public void onDepartmentDeleteRequest(String deptId) {
        System.out.println("Deleting employees for department " + deptId);
        // Simulate employee deletion...
        kafkaTemplate.send("saga.delete.employees.success", deptId);
    }
}
```

---

## ✅ 3. Department Service

### `DepartmentSagaListener.java`

```java
@Component
public class DepartmentSagaListener {

    @KafkaListener(topics = "saga.delete.department.final")
    public void onFinalDelete(String deptId) {
        System.out.println("Final deletion of department " + deptId);
        // departmentRepository.deleteById(Long.parseLong(deptId));
    }
}
```

---

## 🔄 Kafka Setup (Docker)

Same as before:

```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.2.1
    ports:
      - 2181:2181
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.2.1
    ports:
      - 9092:9092
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
```

---

## ✅ How to Test

1. Start all 3 services
2. Call:

   ```bash
   curl -X DELETE http://localhost:8080/saga/delete/department/10
   ```
3. Observe console logs across services:

   ```
   [orchestrator] Saga started for department 10
   [employee] Deleting employees for department 10
   [orchestrator] Employee deletion confirmed. Sending final delete
   [department] Final deletion of department 10
   ```

---

## 🧠 Summary

| Component          | Role                                   |
| ------------------ | -------------------------------------- |
| Orchestrator       | Initiates saga and listens for result  |
| Employee Service   | Deletes employees and notifies back    |
| Department Service | Final action if previous step succeeds |

---

Would you like a full ZIP or GitHub-style project for this Kafka-based Orchestrator version?

