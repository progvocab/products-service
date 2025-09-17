Let's build a complete example of the **Saga Pattern (Choreography-based)** using **Spring Boot** with **Kafka** for two microservices:

---

## 🧱 Use Case

> When a **Department is deleted**, all **Employees in that department** must also be deleted.
> If **employee deletion fails**, the department should **not be deleted**.

---

## 🧰 Stack

* Spring Boot (3.x)
* Spring Kafka
* Apache Kafka (via Docker)
* Two microservices:

  * `department-service`
  * `employee-service`
* Kafka topics:

  * `department.delete.request`
  * `employee.delete.success`
  * `employee.delete.failed`

---

## 📦 Project Structure

```
saga-choreography/
├── kafka/
│   └── docker-compose.yml
├── department-service/
│   └── ...
├── employee-service/
│   └── ...
```

---

## 🔁 Kafka Docker Setup (shared)

📄 `kafka/docker-compose.yml`

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
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

Run Kafka:

```bash
cd kafka
docker-compose up -d
```

---

## 📘 department-service

**Kafka Topics**:

* Publish: `department.delete.request`
* Listen: `employee.delete.success`, `employee.delete.failed`

### 🔧 application.yml

```yaml
spring:
  application.name: department-service
  kafka.bootstrap-servers: localhost:9092
```

### ✅ REST Controller

```java
@RestController
@RequestMapping("/departments")
@RequiredArgsConstructor
public class DepartmentController {
    private final KafkaTemplate<String, String> kafkaTemplate;

    @DeleteMapping("/{id}")
    public ResponseEntity<String> delete(@PathVariable Long id) {
        kafkaTemplate.send("department.delete.request", id.toString());
        return ResponseEntity.accepted().body("Delete requested for department " + id);
    }
}
```

### ✅ Kafka Listener

```java
@KafkaListener(topics = "employee.delete.success")
public void handleEmployeeDeletionSuccess(String departmentId) {
    System.out.println("Employee deleted. Proceeding to delete department " + departmentId);
    // departmentRepository.deleteById(Long.parseLong(departmentId));
}

@KafkaListener(topics = "employee.delete.failed")
public void handleEmployeeDeletionFailure(String departmentId) {
    System.out.println("Employee deletion failed. Aborting department delete " + departmentId);
}
```

---

## 📙 employee-service

**Kafka Topics**:

* Listen: `department.delete.request`
* Publish: `employee.delete.success`, `employee.delete.failed`

### 🔧 application.yml

```yaml
spring:
  application.name: employee-service
  kafka.bootstrap-servers: localhost:9092
```

### ✅ Kafka Listener

```java
@KafkaListener(topics = "department.delete.request")
public void onDepartmentDelete(String departmentId) {
    try {
        // Simulate employee deletion
        System.out.println("Deleting employees for department " + departmentId);
        // employeeRepository.deleteByDepartmentId(Long.parseLong(departmentId));
        // simulate delay
        Thread.sleep(1000);
        kafkaTemplate.send("employee.delete.success", departmentId);
    } catch (Exception e) {
        kafkaTemplate.send("employee.delete.failed", departmentId);
    }
}
```

---

## 🧪 How Saga Works

1. `DELETE /departments/{id}` hits department-service
2. department-service sends event to: `department.delete.request`
3. employee-service consumes event, deletes employees, and:

   * sends `employee.delete.success` if OK
   * sends `employee.delete.failed` if failed
4. department-service listens and deletes department only if success

---

## ✅ Topics Required

Create topics manually (or configure auto-creation):

```bash
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic department.delete.request --partitions 1 --replication-factor 1
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic employee.delete.success --partitions 1 --replication-factor 1
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic employee.delete.failed --partitions 1 --replication-factor 1
```

---

## 🧠 Benefits of Choreography Saga

* No central orchestrator
* Services are decoupled and react to events
* Scales well in event-driven architectures

---

Would you like:

* A GitHub repo ZIP with full working code?
* Add a third microservice for cascading deletes?
* Use JSON DTOs instead of string messages on Kafka?

Let me know!
