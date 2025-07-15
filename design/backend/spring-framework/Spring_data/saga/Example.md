Basic implementation of the **Saga Pattern** using **Spring Boot**, with two microservices: `employee-service` and `department-service`. We’ll demonstrate a **delete operation** where deleting a department triggers the deletion of related employees using a Saga orchestration approach.

---

### ⚙️ Tech Stack:

* Spring Boot
* Spring Web
* Spring Data JPA
* H2/PostgreSQL (for demo)
* Spring Cloud OpenFeign (for service communication)
* Saga Orchestration (Manual or with State Machine)

---

## 🧱 Microservices Overview

### 1. `department-service`

* REST endpoint to delete a department
* Orchestrator initiates Saga by calling `employee-service`

### 2. `employee-service`

* Deletes all employees related to the department
* Supports compensation if rollback is needed (optional extension)

---

## 📁 Structure

```
saga-pattern/
├── employee-service/
└── department-service/
```

---

### 1️⃣ Department Service

**`pom.xml`**

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-openfeign</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    <dependency>
        <groupId>com.h2database</groupId>
        <artifactId>h2</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

**`Department.java`**

```java
@Entity
public class Department {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
}
```

**`DepartmentRepository.java`**

```java
public interface DepartmentRepository extends JpaRepository<Department, Long> {}
```

**Feign Client: `EmployeeClient.java`**

```java
@FeignClient(name = "employee-service", url = "http://localhost:8081")
public interface EmployeeClient {
    @DeleteMapping("/employees/by-department/{deptId}")
    void deleteEmployeesByDepartment(@PathVariable Long deptId);
}
```

**`DepartmentController.java`**

```java
@RestController
@RequestMapping("/departments")
@RequiredArgsConstructor
public class DepartmentController {
    private final DepartmentRepository departmentRepository;
    private final EmployeeClient employeeClient;

    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteDepartment(@PathVariable Long id) {
        if (!departmentRepository.existsById(id)) {
            return ResponseEntity.notFound().build();
        }

        // Start Saga
        try {
            employeeClient.deleteEmployeesByDepartment(id);
            departmentRepository.deleteById(id);
            return ResponseEntity.ok("Department and related employees deleted");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Saga failed: " + e.getMessage());
        }
    }
}
```

**`application.properties`**

```properties
server.port=8080
spring.h2.console.enabled=true
spring.datasource.url=jdbc:h2:mem:deptdb
spring.jpa.hibernate.ddl-auto=update
```

---

### 2️⃣ Employee Service

**`pom.xml`**

Same as above (excluding OpenFeign).

**`Employee.java`**

```java
@Entity
public class Employee {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private Long departmentId;
}
```

**`EmployeeRepository.java`**

```java
public interface EmployeeRepository extends JpaRepository<Employee, Long> {
    void deleteByDepartmentId(Long departmentId);
}
```

**`EmployeeController.java`**

```java
@RestController
@RequestMapping("/employees")
@RequiredArgsConstructor
public class EmployeeController {
    private final EmployeeRepository employeeRepository;

    @DeleteMapping("/by-department/{deptId}")
    public ResponseEntity<Void> deleteByDepartment(@PathVariable Long deptId) {
        employeeRepository.deleteByDepartmentId(deptId);
        return ResponseEntity.ok().build();
    }
}
```

**`application.properties`**

```properties
server.port=8081
spring.h2.console.enabled=true
spring.datasource.url=jdbc:h2:mem:empdb
spring.jpa.hibernate.ddl-auto=update
```

---

## 🧪 Testing the Saga

1. Start both services.
2. Create some departments and employees (via SQL or REST).
3. Call: `DELETE /departments/{id}`.
4. It deletes employees first → then the department → Saga complete.
5. If employee deletion fails, department is not deleted.

---

## 🔄 Optional Enhancements

* Add **Kafka-based** event communication for a **Choreography Saga**.
* Include **compensation transactions** for rollback.
* Use **State Machine / Temporal** for complex orchestration.

---

Would you like:

* Postman collection or curl test examples?
* To use PostgreSQL instead of H2?
* To demonstrate rollback with a compensation method?

Let me know, and I’ll extend this accordingly.



---

## 🧱 1. `department-service`

### `Department.java`

```java
package com.example.departmentservice.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Department {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    // Getters and setters
    public Long getId() { return id; }

    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }

    public void setName(String name) { this.name = name; }
}
```

---

### `DepartmentRepository.java`

```java
package com.example.departmentservice.repository;

import com.example.departmentservice.model.Department;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DepartmentRepository extends JpaRepository<Department, Long> {
}
```

---

### `EmployeeClient.java`

```java
package com.example.departmentservice.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.PathVariable;

@FeignClient(name = "employee-service", url = "http://localhost:8081")
public interface EmployeeClient {

    @DeleteMapping("/employees/by-department/{deptId}")
    void deleteEmployeesByDepartment(@PathVariable("deptId") Long deptId);
}
```

---

### `DepartmentController.java`

```java
package com.example.departmentservice.controller;

import com.example.departmentservice.client.EmployeeClient;
import com.example.departmentservice.repository.DepartmentRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/departments")
@RequiredArgsConstructor
public class DepartmentController {

    private final DepartmentRepository departmentRepository;
    private final EmployeeClient employeeClient;

    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteDepartment(@PathVariable Long id) {
        if (!departmentRepository.existsById(id)) {
            return ResponseEntity.notFound().build();
        }

        try {
            employeeClient.deleteEmployeesByDepartment(id);
            departmentRepository.deleteById(id);
            return ResponseEntity.ok("Department and related employees deleted successfully.");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Saga failed: " + e.getMessage());
        }
    }
}
```

---

### `application.properties`

```properties
server.port=8080
spring.h2.console.enabled=true
spring.datasource.url=jdbc:h2:mem:deptdb
spring.jpa.hibernate.ddl-auto=update
```

---

## 🧱 2. `employee-service`

### `Employee.java`

```java
package com.example.employeeservice.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Employee {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    private Long departmentId;

    // Getters and setters
    public Long getId() { return id; }

    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }

    public void setName(String name) { this.name = name; }

    public Long getDepartmentId() { return departmentId; }

    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
}
```

---

### `EmployeeRepository.java`

```java
package com.example.employeeservice.repository;

import com.example.employeeservice.model.Employee;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EmployeeRepository extends JpaRepository<Employee, Long> {
    void deleteByDepartmentId(Long departmentId);
}
```

---

### `EmployeeController.java`

```java
package com.example.employeeservice.controller;

import com.example.employeeservice.repository.EmployeeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/employees")
@RequiredArgsConstructor
public class EmployeeController {

    private final EmployeeRepository employeeRepository;

    @DeleteMapping("/by-department/{deptId}")
    public ResponseEntity<Void> deleteByDepartment(@PathVariable Long deptId) {
        employeeRepository.deleteByDepartmentId(deptId);
        return ResponseEntity.ok().build();
    }
}
```

---

### `application.properties`

```properties
server.port=8081
spring.h2.console.enabled=true
spring.datasource.url=jdbc:h2:mem:empdb
spring.jpa.hibernate.ddl-auto=update
```

---

## 🧩 Common Setup

### `pom.xml` for `department-service` (with Feign)

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-openfeign</artifactId>
    </dependency>
    <dependency>
        <groupId>com.h2database</groupId>
        <artifactId>h2</artifactId>
    </dependency>
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
</dependencies>
```

### `pom.xml` for `employee-service` (no Feign needed)

Same as above, just omit the Feign dependency.

---

Would you like a **Postman collection**, **Docker support**, or transition to **PostgreSQL** for persistence?

Let me know if you want this orchestrated using a message broker like Kafka or RabbitMQ for a **Choreography Saga** instead.

 The **Saga Pattern** is crucial in microservices for handling **distributed transactions**. Let's understand the **difference in real code** between:

* ✅ **With Saga** (proper compensation and coordination)
* ❌ **Without Saga** (manual, error-prone coordination)

---

## 📘 Scenario: Delete a `Department` and all its `Employees`

Assume:

* `DepartmentService` and `EmployeeService` are separate microservices
* When deleting a department, we must delete its employees first

---

## ❌ Without Saga Pattern

You write imperative code in a monolith-style or a single orchestrator, without compensation:

```java
public void deleteDepartmentAndEmployees(Long deptId) {
    // Step 1: delete employees
    restTemplate.delete("http://employee-service/employees/by-department/" + deptId);

    // Step 2: delete department
    restTemplate.delete("http://department-service/departments/" + deptId);
}
```

### ⚠️ Problem:

* If `step 1` succeeds and `step 2` fails, data is **inconsistent**
* No rollback mechanism
* Retry logic or failure handling is manual

---

## ✅ With Saga Pattern

There are two Saga approaches:

### 🔹 1. **Choreography-Based Saga** (Decentralized)

Each service listens to events and reacts.

```java
// In DepartmentService
public void deleteDepartment(Long deptId) {
    kafkaTemplate.send("department.delete", deptId);
}

// In EmployeeService
@KafkaListener(topics = "department.delete")
public void deleteEmployeesForDepartment(Long deptId) {
    employeeRepository.deleteByDepartmentId(deptId);
    kafkaTemplate.send("department.delete.confirm", deptId);
}
```

Then `DepartmentService` only deletes the department **after** confirming:

```java
@KafkaListener(topics = "department.delete.confirm")
public void proceedToDelete(Long deptId) {
    departmentRepository.deleteById(deptId);
}
```

* Services are **loosely coupled**
* Compensation can be added via reverse events (`employee.restore`, etc.)

---

### 🔹 2. **Orchestration-Based Saga** (Central Coordinator)

A central **Saga Orchestrator** calls services step-by-step and handles rollback.

```java
public void orchestrateDeleteDepartment(Long deptId) {
    try {
        callEmployeeServiceDelete(deptId); // step 1
        callDepartmentServiceDelete(deptId); // step 2
    } catch (Exception e) {
        // Compensation if any step fails
        callEmployeeRestore(deptId);
    }
}
```

Each step has a **compensating action** defined:

* Delete → Restore
* Update → Undo Update
* Create → Delete

This can be built using tools like **Axon, Camunda**, or Spring-based custom code.

---

## 💡 Key Code Differences

| Aspect                | Without Saga        | With Saga                           |
| --------------------- | ------------------- | ----------------------------------- |
| Transaction Control   | Manual via code     | Defined via compensation or events  |
| Error Handling        | Try/catch + retries | Built-in compensation steps         |
| Atomicity             | Broken easily       | Maintained across services          |
| Architecture Coupling | Tightly coupled     | Loosely coupled (esp. Choreography) |
| Code Reusability      | Low                 | High (compensating logic reused)    |
| Failure Recovery      | Manual              | Automatic or semi-automated         |

---

## 🧪 Example Compensation

```java
// EmployeeService: delete
public void deleteByDepartmentId(Long deptId) {
    List<Employee> backup = employeeRepository.findByDepartmentId(deptId);
    employeeRepository.deleteByDepartmentId(deptId);

    // save backup somewhere for rollback
}

// Compensation method
public void restoreEmployees(List<Employee> backup) {
    employeeRepository.saveAll(backup);
}
```

---

## ✅ When to Use Saga

* Microservices must **coordinate updates/deletes across services**
* **Distributed transaction is not possible** (2PC is avoided)
* **Failure handling and consistency** are important

---

Would you like a complete **Spring Boot project** example for:

* 🔁 Saga with Kafka (Choreography)?
* 🧭 Saga with Orchestrator service (REST-based or Kafka-based)?

Let me know!
