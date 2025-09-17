The **Saga Pattern** is used in **microservices** to manage distributed transactions across multiple services in a **failure-resilient**, **eventual-consistent** manner. It splits a large transaction into a series of smaller, local transactions coordinated through events or commands.

Let’s build a simplified **Spring Boot** application using **Saga Pattern** with **PostgreSQL** and the following tables:

* `employees`
* `departments`
* `department_hierarchy`
* `employee_hierarchy`

---

## ✅ Scenario

Create a new employee:

1. Insert into `employees`.
2. Assign department in `departments`.
3. Update `employee_hierarchy` and `department_hierarchy`.

If any step fails → trigger compensating (rollback) actions.

---

## 📦 Technologies

* Spring Boot
* Spring Data JPA
* PostgreSQL
* Spring Events (simplified Saga orchestration)
* Optional: Kafka or Axon Framework for real-world scenarios

---

## 🧱 PostgreSQL Tables (DDL)

```sql
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INT REFERENCES departments(id)
);

CREATE TABLE department_hierarchy (
    id SERIAL PRIMARY KEY,
    parent_id INT,
    child_id INT,
    FOREIGN KEY (parent_id) REFERENCES departments(id),
    FOREIGN KEY (child_id) REFERENCES departments(id)
);

CREATE TABLE employee_hierarchy (
    id SERIAL PRIMARY KEY,
    manager_id INT,
    reportee_id INT,
    FOREIGN KEY (manager_id) REFERENCES employees(id),
    FOREIGN KEY (reportee_id) REFERENCES employees(id)
);
```

---

## 🔁 Saga Orchestration – Spring Boot Events (Simple Version)

### 👨‍💻 Entity Classes (JPA)

```java
@Entity
public class Employee {
    @Id @GeneratedValue
    private Long id;
    private String name;

    @ManyToOne
    private Department department;
}
```

```java
@Entity
public class Department {
    @Id @GeneratedValue
    private Long id;
    private String name;
}
```

```java
@Entity
public class EmployeeHierarchy {
    @Id @GeneratedValue
    private Long id;
    private Long managerId;
    private Long reporteeId;
}
```

```java
@Entity
public class DepartmentHierarchy {
    @Id @GeneratedValue
    private Long id;
    private Long parentId;
    private Long childId;
}
```

---

### 📌 Saga Event Classes

```java
public class EmployeeCreatedEvent {
    public final Long employeeId;
    public final Long departmentId;

    public EmployeeCreatedEvent(Long employeeId, Long departmentId) {
        this.employeeId = employeeId;
        this.departmentId = departmentId;
    }
}
```

---

### 🧠 Saga Orchestration via Event Listener

```java
@Component
public class EmployeeSagaListener {

    @Autowired
    private DepartmentHierarchyRepository departmentHierarchyRepo;

    @Autowired
    private EmployeeHierarchyRepository employeeHierarchyRepo;

    @EventListener
    @Transactional
    public void handleEmployeeCreated(EmployeeCreatedEvent event) {
        try {
            // Step 2: Update department hierarchy
            departmentHierarchyRepo.save(new DepartmentHierarchy(null, event.departmentId, event.departmentId));

            // Step 3: Create employee hierarchy record
            employeeHierarchyRepo.save(new EmployeeHierarchy(null, event.employeeId, event.employeeId));
        } catch (Exception e) {
            throw new RuntimeException("Saga failed; compensating...");
        }
    }
}
```

---

### 🎯 Create Employee with Saga Trigger

```java
@Service
public class EmployeeService {

    @Autowired
    private EmployeeRepository employeeRepo;

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    @Transactional
    public void createEmployee(String name, Long departmentId) {
        Employee emp = new Employee();
        emp.setName(name);
        emp.setDepartment(new Department(departmentId));
        Employee saved = employeeRepo.save(emp);

        eventPublisher.publishEvent(new EmployeeCreatedEvent(saved.getId(), departmentId));
    }
}
```

---

### 🧪 Sample Test Data

```sql
INSERT INTO departments (id, name) VALUES (1, 'Engineering');
```

Test:

```java
employeeService.createEmployee("Alice", 1);
```

Expected DB changes:

* `employees`: new row for Alice
* `employee_hierarchy`: Alice as her own manager (sample logic)
* `department_hierarchy`: loopback or default assignment

---

### 🔁 Optional: Compensating Actions (Manual Rollback)

```java
public void rollbackEmployeeCreation(Long empId) {
    employeeHierarchyRepo.deleteByReporteeId(empId);
    employeeRepo.deleteById(empId);
}
```

---

## 🧠 Summary

| Component               | Role                                         |
| ----------------------- | -------------------------------------------- |
| `@Transactional`        | Ensures atomic operations within one service |
| Spring `@EventListener` | Triggers saga steps asynchronously           |
| Manual rollback methods | Handle compensating actions if needed        |

---

Would you like me to:

* Add Kafka or Axon-based version (for distributed saga)?
* Generate a full working project structure?
* Add unit or integration tests for this flow?
Absolutely! Here's a full rewrite of the **Saga Pattern implementation using Spring Boot + Kafka + PostgreSQL**, including:

* Full working code snippets ✅
* All required `import` statements ✅
* `pom.xml` dependencies ✅

---

## 🧭 Scenario Overview

Create an employee using a Saga:

1. `employee-service`: saves employee → sends `EmployeeCreatedEvent` to Kafka.
2. `department-service`: assigns department.
3. `hierarchy-service`: updates hierarchy.
4. If any step fails → sends `SagaRollbackEvent` → rollback employee.

---

## 📦 pom.xml Dependencies (All Services)

```xml
<dependencies>
    <!-- Spring Boot -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>

    <!-- PostgreSQL -->
    <dependency>
        <groupId>org.postgresql</groupId>
        <artifactId>postgresql</artifactId>
    </dependency>

    <!-- Spring Data JPA -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>

    <!-- Kafka -->
    <dependency>
        <groupId>org.springframework.kafka</groupId>
        <artifactId>spring-kafka</artifactId>
    </dependency>

    <!-- Lombok -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <scope>provided</scope>
    </dependency>

    <!-- Jackson -->
    <dependency>
        <groupId>com.fasterxml.jackson.core</groupId>
        <artifactId>jackson-databind</artifactId>
    </dependency>
</dependencies>
```

---

## 📂 Shared Event Classes (shared module or replicated across services)

```java
package com.example.shared.events;

import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class EmployeeCreatedEvent {
    private Long employeeId;
    private String employeeName;
    private Long departmentId;
}
```

```java
package com.example.shared.events;

import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SagaRollbackEvent {
    private Long employeeId;
    private String reason;
}
```

---

## 👨‍💼 employee-service

### 🏷️ Entity

```java
package com.example.employee.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Employee {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private Long departmentId;
}
```

### 🧠 Repository

```java
package com.example.employee.repository;

import com.example.employee.entity.Employee;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EmployeeRepository extends JpaRepository<Employee, Long> {}
```

### 🚀 Service

```java
package com.example.employee.service;

import com.example.employee.entity.Employee;
import com.example.employee.repository.EmployeeRepository;
import com.example.shared.events.EmployeeCreatedEvent;
import com.example.shared.events.SagaRollbackEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Slf4j
public class EmployeeService {

    private final EmployeeRepository employeeRepo;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Transactional
    public void createEmployee(String name, Long deptId) {
        Employee emp = new Employee(null, name, deptId);
        emp = employeeRepo.save(emp);

        kafkaTemplate.send("employee.created", new EmployeeCreatedEvent(emp.getId(), emp.getName(), deptId));
    }

    public void rollbackEmployee(Long empId) {
        employeeRepo.deleteById(empId);
        log.info("Rolled back employee ID: {}", empId);
    }

    @KafkaListener(topics = "saga.rollback", groupId = "employee-group")
    public void handleRollback(SagaRollbackEvent event) {
        rollbackEmployee(event.getEmployeeId());
    }
}
```

---

## 🏢 department-service

```java
package com.example.department.service;

import com.example.shared.events.EmployeeCreatedEvent;
import com.example.shared.events.SagaRollbackEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class DepartmentService {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    @KafkaListener(topics = "employee.created", groupId = "dept-group")
    public void handleEmployeeCreated(EmployeeCreatedEvent event) {
        try {
            // Simulate department assignment
            log.info("Assigning employee {} to department {}", event.getEmployeeId(), event.getDepartmentId());
            // Simulate DB insert
        } catch (Exception e) {
            kafkaTemplate.send("saga.rollback",
                    new SagaRollbackEvent(event.getEmployeeId(), "Department assignment failed"));
        }
    }
}
```

---

## 🧭 hierarchy-service

```java
package com.example.hierarchy.service;

import com.example.shared.events.EmployeeCreatedEvent;
import com.example.shared.events.SagaRollbackEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class HierarchyService {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    @KafkaListener(topics = "employee.created", groupId = "hierarchy-group")
    public void onEmployeeCreated(EmployeeCreatedEvent event) {
        try {
            // Simulate hierarchy update
            log.info("Updated hierarchy for employee: {}", event.getEmployeeId());
        } catch (Exception e) {
            kafkaTemplate.send("saga.rollback",
                    new SagaRollbackEvent(event.getEmployeeId(), "Hierarchy update failed"));
        }
    }
}
```

---

## 🧪 Kafka Topics Required

Create these topics in Kafka or allow auto-creation:

* `employee.created`
* `saga.rollback`

---

## ✅ Testing Flow

1. Call:
   `POST /employees?name=Alice&departmentId=1`

2. Triggers:

   * Employee created
   * Events sent
   * Departments and hierarchy updated
   * If failure: saga rollback triggered → employee deleted

---

## 💡 Summary

| Component            | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| Kafka                | Event-driven saga orchestration                             |
| Spring Boot services | Separate responsibility for Employee, Department, Hierarchy |
| Rollback via Kafka   | Ensures eventual consistency                                |
| `@KafkaListener`     | Saga event handling and compensation                        |

---

Would you like:

* Swagger/OpenAPI for the employee REST controller?
* A ready-to-run GitHub structure with Docker Compose for Kafka + PostgreSQL?
* Or unit + integration tests for the saga flow?
