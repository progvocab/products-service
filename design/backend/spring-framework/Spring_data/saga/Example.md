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
