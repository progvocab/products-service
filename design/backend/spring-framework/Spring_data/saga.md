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
