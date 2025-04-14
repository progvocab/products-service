Great use case! Let's walk through a complete solution:

---

## 🧾 **Goal**:
1. Create an **Oracle stored procedure** that aggregates daily hours spent in the office by employees.
2. Allow filtering by **user ID** or **department ID**.
3. Call it from a **Spring Boot microservice** and return the result.

---

## ✅ Step 1: Oracle Table Assumption

Let's assume your tracking table is like this:

```sql
CREATE TABLE employee_logs (
    log_id        NUMBER PRIMARY KEY,
    employee_id   NUMBER,
    department_id NUMBER,
    entry_time    TIMESTAMP,
    exit_time     TIMESTAMP
);
```

---

## ✅ Step 2: Oracle Procedure – `GET_DAILY_HOURS`

This procedure will return a cursor with employee ID, log date, and total hours.

```sql
CREATE OR REPLACE PROCEDURE GET_DAILY_HOURS (
    p_user_id       IN NUMBER DEFAULT NULL,
    p_dept_id       IN NUMBER DEFAULT NULL,
    p_result_cursor OUT SYS_REFCURSOR
)
AS
BEGIN
    OPEN p_result_cursor FOR
        SELECT
            employee_id,
            TRUNC(entry_time) AS log_date,
            ROUND(SUM((CAST(exit_time AS DATE) - CAST(entry_time AS DATE)) * 24), 2) AS hours_spent
        FROM
            employee_logs
        WHERE
            (p_user_id IS NULL OR employee_id = p_user_id)
            AND (p_dept_id IS NULL OR department_id = p_dept_id)
        GROUP BY
            employee_id,
            TRUNC(entry_time)
        ORDER BY
            employee_id, log_date;
END;
```

---

## ✅ Step 3: Spring Boot Integration

### 1. **Maven Dependency**

Ensure you have the Oracle JDBC driver:

```xml
<dependency>
    <groupId>com.oracle.database.jdbc</groupId>
    <artifactId>ojdbc8</artifactId>
    <version>19.3.0.0</version>
</dependency>
```

---

### 2. **Java DTO**

```java
public class DailyHoursDTO {
    private Long employeeId;
    private LocalDate logDate;
    private Double hoursSpent;
    
    // Getters and Setters
}
```

---

### 3. **Repository Using JdbcTemplate**

```java
@Repository
public class EmployeeLogRepository {

    @Autowired
    private DataSource dataSource;

    public List<DailyHoursDTO> getDailyHours(Long userId, Long deptId) {
        List<DailyHoursDTO> result = new ArrayList<>();

        try (Connection conn = dataSource.getConnection();
             CallableStatement stmt = conn.prepareCall("{ call GET_DAILY_HOURS(?, ?, ?) }")) {

            if (userId != null) {
                stmt.setLong(1, userId);
            } else {
                stmt.setNull(1, Types.NUMERIC);
            }

            if (deptId != null) {
                stmt.setLong(2, deptId);
            } else {
                stmt.setNull(2, Types.NUMERIC);
            }

            stmt.registerOutParameter(3, OracleTypes.CURSOR);
            stmt.execute();

            ResultSet rs = (ResultSet) stmt.getObject(3);

            while (rs.next()) {
                DailyHoursDTO dto = new DailyHoursDTO();
                dto.setEmployeeId(rs.getLong("employee_id"));
                dto.setLogDate(rs.getDate("log_date").toLocalDate());
                dto.setHoursSpent(rs.getDouble("hours_spent"));
                result.add(dto);
            }

            rs.close();
        } catch (SQLException e) {
            throw new RuntimeException("Failed to fetch daily hours", e);
        }

        return result;
    }
}
```

---

### 4. **Service Layer**

```java
@Service
public class EmployeeLogService {

    @Autowired
    private EmployeeLogRepository repository;

    public List<DailyHoursDTO> getDailyHours(Long userId, Long deptId) {
        return repository.getDailyHours(userId, deptId);
    }
}
```

---

### 5. **Controller**

```java
@RestController
@RequestMapping("/api/logs")
public class EmployeeLogController {

    @Autowired
    private EmployeeLogService service;

    @GetMapping("/daily-hours")
    public List<DailyHoursDTO> getDailyHours(
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) Long deptId) {
        return service.getDailyHours(userId, deptId);
    }
}
```

---

### ✅ Sample API Usage

- **By User**: `/api/logs/daily-hours?userId=101`
- **By Department**: `/api/logs/daily-hours?deptId=5`
- **All Data**: `/api/logs/daily-hours`

---

Would you like to include Swagger docs, JUnit tests, or build this into a Helm-deployable app on EKS too?
