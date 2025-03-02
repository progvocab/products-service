### **Tracking Employee Office Hours & Exposing APIs**  

Your system will now:  
1. **Calculate Time Spent**: Use entry/exit logs to compute office hours.  
2. **Real-Time Tracking**: Show who is currently inside the office.  
3. **Historical Data**: Retrieve time spent per employee over days/weeks/months.  
4. **Expose APIs**: Spring Boot endpoints provide access to real-time & historical data.  

---

## **1. Architecture Changes**  
- **Kafka & Redis** → Real-time tracking of employee presence.  
- **AWS Redshift** → Stores historical office hours.  
- **Spring Boot API** → Provides access to both real-time and historical data.  

---

## **2. Calculating Office Hours**  

### **Step 1: Compute Time Spent Per Employee**  
Modify the Kafka consumer to track entry and exit times:  

```java
@Service
public class EmployeeTrackingService {

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    public void processAccessEvent(String employeeId, String eventType, LocalDateTime timestamp) {
        String cacheKey = "employee:" + employeeId + ":last_entry";
        
        if ("ENTRY".equals(eventType)) {
            redisTemplate.opsForValue().set(cacheKey, timestamp.toString());
        } else if ("EXIT".equals(eventType)) {
            String entryTimeStr = redisTemplate.opsForValue().get(cacheKey);
            if (entryTimeStr != null) {
                LocalDateTime entryTime = LocalDateTime.parse(entryTimeStr);
                Duration duration = Duration.between(entryTime, timestamp);

                // Store the session in Redshift
                jdbcTemplate.update("INSERT INTO employee_hours (employee_id, entry_time, exit_time, duration) VALUES (?, ?, ?, ?)", 
                    employeeId, entryTime, timestamp, duration.toMinutes());

                redisTemplate.delete(cacheKey); // Remove from real-time tracking
            }
        }
    }
}
```
✅ **Tracks real-time presence using Redis.**  
✅ **Saves office hours in Redshift.**  

---

## **3. Real-Time API (Who is Inside the Office?)**  
```java
@RestController
@RequestMapping("/tracking")
public class RealTimeTrackingController {

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @GetMapping("/current")
    public Set<String> getEmployeesInsideOffice() {
        return redisTemplate.keys("employee:*:last_entry")
                .stream()
                .map(key -> key.split(":")[1])
                .collect(Collectors.toSet());
    }
}
```
✅ **Supports `GET /tracking/current` to list employees inside.**  

---

## **4. Historical API (Time Spent in Office)**  
```java
@RestController
@RequestMapping("/history")
public class EmployeeHistoryController {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @GetMapping("/{employeeId}")
    public List<EmployeeSession> getEmployeeHistory(@PathVariable String employeeId) {
        String sql = "SELECT * FROM employee_hours WHERE employee_id = ? ORDER BY entry_time DESC";
        return jdbcTemplate.query(sql, new Object[]{employeeId}, new EmployeeSessionRowMapper());
    }
}
```
✅ **Supports `GET /history/{employeeId}` for historical office hours.**  

---

## **5. Summary**  
- **Redis tracks employees inside in real-time**.  
- **Redshift stores historical office hours**.  
- **Spring Boot APIs provide real-time & historical access**.  

Would you like to **set up automatic alerts if an employee exceeds a certain number of hours?**