### **🔹 System Design: Query Employee Daily Metrics & Display Graph on Frontend**  

This system will:  
1️⃣ **Collect employee data** (entry/exit, hours worked, etc.)  
2️⃣ **Store aggregated daily metrics** in a **data warehouse**  
3️⃣ **Expose APIs** for querying metrics  
4️⃣ **Visualize data** on the frontend using a graph  

---

## **1️⃣ System Architecture**
### **🔹 Data Flow**
1. **Data Ingestion**  
   - IoT devices + cameras generate access logs (entry/exit events).  
   - Data is streamed via **Kafka** to **AWS S3 (Raw Data Lake)**.  

2. **Data Processing & Aggregation**  
   - AWS **Glue/EMR (Spark)** processes raw logs to calculate **daily metrics** (hours worked, overtime, etc.).  
   - Aggregated data is stored in **S3 (Processed Data Lake) & Redshift**.  

3. **Query API Layer**  
   - A **Spring Boot microservice** exposes REST APIs for frontend queries.  
   - **Redis** caches recent results for performance.  

4. **Frontend Visualization**  
   - React.js/Next.js fetches data via APIs and displays **charts** using **Chart.js, D3.js, or Highcharts**.  

---

## **2️⃣ Database Schema (Redshift for OLAP Queries)**
**Table: `employee_daily_metrics`**  
```sql
CREATE TABLE employee_daily_metrics (
    employee_id VARCHAR(50),
    date DATE,
    total_hours DECIMAL(5,2),
    overtime_hours DECIMAL(5,2),
    entry_count INT,
    anomaly_detected BOOLEAN,
    PRIMARY KEY (employee_id, date)
);
```

---

## **3️⃣ Backend (Spring Boot + Redis + Redshift)**
### **🔹 API to Fetch Daily Metrics**
**Endpoint:** `GET /metrics/daily?employee_id=123&date=2025-03-08`  
```java
@RestController
@RequestMapping("/metrics")
public class EmployeeMetricsController {

    @Autowired
    private EmployeeMetricsService metricsService;

    @GetMapping("/daily")
    public ResponseEntity<EmployeeDailyMetrics> getDailyMetrics(
        @RequestParam String employee_id, 
        @RequestParam String date) {
        
        return ResponseEntity.ok(metricsService.getDailyMetrics(employee_id, date));
    }
}
```

---

### **🔹 Fetch Data from Redshift & Use Redis Caching**
```java
@Service
public class EmployeeMetricsService {

    @Autowired
    private RedisTemplate<String, EmployeeDailyMetrics> redisTemplate;
    
    @Autowired
    private JdbcTemplate jdbcTemplate;

    public EmployeeDailyMetrics getDailyMetrics(String employeeId, String date) {
        String cacheKey = "metrics:" + employeeId + ":" + date;
        
        // Check cache first
        EmployeeDailyMetrics cachedMetrics = redisTemplate.opsForValue().get(cacheKey);
        if (cachedMetrics != null) return cachedMetrics;
        
        // Fetch from Redshift
        String sql = "SELECT * FROM employee_daily_metrics WHERE employee_id = ? AND date = ?";
        EmployeeDailyMetrics metrics = jdbcTemplate.queryForObject(sql, new Object[]{employeeId, date}, new BeanPropertyRowMapper<>(EmployeeDailyMetrics.class));
        
        // Cache result for future queries
        redisTemplate.opsForValue().set(cacheKey, metrics, Duration.ofHours(1));
        
        return metrics;
    }
}
```

---

## **4️⃣ Frontend (React.js with Chart.js)**
### **🔹 Fetch API and Display Graph**
```js
import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";

const EmployeeMetricsChart = ({ employeeId }) => {
    const [metrics, setMetrics] = useState([]);

    useEffect(() => {
        fetch(`/metrics/daily?employee_id=${employeeId}&date=2025-03-08`)
            .then(res => res.json())
            .then(data => setMetrics(data));
    }, [employeeId]);

    const chartData = {
        labels: metrics.map(m => m.date),
        datasets: [
            {
                label: "Total Hours",
                data: metrics.map(m => m.total_hours),
                borderColor: "blue",
                fill: false,
            },
            {
                label: "Overtime Hours",
                data: metrics.map(m => m.overtime_hours),
                borderColor: "red",
                fill: false,
            }
        ]
    };

    return <Line data={chartData} />;
};

export default EmployeeMetricsChart;
```

---

## **5️⃣ Performance Optimization**
✅ **Redis Caching** → Speeds up queries for frequently accessed data.  
✅ **Redshift Columnar Storage** → Optimized for analytical queries.  
✅ **Partitioning in S3 & Redshift** → Use `date` and `employee_id` for fast lookups.  
✅ **API Rate Limiting** → Use **API Gateway** to prevent abuse.  

---

## **6️⃣ Summary**
| **Component**  | **Technology Used** |
|---------------|---------------------|
| **Data Ingestion** | Kafka + S3 (Raw Data Lake) |
| **Processing** | AWS Glue, EMR (Spark) |
| **Data Storage** | Redshift + S3 (Processed Data Lake) |
| **Backend API** | Spring Boot + Redis |
| **Frontend** | React.js + Chart.js |
| **Caching** | Redis for fast API responses |

🚀 **Would you like a real-time version using WebSockets for live updates?**