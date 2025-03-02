### **KPI Aggregation & Time Series Data Pipeline**  
Your system will now:  
1. **Aggregate Data**: Compute **10-minute & 30-minute** aggregates.  
2. **Store Time-Series Data**: Use **Redis (if required) for caching**.  
3. **Expose REST APIs**: Spring Boot microservices provide access to KPIs.  

---

## **1. Aggregating Data with AWS Glue or Spark (EMR)**  
You can use **AWS Glue or EMR (Spark) to compute KPI aggregates**.

### **Spark Job for KPI Calculation (`kpi_aggregator.py`)**  
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, avg, count

spark = SparkSession.builder.appName("KPIAggregation").getOrCreate()

# Load Data from S3 Data Lake
df = spark.read.parquet("s3://your-datalake-bucket/cleaned_data/")

# Convert timestamp column
df = df.withColumn("timestamp", df["timestamp"].cast("timestamp"))

# Compute 10-min and 30-min aggregates
kpi_10min = df.groupBy(window(df.timestamp, "10 minutes"), "device_id").agg(avg("access_count").alias("avg_access"))
kpi_30min = df.groupBy(window(df.timestamp, "30 minutes"), "device_id").agg(count("user_id").alias("total_users"))

# Store in S3
kpi_10min.write.mode("overwrite").parquet("s3://your-processed-bucket/kpi_10min/")
kpi_30min.write.mode("overwrite").parquet("s3://your-processed-bucket/kpi_30min/")
```
✅ **Calculates averages & counts for each device at 10-min & 30-min intervals**.  
✅ **Stores aggregated data in S3 processed bucket**.  

---

## **2. Caching Time-Series Data in Redis**  
To optimize real-time API access, we **cache results in Redis**.

### **Spring Boot Service to Cache KPI Data**  
```java
@Service
public class KPIService {

    @Autowired
    private StringRedisTemplate redisTemplate;

    private final AmazonS3 s3Client;

    public KPIService() {
        this.s3Client = AmazonS3ClientBuilder.standard().withRegion(Regions.US_EAST_1).build();
    }

    public String getKPI(String deviceId, String interval) {
        String cacheKey = "kpi:" + deviceId + ":" + interval;
        String cachedResult = redisTemplate.opsForValue().get(cacheKey);

        if (cachedResult != null) {
            return cachedResult; // Return from cache
        }

        // Load from S3 if not in cache
        S3Object object = s3Client.getObject("your-processed-bucket", "kpi_" + interval + "/" + deviceId + ".json");
        String data = IOUtils.toString(object.getObjectContent(), StandardCharsets.UTF_8);

        // Store in Redis for faster access next time
        redisTemplate.opsForValue().set(cacheKey, data, 10, TimeUnit.MINUTES);
        return data;
    }
}
```
✅ **Checks Redis first** for cached KPIs.  
✅ **Loads from S3 if not cached & stores in Redis for 10 mins**.  

---

## **3. Spring Boot REST API to Access KPIs**  
Expose **KPI data via API**.

```java
@RestController
@RequestMapping("/kpi")
public class KPIController {

    @Autowired
    private KPIService kpiService;

    @GetMapping("/{deviceId}/{interval}")
    public ResponseEntity<String> getKPI(@PathVariable String deviceId, @PathVariable String interval) {
        return ResponseEntity.ok(kpiService.getKPI(deviceId, interval));
    }
}
```
✅ **Supports `GET /kpi/{deviceId}/{interval}`** (e.g., `/kpi/device123/10min`).  
✅ **Uses Redis to serve requests quickly**.  

---

## **4. Summary**  
- **Compute KPIs (AWS Glue or Spark on EMR).**  
- **Store results in S3 & cache in Redis** for faster API access.  
- **Spring Boot API exposes KPIs** to mobile apps.  

Would you like help **deploying Redis in AWS (Elasticache or Kubernetes)?**