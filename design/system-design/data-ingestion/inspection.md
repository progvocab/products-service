### **Integrating Image Processing & Facial Recognition into Your Existing Pipeline**  

Your updated system will now:  
1. **Capture Images**: Cameras at entry/exit send images to an **S3 bucket**.  
2. **Detect & Recognize Faces**: Use **AWS Rekognition** or an ML model (SageMaker).  
3. **Store Annotations**: Save detection data in **S3, Redis, and Redshift**.  
4. **Expose APIs**: Spring Boot service provides access to facial recognition results.  

---

## **1. Architecture Changes**  
- **IoT Cameras** → Upload images to S3.  
- **AWS Lambda or Step Functions** → Trigger image processing.  
- **AWS Rekognition / SageMaker** → Perform facial recognition.  
- **Spring Boot API** → Query recognized people.  
- **Redis & Redshift** → Store recognition results for fast retrieval.  

---

## **2. Image Processing Workflow**  

### **Step 1: Capture & Store Images in S3**  
Configure IoT cameras to **upload images** to S3 in real-time.  

```bash
aws s3 cp entry_image.jpg s3://your-entry-bucket/
```
✅ **Images are stored securely in S3**.  

---

### **Step 2: Trigger Facial Recognition with AWS Lambda**  
An **AWS Lambda function** detects faces when a new image is uploaded.  

```python
import boto3
rekognition = boto3.client('rekognition')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    image_key = event['Records'][0]['s3']['object']['key']
    
    response = rekognition.detect_faces(
        Image={'S3Object': {'Bucket': bucket, 'Name': image_key}},
        Attributes=['ALL']
    )
    
    return response
```
✅ **Automatically detects faces on image upload**.  

---

### **Step 3: Identify the Person Using Rekognition Face Matching**  
```python
response = rekognition.search_faces_by_image(
    CollectionId='employees',
    Image={'S3Object': {'Bucket': 'your-entry-bucket', 'Name': image_key}},
    MaxFaces=1,
    FaceMatchThreshold=90
)
if response['FaceMatches']:
    person_id = response['FaceMatches'][0]['Face']['FaceId']
    print(f"Identified Person: {person_id}")
```
✅ **Compares faces to a pre-stored employee database**.  

---

### **Step 4: Store Recognition Data in Redis & Redshift**  

#### **Cache the Latest Entry in Redis**
```python
import redis
r = redis.Redis(host='your-redis-host', port=6379, db=0)

r.set(f"user:{person_id}:last_entry", timestamp)
```
✅ **Stores latest access timestamp in Redis for fast lookups**.  

---

#### **Save Entry Data to Redshift**
```sql
INSERT INTO access_logs (person_id, timestamp, image_url) 
VALUES ('abc123', '2025-02-25 10:30:00', 's3://your-entry-bucket/image123.jpg');
```
✅ **Keeps historical access records for analytics**.  

---

## **3. Spring Boot API for Access Logs**  

### **Spring Boot Service to Fetch Access Logs**
```java
@Service
public class AccessLogService {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    public List<AccessLog> getRecentAccessLogs(String personId) {
        String sql = "SELECT * FROM access_logs WHERE person_id = ? ORDER BY timestamp DESC LIMIT 10";
        return jdbcTemplate.query(sql, new Object[]{personId}, new AccessLogRowMapper());
    }
}
```
✅ **Queries Redshift for access history**.  

---

### **Spring Boot API Controller**
```java
@RestController
@RequestMapping("/access")
public class AccessLogController {

    @Autowired
    private AccessLogService accessLogService;

    @GetMapping("/{personId}")
    public ResponseEntity<List<AccessLog>> getAccessLogs(@PathVariable String personId) {
        return ResponseEntity.ok(accessLogService.getRecentAccessLogs(personId));
    }
}
```
✅ **Supports `GET /access/{personId}` to retrieve recent entries**.  

---

## **4. Summary**  
- **Cameras upload images to S3**.  
- **AWS Lambda & Rekognition detect & recognize faces**.  
- **Redis caches the last access** & **Redshift stores history**.  
- **Spring Boot API exposes access logs**.  

Would you like to **train your own face recognition model instead of using AWS Rekognition?**