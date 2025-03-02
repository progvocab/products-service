### **Detecting Anomalies in Employee Office Hours & Auto-Creating Cases**  

Your system will now:  
1. **Detect Anomalies**: Identify unusual time spent in the office.  
2. **Create Cases Automatically**: If an anomaly is found, log a case.  
3. **Expose APIs**: View active anomaly cases.  

---

## **1. Architecture Changes**  
- **AWS SageMaker** → Train an anomaly detection model.  
- **Kafka Consumer / AWS Lambda** → Trigger anomaly checks.  
- **Spring Boot API** → Expose detected anomalies & cases.  
- **MongoDB / Redshift** → Store active cases.  

---

## **2. Step-by-Step Implementation**  

### **Step 1: Train an Anomaly Detection Model in SageMaker**  

```python
from sagemaker import Session
from sagemaker.sklearn import SKLearn

sagemaker_session = Session()
role = "arn:aws:iam::your-account-id:role/SageMakerRole"

sklearn_estimator = SKLearn(
    entry_point="train_anomaly.py",
    role=role,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session
)

sklearn_estimator.fit("s3://your-datalake-bucket/office_hours_data/")
```
✅ **Trains an anomaly detection model on employee office hours.**  

---

### **Step 2: Deploy the Model & Check for Anomalies**  

Modify the Kafka Consumer to check for anomalies:  

```java
@Service
public class AnomalyDetectionService {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private CaseManagementService caseManagementService;

    private final String modelEndpoint = "https://your-sagemaker-endpoint.amazonaws.com";

    public void checkForAnomaly(String employeeId, long duration) {
        String payload = "{\"duration\": " + duration + "}";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        ResponseEntity<String> response = restTemplate.postForEntity(modelEndpoint, new HttpEntity<>(payload, headers), String.class);
        
        if (Boolean.parseBoolean(response.getBody())) {
            caseManagementService.createCase(employeeId, duration);
        }
    }
}
```
✅ **Calls SageMaker to detect anomalies & triggers case creation.**  

---

### **Step 3: Automatically Create Cases in MongoDB**  

```java
@Service
public class CaseManagementService {

    @Autowired
    private MongoTemplate mongoTemplate;

    public void createCase(String employeeId, long duration) {
        CaseRecord caseRecord = new CaseRecord(employeeId, duration, LocalDateTime.now(), "OPEN");
        mongoTemplate.save(caseRecord);
    }
}
```
✅ **Stores anomaly cases in MongoDB.**  

---

### **Step 4: API to View Active Cases**  

```java
@RestController
@RequestMapping("/cases")
public class CaseController {

    @Autowired
    private MongoTemplate mongoTemplate;

    @GetMapping("/open")
    public List<CaseRecord> getOpenCases() {
        Query query = new Query(Criteria.where("status").is("OPEN"));
        return mongoTemplate.find(query, CaseRecord.class);
    }
}
```
✅ **Supports `GET /cases/open` to view active anomaly cases.**  

---

## **3. Summary**  
- **Uses SageMaker for anomaly detection.**  
- **Spring Boot service checks anomalies & logs cases.**  
- **MongoDB stores active anomaly cases.**  
- **APIs provide access to open cases.**  

Would you like **alerts (email/SMS) when an anomaly is detected?**