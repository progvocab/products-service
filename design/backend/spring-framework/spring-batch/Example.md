Here’s a complete example of a Spring Batch microservice that:

1. Uploads a CSV file,
2. Reads employee data,
3. Processes each line,
4. Sends a POST request to an employee API,
5. Uses `ItemReader`, `ItemProcessor`, and `ItemWriter`.

### 1. CSV File Sample (`employees.csv`)

```
name,email,department
John Doe,john@example.com,Engineering
Jane Smith,jane@example.com,Marketing
```

---

### 2. Spring Batch Configuration

```java
@Configuration
@EnableBatchProcessing
public class BatchConfig {

    @Autowired private JobBuilderFactory jobBuilderFactory;
    @Autowired private StepBuilderFactory stepBuilderFactory;
    @Autowired private RestTemplate restTemplate;

    @Bean
    public Job employeeJob() {
        return jobBuilderFactory.get("employeeJob")
                .start(employeeStep())
                .build();
    }

    @Bean
    public Step employeeStep() {
        return stepBuilderFactory.get("employeeStep")
                .<EmployeeInput, EmployeeInput>chunk(10)
                .reader(fileItemReader(null))
                .processor(employeeProcessor())
                .writer(employeeWriter())
                .build();
    }

    @Bean
    @StepScope
    public FlatFileItemReader<EmployeeInput> fileItemReader(@Value("#{jobParameters['filePath']}") String path) {
        FlatFileItemReader<EmployeeInput> reader = new FlatFileItemReader<>();
        reader.setResource(new FileSystemResource(path));
        reader.setLinesToSkip(1);
        reader.setLineMapper((line, lineNumber) -> {
            String[] fields = line.split(",");
            return new EmployeeInput(fields[0], fields[1], fields[2]);
        });
        return reader;
    }

    @Bean
    public ItemProcessor<EmployeeInput, EmployeeInput> employeeProcessor() {
        return item -> {
            // Add validation or transformation logic here
            return item;
        };
    }

    @Bean
    public ItemWriter<EmployeeInput> employeeWriter() {
        return items -> {
            for (EmployeeInput emp : items) {
                restTemplate.postForEntity("http://localhost:8081/employees", emp, Void.class);
            }
        };
    }

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

---

### 3. Domain Class

```java
@Data
@AllArgsConstructor
@NoArgsConstructor
public class EmployeeInput {
    private String name;
    private String email;
    private String department;
}
```

---

### 4. Triggering the Job

Use Spring Boot controller to upload the file and trigger the batch job:

```java
@RestController
@RequestMapping("/upload")
public class FileUploadController {

    @Autowired private JobLauncher jobLauncher;
    @Autowired private Job employeeJob;

    @PostMapping
    public String upload(@RequestParam("file") MultipartFile file) throws Exception {
        String path = "uploads/" + file.getOriginalFilename();
        file.transferTo(new File(path));

        JobParameters params = new JobParametersBuilder()
                .addString("filePath", path)
                .addLong("time", System.currentTimeMillis())
                .toJobParameters();

        jobLauncher.run(employeeJob, params);
        return "Batch job triggered";
    }
}
```

---

### 5. Auto-generated Tables

Spring Batch auto-creates the following tables (if `JdbcJobRepository` is used):

- `BATCH_JOB_INSTANCE`
- `BATCH_JOB_EXECUTION`
- `BATCH_JOB_EXECUTION_PARAMS`
- `BATCH_STEP_EXECUTION`
- `BATCH_STEP_EXECUTION_CONTEXT`
- `BATCH_JOB_EXECUTION_CONTEXT`

Make sure your schema is initialized from the Spring Batch SQL files (available in the Spring Batch GitHub repo or starter JAR).

---

Let me know if you want to use in-memory DB like H2 or a persistent DB, or if you'd like Docker/Helm setup for this.


Here's a `curl` command to call the `/upload` endpoint and upload the `employees.csv` file:

```bash
curl -X POST http://localhost:8080/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/employees.csv"
```

**Replace** `/path/to/employees.csv` with the actual path to your CSV file.

Example:

```bash
curl -X POST http://localhost:8080/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@employees.csv"
```

Let me know if your Spring Boot app is running on a different port or behind a proxy like Nginx.