# **Spring Batch: Comprehensive Guide**  

**Spring Batch** is a **framework** for **processing large volumes of data in batch jobs**. It is designed to handle high-volume, complex batch-processing needs like reading/writing from databases, transforming data, and handling large ETL (Extract, Transform, Load) workflows.  

---

# **🔹 1. Key Features of Spring Batch**
| Feature | Description |
|---------|------------|
| **Chunk-Based Processing** | Reads a set of records, processes them, and writes in bulk. |
| **Tasklet-Based Processing** | Executes a single operation as a step (e.g., sending an email). |
| **Transaction Management** | Ensures batch processing is atomic and can be rolled back. |
| **Restart & Recovery** | Supports restartability in case of failures. |
| **Parallel & Multi-threaded Execution** | Supports concurrent batch job execution. |
| **Job Scheduling & Monitoring** | Works with **Spring Scheduler, Quartz, or external schedulers**. |

---

# **🔹 2. Spring Batch Architecture**
A **Spring Batch Job** consists of **Steps**, and each step follows this pattern:

1. **Job** → A batch process that contains multiple **Steps**.  
2. **Step** → A stage in a job (e.g., reading, processing, writing).  
3. **ItemReader** → Reads data from a source (database, file, API).  
4. **ItemProcessor** → Processes and transforms the data.  
5. **ItemWriter** → Writes data to a target (database, file, API).  

🔹 **Example Flow:** **Read → Process → Write**  

---

# **🔹 3. Creating a Spring Batch Job**
### **✅ Step 1: Add Dependencies**
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-batch</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<dependency>
    <groupId>org.h2database</groupId>
    <artifactId>h2</artifactId>
    <scope>runtime</scope>
</dependency>
```
Spring Batch **automatically configures an embedded H2 database** for storing job metadata.

---

### **✅ Step 2: Define a Batch Job**
```java
import org.springframework.batch.core.Job;
import org.springframework.batch.core.Step;
import org.springframework.batch.core.configuration.annotation.JobBuilderFactory;
import org.springframework.batch.core.configuration.annotation.StepBuilderFactory;
import org.springframework.batch.core.launch.support.RunIdIncrementer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class BatchConfig {

    @Bean
    public Job job(JobBuilderFactory jobBuilderFactory, Step step) {
        return jobBuilderFactory.get("employeeJob")
                .incrementer(new RunIdIncrementer()) // Ensures unique job instance
                .start(step) // Start the step
                .build();
    }

    @Bean
    public Step step(StepBuilderFactory stepBuilderFactory, EmployeeProcessor processor) {
        return stepBuilderFactory.get("step")
                .<Employee, Employee>chunk(5) // Process in chunks of 5
                .reader(new EmployeeReader())
                .processor(processor)
                .writer(new EmployeeWriter())
                .build();
    }
}
```
🔹 **Chunk-Based Processing:** Reads **5 records at a time**, processes them, and writes to a database.

---

### **✅ Step 3: Create an ItemReader**
📌 Reads **data** from a CSV file, database, or REST API.  
```java
import org.springframework.batch.item.ItemReader;
import java.util.Arrays;
import java.util.Iterator;

public class EmployeeReader implements ItemReader<Employee> {
    private final Iterator<Employee> employeeIterator = Arrays.asList(
            new Employee(1, "Alice"), new Employee(2, "Bob"), new Employee(3, "Charlie")
    ).iterator();

    @Override
    public Employee read() {
        return employeeIterator.hasNext() ? employeeIterator.next() : null;
    }
}
```
🔹 **Reads employees one by one** until the list is empty.

---

### **✅ Step 4: Create an ItemProcessor**
📌 **Processes data (e.g., transforms, filters, enriches).**  
```java
import org.springframework.batch.item.ItemProcessor;

public class EmployeeProcessor implements ItemProcessor<Employee, Employee> {
    @Override
    public Employee process(Employee employee) {
        employee.setName(employee.getName().toUpperCase()); // Convert name to uppercase
        return employee;
    }
}
```
🔹 **Transforms Employee names to uppercase before writing.**

---

### **✅ Step 5: Create an ItemWriter**
📌 **Writes the processed data to a database or file.**  
```java
import org.springframework.batch.item.ItemWriter;
import java.util.List;

public class EmployeeWriter implements ItemWriter<Employee> {
    @Override
    public void write(List<? extends Employee> employees) {
        employees.forEach(System.out::println); // Simulate saving to a database
    }
}
```
🔹 **Writes employees to the console (can be replaced with a DB writer).**  

---

# **🔹 4. Running a Spring Batch Job**
Spring Batch automatically **detects jobs and runs them** on application startup.

### **✅ To Run the Job Manually:**
```java
import org.springframework.batch.core.Job;
import org.springframework.batch.core.JobExecution;
import org.springframework.batch.core.JobLauncher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class JobRunner implements CommandLineRunner {
    @Autowired private JobLauncher jobLauncher;
    @Autowired private Job job;

    @Override
    public void run(String... args) throws Exception {
        JobExecution execution = jobLauncher.run(job, new JobParameters());
        System.out.println("Job Execution Status: " + execution.getStatus());
    }
}
```
🔹 Runs the batch job when the application starts.  

---

# **🔹 5. Spring Batch with Database (JPA)**
📌 **Spring Batch can directly read and write from a relational database.**  

### **✅ Reading from a Database (JPA Reader)**
```java
@Bean
public ItemReader<Employee> databaseReader(EntityManagerFactory entityManagerFactory) {
    return new JpaPagingItemReaderBuilder<Employee>()
            .name("employeeReader")
            .entityManagerFactory(entityManagerFactory)
            .queryString("SELECT e FROM Employee e")
            .pageSize(10)
            .build();
}
```
🔹 **Fetches data from the database in pages of 10 records.**  

### **✅ Writing to a Database (JPA Writer)**
```java
@Bean
public ItemWriter<Employee> databaseWriter(EntityManagerFactory entityManagerFactory) {
    return new JpaItemWriterBuilder<Employee>()
            .entityManagerFactory(entityManagerFactory)
            .build();
}
```
🔹 **Writes processed data back to the database.**  

---

# **🔹 6. Parallel Processing & Scaling**
Spring Batch supports **multi-threaded processing** for high performance.

### **✅ Multi-threaded Step**
```java
@Bean
public TaskExecutor taskExecutor() {
    ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
    executor.setCorePoolSize(4); // 4 parallel threads
    executor.setMaxPoolSize(8);
    executor.setQueueCapacity(10);
    executor.initialize();
    return executor;
}

@Bean
public Step parallelStep(StepBuilderFactory stepBuilderFactory) {
    return stepBuilderFactory.get("parallelStep")
            .<Employee, Employee>chunk(10)
            .reader(new EmployeeReader())
            .processor(new EmployeeProcessor())
            .writer(new EmployeeWriter())
            .taskExecutor(taskExecutor()) // Enable parallel execution
            .build();
}
```
🔹 **Processes multiple records in parallel using a thread pool.**  

---

# **🔹 7. Best Practices**
✅ **Use `chunk()` instead of `tasklet()`** for large data processing.  
✅ **Enable retry & error handling** using `.faultTolerant()` in steps.  
✅ **Use database-backed job repository** for production.  
✅ **Enable parallel processing** for large-scale workloads.  

---

# **🔹 Summary**
| Feature | Description |
|---------|------------|
| **ItemReader** | Reads data from a source (DB, CSV, API). |
| **ItemProcessor** | Processes or transforms data. |
| **ItemWriter** | Writes data to a target (DB, file). |
| **Chunk-Based Processing** | Processes data in batches for efficiency. |
| **Parallel Processing** | Uses multiple threads for speed. |

---

# **🚀 Do you need a full working Spring Batch project?**