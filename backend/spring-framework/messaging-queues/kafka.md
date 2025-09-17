To **run Kafka and integrate it with a Spring Boot application**, follow these steps:

---

## 🔧 Step 1: Run Kafka (Locally with Docker)

The easiest way is to use Docker Compose:

**`docker-compose.yml`**:

```yaml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

Run it:

```bash
docker-compose up -d
```

Kafka will now be available at `localhost:9092`.

---

## ☕ Step 2: Create Spring Boot Project with Kafka

Use [Spring Initializr](https://start.spring.io/) with these dependencies:

* Spring Web
* Spring for Apache Kafka

Or use this `pom.xml` entry:

```xml
<dependency>
  <groupId>org.springframework.kafka</groupId>
  <artifactId>spring-kafka</artifactId>
</dependency>
```

---

## ⚙️ Step 3: Kafka Configuration in `application.yml` or `application.properties`

```yaml
spring:
  kafka:
    bootstrap-servers: localhost:9092
    consumer:
      group-id: my-group
      auto-offset-reset: earliest
      key-deserializer: org.apache.kafka.common.serialization.StringDeserializer
      value-deserializer: org.apache.kafka.common.serialization.StringDeserializer
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer
```

---

## 📤 Step 4: Kafka Producer

```java
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
public class KafkaProducerService {

    private final KafkaTemplate<String, String> kafkaTemplate;

    public KafkaProducerService(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    public void sendMessage(String topic, String message) {
        kafkaTemplate.send(topic, message);
    }
}
```

---

## 📥 Step 5: Kafka Consumer

```java
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

@Service
public class KafkaConsumerService {

    @KafkaListener(topics = "my-topic", groupId = "my-group")
    public void listen(String message) {
        System.out.println("Received: " + message);
    }
}
```

---

## 🧪 Step 6: Send Message via REST API

```java
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/kafka")
public class KafkaController {

    private final KafkaProducerService producer;

    public KafkaController(KafkaProducerService producer) {
        this.producer = producer;
    }

    @PostMapping("/publish")
    public String sendMessage(@RequestParam String message) {
        producer.sendMessage("my-topic", message);
        return "Message sent!";
    }
}
```

---

## 🧪 Step 7: Test It

1. Start Spring Boot app.
2. Use curl or Postman:

```bash
curl -X POST "http://localhost:8080/kafka/publish?message=HelloKafka"
```

Console should log:

```
Received: HelloKafka
```

---

## ✅ Optional Tips

* Use **Avro/JSON** serializers for structured data.
* Add **Kafka Streams** or **Spring Cloud Stream** for higher-level abstraction.
* Add **Retry/Dead Letter** configuration for fault-tolerance.

Would you like me to generate this as a full working Maven or Gradle project with code files?

