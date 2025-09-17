You can set up **Kafdrop** (a web UI for viewing Kafka topics, messages, brokers, etc.) using **Docker** very easily.

---

## ✅ Step-by-Step: Run Kafdrop via Docker

### 🔹 Option 1: Minimal Setup (if Kafka is already running on localhost:9092)

```bash
docker run -d \
  -p 9000:9000 \
  -e KAFKA_BROKERCONNECT=host.docker.internal:9092 \
  obsidiandynamics/kafdrop
```

* `host.docker.internal` allows the container to reach Kafka running on the host machine.
* Access Kafdrop at: [http://localhost:9000](http://localhost:9000)

---

### 🔹 Option 2: Use Docker Compose (Kafka + Zookeeper + Kafdrop)

Create a file named `docker-compose.yml`:

```yaml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  kafdrop:
    image: obsidiandynamics/kafdrop
    depends_on:
      - kafka
    ports:
      - "9000:9000"
    environment:
      KAFKA_BROKERCONNECT: kafka:9092
```

### Run it:

```bash
docker-compose up -d
```

---

## 🌐 Access Kafdrop UI

Visit:
👉 [http://localhost:9000](http://localhost:9000)

You’ll be able to:

* See Kafka topics
* Browse messages
* Inspect partitions and consumer groups

---

Let me know if you want:

* TLS / secured Kafka support
* Integration with a custom Kafka instance
* JSON message preview configuration

