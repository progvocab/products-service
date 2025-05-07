Here’s a **complete example of a Java Spring Boot CI/CD pipeline using Buildkite**, including:

* Build and test with Gradle
* Use of `.buildkite/pipeline.yml`
* Sample `Dockerfile` (optional)
* Deployment placeholder

---

## 🧱 Project Structure

```
my-spring-boot-app/
├── .buildkite/
│   └── pipeline.yml
├── src/
│   └── main/java/...
├── build.gradle
└── Dockerfile (optional)
```

---

## 🛠️ 1. `.buildkite/pipeline.yml`

```yaml
steps:
  - label: ":white_check_mark: Code Check"
    command: ./gradlew check
    agents:
      queue: "default"

  - label: ":test_tube: Run Unit Tests"
    command: ./gradlew test
    agents:
      queue: "default"

  - wait

  - label: ":package: Build JAR"
    command: ./gradlew bootJar
    artifact_paths:
      - build/libs/*.jar
    agents:
      queue: "default"

  - wait

  - label: ":rocket: Deploy to Dev"
    command: ./scripts/deploy-dev.sh
    if: build.branch == "main"
    agents:
      queue: "default"
```

---

## ⚙️ 2. Sample `build.gradle`

Ensure you include these tasks:

```groovy
plugins {
    id 'org.springframework.boot' version '3.2.0'
    id 'java'
}

group = 'com.example'
version = '1.0.0'
sourceCompatibility = '17'

repositories {
    mavenCentral()
}

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    testImplementation 'org.springframework.boot:spring-boot-starter-test'
}

test {
    useJUnitPlatform()
}
```

---

## 🐳 3. (Optional) `Dockerfile`

```Dockerfile
FROM eclipse-temurin:17-jdk-alpine
ARG JAR_FILE=build/libs/*.jar
COPY ${JAR_FILE} app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
```

---

## 🚀 4. Deployment Script `scripts/deploy-dev.sh`

```bash
#!/bin/bash
set -e

echo "Deploying Spring Boot app to Dev environment..."
# Replace this with your actual deploy logic (e.g., kubectl, SCP, AWS CLI)
# Example:
# scp build/libs/*.jar user@server:/opt/apps/
```

> Make sure this script is executable: `chmod +x scripts/deploy-dev.sh`

---

## 🧪 Trigger the Build

Push your code to GitHub. The Buildkite pipeline will:

1. Run code checks
2. Run unit tests
3. Build the JAR
4. Deploy to Dev on the `main` branch

---

Would you like to integrate Docker build + push, or use Helm/Kubernetes as part of deployment?
