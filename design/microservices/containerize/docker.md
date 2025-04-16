Here’s a basic but production-friendly `Dockerfile` to build a Docker image for a Spring Boot application (using a fat/uber JAR):

---

### **1. Dockerfile**

```Dockerfile
# Use an official OpenJDK runtime as a parent image
FROM eclipse-temurin:17-jdk-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the Spring Boot jar into the container
COPY target/myapp.jar app.jar

# Expose the port your app runs on
EXPOSE 8080

# Command to run the application
ENTRYPOINT ["java", "-jar", "app.jar"]
```

---

### **2. Build Your Image**

Assuming your `jar` file is built using `mvn package` or `gradle bootJar` and resides at `target/myapp.jar`, run:

```bash
docker build -t springboot-app .
```

---

### **3. Run the Container**

```bash
docker run -p 8080:8080 springboot-app
```

---

### **Optional Optimizations**

- Use **multi-stage builds** to reduce image size.
- Set `ENV JAVA_OPTS` if you want to pass JVM options dynamically.
- Use `exec form` of `ENTRYPOINT` (`["java", "-jar", "app.jar"]`) to handle signals correctly (important for graceful shutdown).

Would you like a multi-stage version or one using Jib (for Maven/Gradle integration)?